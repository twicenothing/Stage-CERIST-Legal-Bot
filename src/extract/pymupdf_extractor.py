import os
import re
from pathlib import Path

import fitz  # PyMuPDF


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def find_project_root() -> Path:
    """
    Finds the project root by walking upward until a folder containing 'data'
    or '.env' is found.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "data").exists() or (parent / ".env").exists():
            return parent

    # Fallback: works if script is under src/something/
    return Path(__file__).resolve().parents[2]


BASE_DIR = find_project_root()

# Input PDF folders.
# Each one can contain year folders inside it:
# data/pdf/2005/F2.pdf
# data/pdf_old/2017/F2017006.pdf
INPUT_PDF_DIRS = [
    BASE_DIR / "data" / "pdf",
    BASE_DIR / "data" / "pdf_old",
]

# New output folder for page-level TXT.
# Use a separate folder so you do not overwrite your old data/txt pipeline.
OUTPUT_TXT_DIR = Path(
    os.getenv("PAGE_TXT_DIR", str(BASE_DIR / "data" / "txt_pages"))
)

# Behavior flags
SKIP_FIRST_PAGE = True
SKIP_SOMMAIRE_PAGES = True

# Old PDFs often need slightly different column/separator handling.
LEGACY_START_YEAR = 2005
LEGACY_END_YEAR = 2018


# ==============================================================================
# CLEANING HELPERS
# ==============================================================================

def remove_arabic(text: str) -> str:
    """
    Removes Arabic characters from extracted text.
    """
    return re.sub(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+",
        "",
        text or "",
    )


def normalize_text(text: str) -> str:
    """
    Light cleanup while preserving line structure.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_sommaire_page(text_blocks) -> bool:
    """
    Detects summary/table-of-contents pages.
    Handles cases where 'sommaire' is spaced like: s o m m a i r e.
    """
    if not text_blocks:
        return False

    valid_blocks = [
        b for b in text_blocks
        if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()
    ]

    valid_blocks.sort(key=lambda b: b[1])

    header_text = " ".join([str(b[4]) for b in valid_blocks[:15]]).lower()
    header_text_clean = re.sub(r"\s+", "", header_text)

    return "sommaire" in header_text_clean


def is_ignored_title(text: str) -> bool:
    """
    Detects large section titles that should act as layout separators
    or be removed from flow.
    """
    clean_text = " ".join(str(text or "").strip().lower().split())

    exact_matches = [
        "decisions et avis",
        "décisions et avis",
        "arretes",
        "arrêtés",
        "arrêtes",
        "arretes, decisions et avis",
        "arrêtés, décisions et avis",
        "conventions et accords internationaux",
        "decrets",
        "décrets",
        "decisions individuelles",
        "décisions individuelles",
        "annonces et communications",
        "reglements",
        "règlements",
        "lois",
        "proclamations",
        "reglements interieurs",
        "règlements intérieurs",
        "arretes et proclamations",
        "arrêtés et proclamations",
        "proclamations et decisions",
        "proclamations et décisions",
        "avis",
        "avis et lois",
        "ordonnances",
        "instructions presidentielles",
        "instructions présidentielles",
        "d e c r e t s",
    ]

    return clean_text in exact_matches


# ==============================================================================
# LEGACY DETECTION
# ==============================================================================

def detect_legacy_mode(pdf_path: Path) -> bool:
    """
    Detects old-layout PDFs using year folders or filename patterns.
    Legacy years: 2005 to 2018 inclusive.
    """
    path_text = " ".join([p.lower() for p in pdf_path.parts]) + " " + pdf_path.stem.lower()

    for year in range(LEGACY_START_YEAR, LEGACY_END_YEAR + 1):
        if str(year) in path_text:
            return True

    return False


# ==============================================================================
# DOUBLE-COLUMN PAGE EXTRACTION
# ==============================================================================

def get_sorted_text_from_page(page, legacy_mode: bool = False) -> str:
    """
    Extracts readable text from a JO page.

    This keeps the old double-column logic:
    - extract text blocks
    - detect full-width separators/titles
    - process text in vertical bands
    - inside each band, read left column then right column

    IMPORTANT:
    This version intentionally does NOT detect or inject tables.
    For the full-vision pipeline, tables are handled by rendering the original PDF page.
    """
    page_width = page.rect.width
    mid_point = page_width / 2

    raw_blocks = page.get_text("blocks")
    valid_blocks = []

    for b in raw_blocks:
        if len(b) < 7:
            continue

        x0, y0, x1, y1, text, block_no, block_type = b

        if block_type != 0:
            continue

        text = str(text or "").strip()

        if not text:
            continue

        # Remove Arabic early so layout text is cleaner.
        text = remove_arabic(text).strip()

        if not text:
            continue

        valid_blocks.append((x0, y0, x1, y1, text, block_no, block_type))

    # Global vertical order
    valid_blocks.sort(key=lambda b: b[1])

    final_text = ""
    current_band_blocks = []

    def process_band(band_blocks):
        """
        Reads one horizontal band:
        left column from top to bottom, then right column from top to bottom.
        """
        if not band_blocks:
            return ""

        left_col = []
        right_col = []

        for block in band_blocks:
            block_center_x = (block[0] + block[2]) / 2

            if block_center_x < mid_point:
                left_col.append(block)
            else:
                right_col.append(block)

        left_col.sort(key=lambda x: x[1])
        right_col.sort(key=lambda x: x[1])

        band_text = ""

        for block in left_col + right_col:
            band_text += block[4].strip() + "\n"

        return band_text

    for b in valid_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b

        block_width = x1 - x0
        is_separator = False

        # Section titles and wide centered blocks act as "walls"
        # separating column zones.
        if is_ignored_title(text):
            is_separator = True

        elif block_width > (page_width * 0.75) and not any(
            k in text for k in ["ANNEXE", "ETAT ANNEXE", "ETAT ANNEXE (suite)"]
        ):
            is_separator = True

        # Legacy fix:
        # In older PDFs, some column blocks are falsely detected as wide separators.
        # If the block is entirely on one side of the page, do not treat it as a separator.
        if legacy_mode and is_separator and not is_ignored_title(text):
            if x0 > mid_point or x1 < mid_point:
                is_separator = False

        if is_separator:
            final_text += process_band(current_band_blocks)
            current_band_blocks = []

            # Ignored section titles are removed.
            # Other separators are preserved.
            if not is_ignored_title(text):
                final_text += text.strip() + "\n\n"

        else:
            current_band_blocks.append(b)

    final_text += process_band(current_band_blocks)

    return normalize_text(final_text)


# ==============================================================================
# PDF DISCOVERY
# ==============================================================================

def collect_pdf_files():
    """
    Collects all PDFs recursively from data/pdf and data/pdf_old.
    Ignores year folders; searches inside all subfolders.
    """
    pdf_files = []

    for input_dir in INPUT_PDF_DIRS:
        if not input_dir.exists():
            print(f"⚠️ Input directory not found: {input_dir}")
            continue

        for path in input_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                pdf_files.append(path)

    return sorted(pdf_files, key=lambda p: str(p).lower())


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

def process_pdf_to_page_marked_txt(pdf_path: Path, output_path: Path):
    """
    Converts one PDF into one TXT file with page markers:

    <<<PAGE_2>>>
    extracted text...

    <<<PAGE_3>>>
    extracted text...
    """
    full_doc_text = ""
    pages_written = 0
    pages_skipped_first = 0
    pages_skipped_sommaire = 0
    pages_empty = 0

    legacy_mode = detect_legacy_mode(pdf_path)

    doc = fitz.open(str(pdf_path))

    try:
        for page_index, page in enumerate(doc):
            physical_page_num = page_index + 1

            if SKIP_FIRST_PAGE and page_index == 0:
                pages_skipped_first += 1
                continue

            blocks = page.get_text("blocks")

            if SKIP_SOMMAIRE_PAGES and is_sommaire_page(blocks):
                pages_skipped_sommaire += 1
                continue

            page_text = get_sorted_text_from_page(
                page,
                legacy_mode=legacy_mode,
            )

            page_text = remove_arabic(page_text)
            page_text = normalize_text(page_text)

            if not page_text:
                pages_empty += 1
                continue

            full_doc_text += f"\n\n<<<PAGE_{physical_page_num}>>>\n"
            full_doc_text += page_text
            full_doc_text += "\n"

            pages_written += 1

    finally:
        doc.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_doc_text.strip() + "\n")

    return {
        "pages_written": pages_written,
        "pages_skipped_first": pages_skipped_first,
        "pages_skipped_sommaire": pages_skipped_sommaire,
        "pages_empty": pages_empty,
        "legacy_mode": legacy_mode,
    }


def process_all_pdfs():
    OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = collect_pdf_files()

    if not pdf_files:
        print("⚠️ No PDF files found.")
        print("Checked folders:")
        for d in INPUT_PDF_DIRS:
            print(f"  - {d}")
        return

    print("=" * 100)
    print("🚀 PAGE-MARKED PDF TEXT EXTRACTION")
    print("=" * 100)
    print(f"Project root: {BASE_DIR}")
    print("Input PDF folders:")
    for d in INPUT_PDF_DIRS:
        print(f"  - {d}")
    print(f"Output TXT folder: {OUTPUT_TXT_DIR}")
    print(f"Total PDFs found: {len(pdf_files)}")
    print("=" * 100)

    seen_output_names = set()
    total_pages_written = 0
    total_errors = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        output_name = f"{pdf_path.stem}.txt"

        # Avoid accidental overwrite if same PDF stem exists in pdf and pdf_old.
        if output_name.lower() in seen_output_names:
            parent_hint = pdf_path.parent.name
            output_name = f"{pdf_path.stem}_{parent_hint}.txt"

        seen_output_names.add(output_name.lower())

        output_path = OUTPUT_TXT_DIR / output_name

        print(f"[{index}/{len(pdf_files)}] Processing {pdf_path.name}...", end=" ")

        try:
            stats = process_pdf_to_page_marked_txt(pdf_path, output_path)
            total_pages_written += stats["pages_written"]

            print(
                f"✅ pages={stats['pages_written']} "
                f"| sommaire_skipped={stats['pages_skipped_sommaire']} "
                f"| empty={stats['pages_empty']} "
                f"| legacy={stats['legacy_mode']}"
            )

        except Exception as e:
            total_errors += 1
            print(f"❌ ERROR: {e}")

    print("=" * 100)
    print("🎉 Extraction complete.")
    print(f"TXT output folder: {OUTPUT_TXT_DIR}")
    print(f"Total pages written: {total_pages_written}")
    print(f"Errors: {total_errors}")
    print("=" * 100)


if __name__ == "__main__":
    process_all_pdfs()
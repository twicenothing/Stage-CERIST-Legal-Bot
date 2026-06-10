import os
import re
from pathlib import Path

import fitz  # PyMuPDF


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# This script is for old PDFs only.
BASE_PDF_DIR = BASE_DIR / "data" / "pdf_old"

# Output for the new full-vision pipeline.
# Same folder as the new page-based extractor.
OUTPUT_TXT_DIR = Path(
    os.getenv("PAGE_TXT_DIR", str(BASE_DIR / "data" / "txt_pages"))
)


# ==============================================================================
# CLEANING HELPERS
# ==============================================================================

def remove_arabic(text: str) -> str:
    """Supprime les caractères arabes via Regex."""
    return re.sub(
        r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+",
        "",
        text or "",
    )


def normalize_text(text: str) -> str:
    """
    Light cleanup while preserving page and line structure.
    """
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_sommaire_page(text_blocks) -> bool:
    """
    Détecte les pages de sommaire.
    Gère aussi les cas où 'sommaire' est écrit avec des espaces.
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
    Vérifie si le texte est un titre à ignorer.
    Gère les lettres espacées comme 'D E C R E T S'
    en supprimant tous les espaces.
    """
    clean_text = str(text or "").strip().lower()
    clean_text_no_spaces = re.sub(r"\s+", "", clean_text)

    exact_matches_no_spaces = [
        "decisionsetavis",
        "décisionsetavis",
        "arretes",
        "arrêtés",
        "arrêtes",
        "arretes,decisionsetavis",
        "arrêtés,décisionsetavis",
        "conventionsetaccordsinternationaux",
        "decrets",
        "décrets",
        "decisionsindividuelles",
        "décisionsindividuelles",
        "annoncesetcommunications",
        "reglements",
        "règlements",
        "lois",
        "proclamations",
        "reglementsinterieurs",
        "règlementsintérieurs",
        "arretesetproclamations",
        "arrêtésétproclamations",
        "arrêtésetproclamations",
        "proclamationsetdecisions",
        "proclamationsetdécisions",
        "avis",
        "avisetlois",
        "ordonnances",
        "instructionspresidentielles",
        "instructionsprésidentielles",
    ]

    return clean_text_no_spaces in exact_matches_no_spaces


# ==============================================================================
# SPECIAL 2003-2004 EXTRACTION LOGIC
# ==============================================================================

def get_text_2003_2004(page) -> str:
    """
    Méthode spéciale pour les anciens PDF 2003-2004.

    Kept:
    - strict geometric sorting
    - double-column reading
    - separator/wall detection
    - header removal

    Removed:
    - table detection
    - table Markdown injection
    - table JSON sidecars
    - table chunks

    In the full-vision architecture, the text is only used to find the right page.
    The final answer will come from the rendered original PDF page.
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

        # Ignore journal header.
        clean_t = text.lower()
        if y0 < 80 and (
            "journal officiel" in clean_t
            or "republique algerienne" in clean_t
            or "république algérienne" in clean_t
        ):
            continue

        text = remove_arabic(text).strip()

        if not text:
            continue

        valid_blocks.append((x0, y0, x1, y1, text, block_no, block_type))

    valid_blocks.sort(key=lambda b: b[1])

    final_text = ""
    current_band_blocks = []

    def process_band(band_blocks):
        if not band_blocks:
            return ""

        left_col = []
        right_col = []

        for block in band_blocks:
            center_x = (block[0] + block[2]) / 2

            if center_x < mid_point:
                left_col.append(block)
            else:
                right_col.append(block)

        left_col.sort(key=lambda block: block[1])
        right_col.sort(key=lambda block: block[1])

        band_text = ""

        for block in left_col:
            band_text += block[4].strip() + "\n\n"

        for block in right_col:
            band_text += block[4].strip() + "\n\n"

        return band_text

    for b in valid_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b

        block_width = x1 - x0
        is_separator = False

        if is_ignored_title(text):
            is_separator = True

        elif block_width > (page_width * 0.40):
            # If the block crosses the middle of the page, it is probably a wall/title,
            # not a normal column block.
            if x0 < (mid_point - 15) and x1 > (mid_point + 15):
                is_separator = True

        else:
            center_x = (x0 + x1) / 2

            if abs(center_x - mid_point) < (page_width * 0.1) and block_width < (page_width * 0.5):
                if "——" in text or "ETAT ANNEXE" in text.upper() or "ÉTAT ANNEXE" in text.upper():
                    is_separator = True

        if is_separator:
            final_text += process_band(current_band_blocks)
            current_band_blocks = []

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
    Collect all PDFs recursively inside data/pdf_old.
    Ignores year folders and searches all subfolders.
    """
    if not BASE_PDF_DIR.exists():
        print(f"❌ Erreur: Le dossier '{BASE_PDF_DIR}' n'existe pas.")
        return []

    pdf_files = []

    for path in BASE_PDF_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files.append(path)

    return sorted(pdf_files, key=lambda p: str(p).lower())


# ==============================================================================
# PROCESSING
# ==============================================================================

def process_pdf_to_txt(pdf_path: Path, output_path: Path):
    """
    Converts one old PDF into one TXT file with page markers:

    <<<PAGE_3>>>
    page text...

    <<<PAGE_4>>>
    page text...
    """
    full_doc_text = ""

    pages_written = 0
    pages_skipped_first = 0
    pages_skipped_sommaire = 0
    pages_empty = 0

    filename = pdf_path.name

    doc = fitz.open(str(pdf_path))

    try:
        for page_num, page in enumerate(doc):
            physical_page_num = page_num + 1

            # Special case: F2004004.pdf
            # Skip page cover + real sommaire only.
            # Do NOT call is_sommaire_page after that because the cahier des charges
            # may contain internal "sommaire" pages that should stay.
            if filename.lower() == "f2004004.pdf":
                if page_num < 2:
                    pages_skipped_first += 1
                    continue

            else:
                # General old-PDF rule: skip first page.
                if page_num == 0:
                    pages_skipped_first += 1
                    continue

                blocks = page.get_text("blocks")

                if is_sommaire_page(blocks):
                    pages_skipped_sommaire += 1
                    print(f"   🚫 {filename} - Page {physical_page_num} ignorée (Sommaire)")
                    continue

            page_text = get_text_2003_2004(page)
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
    }


def main():
    OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = collect_pdf_files()

    if not pdf_files:
        print("⚠️ Aucun PDF trouvé.")
        print(f"Dossier vérifié: {BASE_PDF_DIR}")
        return

    print("=" * 100)
    print("🚀 OLD PDF PAGE-MARKED TEXT EXTRACTION")
    print("=" * 100)
    print(f"Base dir: {BASE_DIR}")
    print(f"Input old PDF dir: {BASE_PDF_DIR}")
    print(f"Output TXT dir: {OUTPUT_TXT_DIR}")
    print(f"Total PDFs found: {len(pdf_files)}")
    print("=" * 100)

    seen_output_names = set()
    total_pages_written = 0
    total_errors = 0

    for index, pdf_path in enumerate(pdf_files, start=1):
        output_name = f"{pdf_path.stem}.txt"

        # Avoid accidental overwrite if duplicate stems exist in different year folders.
        if output_name.lower() in seen_output_names:
            parent_hint = pdf_path.parent.name
            output_name = f"{pdf_path.stem}_{parent_hint}.txt"

        seen_output_names.add(output_name.lower())

        output_path = OUTPUT_TXT_DIR / output_name

        print(f"[{index}/{len(pdf_files)}] Processing {pdf_path.name}...", end=" ")

        try:
            stats = process_pdf_to_txt(pdf_path, output_path)
            total_pages_written += stats["pages_written"]

            print(
                f"✅ pages={stats['pages_written']} "
                f"| sommaire_skipped={stats['pages_skipped_sommaire']} "
                f"| empty={stats['pages_empty']}"
            )

        except Exception as e:
            total_errors += 1
            print(f"❌ ERROR: {e}")

    print("=" * 100)
    print("🎉 Old PDF extraction complete.")
    print(f"TXT output folder: {OUTPUT_TXT_DIR}")
    print(f"Total pages written: {total_pages_written}")
    print(f"Errors: {total_errors}")
    print("=" * 100)


if __name__ == "__main__":
    main()
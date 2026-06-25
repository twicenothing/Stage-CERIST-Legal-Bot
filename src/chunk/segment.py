import os
import re
import json
from pathlib import Path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / "data").exists() or (parent / ".env").exists():
            return parent

    return Path(__file__).resolve().parents[2]


BASE_DIR = find_project_root()

INPUT_FOLDER = Path(
    os.getenv("PAGE_TXT_DIR", str(BASE_DIR / "data" / "txt_pages"))
)

OUTPUT_FOLDER = Path(
    os.getenv("PAGE_JSON_DIR", str(BASE_DIR / "data" / "json_pages"))
)

# Recursive windows inside each page.
PAGE_WINDOW_SIZE = int(os.getenv("PAGE_WINDOW_SIZE", "1500"))
PAGE_WINDOW_OVERLAP = int(os.getenv("PAGE_WINDOW_OVERLAP", "250"))

# If page is shorter than this, only page_full is created.
# This avoids duplicate full_page + identical window chunks.
MIN_PAGE_LENGTH_FOR_WINDOWS = int(os.getenv("MIN_PAGE_LENGTH_FOR_WINDOWS", "1800"))


# ==============================================================================
# CLEANING / NORMALIZATION
# ==============================================================================

def clean_text(text: str) -> str:
    """
    Light cleaning while preserving useful line structure.
    Page markers survive this.
    """
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[^\x00-\x7F\u0080-\uFFFF\n]+", " ", text)
    return text.strip()


def normalize_source_file_to_pdf(txt_filename: str) -> str:
    """
    F202009.txt -> F202009.pdf
    F202009_2005.txt -> F202009_2005.pdf
    """
    name = os.path.basename(str(txt_filename or "").strip())
    stem = os.path.splitext(name)[0]
    return f"{stem}.pdf"


def safe_id_part(value: str) -> str:
    """
    Stable ID-friendly file stem.
    """
    value = str(value or "")
    value = os.path.splitext(os.path.basename(value))[0]
    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", value)
    return value.strip("_")


# ==============================================================================
# DOCUMENT METADATA PARSING
# ==============================================================================

def parse_metadata_value(value: str):
    """
    Converts simple metadata values to useful Python types.

    journal_year: 2002 -> int
    empty values stay as ""
    """
    value = str(value or "").strip()

    if value == "":
        return ""

    if re.fullmatch(r"\d+", value):
        try:
            return int(value)
        except Exception:
            return value

    return value


def extract_document_metadata_from_txt(text: str) -> dict:
    """
    Parses the metadata header added by the extraction scripts.

    Expected format:

    <<<DOCUMENT_METADATA>>>
    source_file: F2002001.pdf
    journal_number: 01
    journal_date_text: 6 janvier 2002
    journal_date_iso: 2002-01-06
    journal_year: 2002
    <<<END_DOCUMENT_METADATA>>>
    """
    metadata = {
        "source_file": "",
        "journal_number": "",
        "journal_date_text": "",
        "journal_date_iso": "",
        "journal_year": 0,
    }

    pattern = re.compile(
        r"<<<DOCUMENT_METADATA>>>\s*(.*?)\s*<<<END_DOCUMENT_METADATA>>>",
        flags=re.DOTALL,
    )

    match = pattern.search(text or "")

    if not match:
        return metadata

    block = match.group(1)

    for line in block.splitlines():
        line = line.strip()

        if not line or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = parse_metadata_value(value)

        if key:
            metadata[key] = value

    # Safety normalization
    try:
        metadata["journal_year"] = int(metadata.get("journal_year") or 0)
    except Exception:
        metadata["journal_year"] = 0

    return metadata


def remove_document_metadata_block(text: str) -> str:
    """
    Removes the metadata header before page parsing.

    This prevents the metadata block from accidentally becoming part of a page
    if a TXT file has no page markers.
    """
    return re.sub(
        r"<<<DOCUMENT_METADATA>>>\s*.*?\s*<<<END_DOCUMENT_METADATA>>>\s*",
        "",
        text or "",
        flags=re.DOTALL,
    ).strip()


def build_common_metadata(
    source_txt: str,
    source_pdf: str,
    document_metadata: dict,
) -> dict:
    """
    Metadata copied into every chunk.
    Chroma metadata must stay simple: str, int, float, bool.
    """
    common = {
        "source_file": source_pdf,
        "source_txt": source_txt,
        "journal_number": document_metadata.get("journal_number", ""),
        "journal_date_text": document_metadata.get("journal_date_text", ""),
        "journal_date_iso": document_metadata.get("journal_date_iso", ""),
        "journal_year": document_metadata.get("journal_year", 0),
    }

    # If extractor provided source_file, keep it as an additional trace field.
    extracted_source_file = document_metadata.get("source_file", "")

    if extracted_source_file:
        common["metadata_source_file"] = extracted_source_file

    return common


# ==============================================================================
# PAGE PARSING
# ==============================================================================

def extract_pages_from_marked_txt(text: str):
    """
    Parses TXT files containing markers like:

    <<<PAGE_2>>>
    page text...

    <<<PAGE_3>>>
    page text...

    Returns:
        [{"page": 2, "text": "..."}, ...]
    """
    pattern = re.compile(r"<<<PAGE_(\d+)>>>\s*")
    matches = list(pattern.finditer(text))

    pages = []

    if not matches:
        cleaned = clean_text(text)
        if cleaned:
            pages.append({
                "page": "Inconnu",
                "text": cleaned,
            })
        return pages

    for i, match in enumerate(matches):
        page_num = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        page_text = clean_text(text[start:end])

        if page_text:
            pages.append({
                "page": page_num,
                "text": page_text,
            })

    return pages


# ==============================================================================
# RECURSIVE SPLITTER
# ==============================================================================

def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators=None,
) -> list[str]:
    """
    Pure Python recursive character chunker.

    Important:
    This is applied per page, so chunks never cross page boundaries.
    """
    if separators is None:
        separators = [
            "\nArt. ",
            "\nArticle ",
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ]

    text = clean_text(text)

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    active_separator = separators[-1]

    for sep in separators:
        if sep == "":
            active_separator = sep
            break

        if sep in text:
            active_separator = sep
            break

    if active_separator == "":
        splits = list(text)
    else:
        splits = text.split(active_separator)

    chunks = []
    current_chunk_splits = []
    current_len = 0

    for split in splits:
        if split == "":
            continue

        if len(split) > chunk_size:
            if current_chunk_splits:
                chunks.append(active_separator.join(current_chunk_splits).strip())
                current_chunk_splits = []
                current_len = 0

            try:
                next_separators = separators[separators.index(active_separator) + 1:]
            except ValueError:
                next_separators = [""]

            recursed_chunks = recursive_split(
                split,
                chunk_size,
                chunk_overlap,
                next_separators,
            )
            chunks.extend(recursed_chunks)
            continue

        sep_len = len(active_separator) if current_chunk_splits else 0

        if current_len + sep_len + len(split) > chunk_size:
            chunk = active_separator.join(current_chunk_splits).strip()

            if chunk:
                chunks.append(chunk)

            overlap_splits = []
            overlap_len = 0

            for s in reversed(current_chunk_splits):
                s_len = len(s) + (len(active_separator) if overlap_splits else 0)

                if overlap_len + s_len <= chunk_overlap:
                    overlap_splits.insert(0, s)
                    overlap_len += s_len
                else:
                    break

            current_chunk_splits = overlap_splits
            current_len = overlap_len

        current_chunk_splits.append(split)

        if len(current_chunk_splits) > 1:
            current_len += len(active_separator) + len(split)
        else:
            current_len += len(split)

    if current_chunk_splits:
        chunk = active_separator.join(current_chunk_splits).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def find_chunk_start(page_text: str, chunk_text: str, search_offset: int) -> int:
    """
    Finds chunk start inside the page text.
    Handles overlap by using a moving cursor.
    """
    idx = page_text.find(chunk_text, search_offset)

    if idx == -1:
        idx = page_text.find(chunk_text)

    if idx == -1:
        idx = search_offset

    return idx


# ==============================================================================
# CHUNK CREATION
# ==============================================================================

def build_page_full_chunk(
    source_txt: str,
    source_pdf: str,
    file_stem: str,
    page_num,
    page_text: str,
    chunk_index: int,
    document_metadata: dict,
):
    page_id = f"{file_stem}_p{page_num}"

    chunk_text = (
        f"Source: {source_pdf}\n"
        f"Page: {page_num}\n"
        f"Date du Journal Officiel: {document_metadata.get('journal_date_iso', '')}\n"
        f"Type: page complète\n\n"
        f"{page_text}"
    )

    metadata = build_common_metadata(
        source_txt=source_txt,
        source_pdf=source_pdf,
        document_metadata=document_metadata,
    )

    metadata.update({
        "page": page_num,
        "page_id": page_id,
        "chunking_method": "page_full",
        "chunk_format": "full_page_text",
        "char_start": 0,
        "char_end": len(page_text),
        "text_chars": len(page_text),
    })

    return {
        "id": f"{page_id}_full",
        "chunk_index": chunk_index,
        "text": chunk_text,
        "metadata": metadata,
    }


def build_page_window_chunks(
    source_txt: str,
    source_pdf: str,
    file_stem: str,
    page_num,
    page_text: str,
    starting_chunk_index: int,
    document_metadata: dict,
):
    page_id = f"{file_stem}_p{page_num}"

    windows = recursive_split(
        page_text,
        PAGE_WINDOW_SIZE,
        PAGE_WINDOW_OVERLAP,
    )

    chunks = []
    search_offset = 0
    chunk_index = starting_chunk_index

    for window_index, window_text in enumerate(windows, start=1):
        if not window_text.strip():
            continue

        start_idx = find_chunk_start(page_text, window_text, search_offset)
        end_idx = start_idx + len(window_text)

        search_offset = max(start_idx + 1, search_offset + 1)

        chunk_text = (
            f"Source: {source_pdf}\n"
            f"Page: {page_num}\n"
            f"Date du Journal Officiel: {document_metadata.get('journal_date_iso', '')}\n"
            f"Type: fenêtre de page\n"
            f"Fenêtre: {window_index}\n\n"
            f"{window_text}"
        )

        metadata = build_common_metadata(
            source_txt=source_txt,
            source_pdf=source_pdf,
            document_metadata=document_metadata,
        )

        metadata.update({
            "page": page_num,
            "page_id": page_id,
            "window_index": window_index,
            "chunking_method": "page_window",
            "chunk_format": "page_window_text",
            "char_start": start_idx,
            "char_end": end_idx,
            "text_chars": len(window_text),
            "window_size": PAGE_WINDOW_SIZE,
            "window_overlap": PAGE_WINDOW_OVERLAP,
        })

        chunks.append({
            "id": f"{page_id}_w{window_index}",
            "chunk_index": chunk_index,
            "text": chunk_text,
            "metadata": metadata,
        })

        chunk_index += 1

    return chunks


def process_one_txt_file(input_path: Path, output_path: Path):
    source_txt = input_path.name
    source_pdf = normalize_source_file_to_pdf(source_txt)
    file_stem = safe_id_part(source_pdf)

    with open(input_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    document_metadata = extract_document_metadata_from_txt(raw_content)

    content = remove_document_metadata_block(raw_content)
    content = clean_text(content)

    pages = extract_pages_from_marked_txt(content)

    chunks = []
    page_stats = []

    chunk_index = 1

    for page_item in pages:
        page_num = page_item["page"]
        page_text = page_item["text"]

        if not page_text.strip():
            continue

        # 1. Always create full page chunk.
        full_chunk = build_page_full_chunk(
            source_txt=source_txt,
            source_pdf=source_pdf,
            file_stem=file_stem,
            page_num=page_num,
            page_text=page_text,
            chunk_index=chunk_index,
            document_metadata=document_metadata,
        )

        chunks.append(full_chunk)
        chunk_index += 1

        windows_count = 0

        # 2. Only create page windows for longer pages.
        if len(page_text) >= MIN_PAGE_LENGTH_FOR_WINDOWS:
            window_chunks = build_page_window_chunks(
                source_txt=source_txt,
                source_pdf=source_pdf,
                file_stem=file_stem,
                page_num=page_num,
                page_text=page_text,
                starting_chunk_index=chunk_index,
                document_metadata=document_metadata,
            )

            chunks.extend(window_chunks)
            chunk_index += len(window_chunks)
            windows_count = len(window_chunks)

        page_stats.append({
            "page": page_num,
            "page_chars": len(page_text),
            "created_page_full": True,
            "created_windows": windows_count,
        })

    final_output = {
        "source_txt": source_txt,
        "source_file": source_pdf,
        "document_metadata": document_metadata,
        "journal_number": document_metadata.get("journal_number", ""),
        "journal_date_text": document_metadata.get("journal_date_text", ""),
        "journal_date_iso": document_metadata.get("journal_date_iso", ""),
        "journal_year": document_metadata.get("journal_year", 0),
        "chunking_strategy": "page_full_plus_page_window",
        "total_pages": len(pages),
        "total_chunks": len(chunks),
        "config": {
            "page_window_size": PAGE_WINDOW_SIZE,
            "page_window_overlap": PAGE_WINDOW_OVERLAP,
            "min_page_length_for_windows": MIN_PAGE_LENGTH_FOR_WINDOWS,
        },
        "page_stats": page_stats,
        "chunks": chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "page_full": sum(
            1 for c in chunks
            if c["metadata"].get("chunking_method") == "page_full"
        ),
        "page_window": sum(
            1 for c in chunks
            if c["metadata"].get("chunking_method") == "page_window"
        ),
        "journal_date_iso": document_metadata.get("journal_date_iso", ""),
        "journal_year": document_metadata.get("journal_year", 0),
    }


# ==============================================================================
# BATCH PROCESSING
# ==============================================================================

def process_all_txt_to_page_json():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    files = sorted([
        f for f in INPUT_FOLDER.iterdir()
        if f.is_file() and f.suffix.lower() == ".txt"
    ])

    if not files:
        print(f"⚠️ No .txt files found in {INPUT_FOLDER}")
        return

    print("=" * 100)
    print("🚀 PAGE-BASED CHUNKING FOR FULL VISION RAG")
    print("=" * 100)
    print(f"Input folder:  {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Files found:   {len(files)}")
    print(f"Window size:   {PAGE_WINDOW_SIZE}")
    print(f"Overlap:       {PAGE_WINDOW_OVERLAP}")
    print("=" * 100)

    total_pages = 0
    total_chunks = 0
    total_page_full = 0
    total_page_window = 0
    total_missing_dates = 0
    errors = 0

    for index, input_path in enumerate(files, start=1):
        output_filename = input_path.stem + "_pages.json"
        output_path = OUTPUT_FOLDER / output_filename

        print(f"[{index}/{len(files)}] Processing {input_path.name}...", end=" ")

        try:
            stats = process_one_txt_file(input_path, output_path)

            total_pages += stats["pages"]
            total_chunks += stats["chunks"]
            total_page_full += stats["page_full"]
            total_page_window += stats["page_window"]

            if not stats.get("journal_date_iso"):
                total_missing_dates += 1

            print(
                f"✅ pages={stats['pages']} "
                f"| chunks={stats['chunks']} "
                f"| full={stats['page_full']} "
                f"| windows={stats['page_window']} "
                f"| date={stats.get('journal_date_iso') or 'NOT_FOUND'}"
            )

        except Exception as e:
            errors += 1
            print(f"❌ ERROR: {e}")

    print("=" * 100)
    print("🎉 Page chunking complete.")
    print(f"Total pages:        {total_pages}")
    print(f"Total chunks:       {total_chunks}")
    print(f"Page full chunks:   {total_page_full}")
    print(f"Page window chunks: {total_page_window}")
    print(f"Missing dates:      {total_missing_dates}")
    print(f"Errors:             {errors}")
    print(f"Output folder:      {OUTPUT_FOLDER}")
    print("=" * 100)


if __name__ == "__main__":
    process_all_txt_to_page_json()
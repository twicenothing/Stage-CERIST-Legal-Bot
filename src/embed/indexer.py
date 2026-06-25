import os
import json
from pathlib import Path

import torch
import chromadb
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ==============================================================================
# PROJECT ROOT / ENV
# ==============================================================================

def find_project_root() -> Path:
    """
    Find the real project root, not src/embed.

    Expected project root contains things like:
    - requirements.txt
    - backend/
    - src/
    - data/
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (
            (parent / "requirements.txt").exists()
            and (parent / "backend").exists()
            and (parent / "src").exists()
        ):
            return parent

    for parent in current.parents:
        if (parent / ".git").exists():
            return parent

    for parent in current.parents:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent

    return Path(__file__).resolve().parents[2]


BASE_DIR = find_project_root()

# Load .env from the real root
load_dotenv(BASE_DIR / ".env")


def resolve_project_path(env_name: str, default_relative_path: str) -> Path:
    """
    Allows:
      PAGE_JSON_DIR=data/json_pages
    or:
      PAGE_JSON_DIR=/absolute/path/to/json_pages
    """
    value = os.getenv(env_name)

    if value:
        path = Path(value)
        if path.is_absolute():
            return path
        return BASE_DIR / path

    return BASE_DIR / default_relative_path


# ==============================================================================
# CONFIGURATION
# ==============================================================================

PAGE_JSON_DIR = resolve_project_path(
    "PAGE_JSON_DIR",
    "data/json_pages",
)

CHROMA_PATH = resolve_project_path(
    "CHROMA_PATH",
    "data/chroma_db",
)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

ADD_BATCH_SIZE = int(os.getenv("ADD_BATCH_SIZE", "512"))
ENCODE_BATCH_SIZE = int(os.getenv("PAGE_ENCODE_BATCH_SIZE", "8"))
MODEL_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "8192"))

RESET_COLLECTION = os.getenv("RESET_CHROMA_COLLECTION", "true").lower() in {
    "1",
    "true",
    "yes",
    "y",
}


# ==============================================================================
# METADATA HELPERS
# ==============================================================================

def safe_metadata_value(value):
    """
    Chroma metadata values must be:
    str, int, float, bool.

    No None, lists, or dicts.
    """
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def clean_metadata(metadata):
    """
    Ensures all metadata values are Chroma-compatible.
    """
    return {
        str(k): safe_metadata_value(v)
        for k, v in (metadata or {}).items()
    }


def normalize_source_file(source_file: str) -> str:
    """
    Ensures source_file always points to a PDF filename.
    """
    source_file = str(source_file or "").strip()

    if not source_file:
        return ""

    base = os.path.basename(source_file)

    if base.lower().endswith(".txt"):
        base = os.path.splitext(base)[0] + ".pdf"

    if base.lower().endswith(".json"):
        base = os.path.splitext(base)[0] + ".pdf"

    if base.lower().endswith("_pages.pdf"):
        base = base.replace("_pages.pdf", ".pdf")

    if base.lower().endswith("_recursive.pdf"):
        base = base.replace("_recursive.pdf", ".pdf")

    if not base.lower().endswith(".pdf"):
        base += ".pdf"

    return base


def make_safe_chroma_id(raw_id: str) -> str:
    raw_id = str(raw_id or "").strip()
    raw_id = raw_id.replace(" ", "_")
    raw_id = raw_id.replace("/", "_")
    raw_id = raw_id.replace("\\", "_")
    return raw_id


def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default

        return int(value)
    except Exception:
        return default


def normalize_document_metadata(data: dict) -> dict:
    """
    Reads document-level metadata from the page JSON.

    Your updated chunking JSON contains:
      document_metadata
      journal_number
      journal_date_text
      journal_date_iso
      journal_year

    This function normalizes them and gives safe defaults.
    """
    data = data or {}

    document_metadata = data.get("document_metadata", {}) or {}

    journal_number = (
        document_metadata.get("journal_number")
        or data.get("journal_number")
        or ""
    )

    journal_date_text = (
        document_metadata.get("journal_date_text")
        or data.get("journal_date_text")
        or ""
    )

    journal_date_iso = (
        document_metadata.get("journal_date_iso")
        or data.get("journal_date_iso")
        or ""
    )

    journal_year = safe_int(
        document_metadata.get("journal_year")
        or data.get("journal_year")
        or 0
    )

    metadata_source_file = (
        document_metadata.get("source_file")
        or data.get("source_file")
        or ""
    )

    return {
        "journal_number": str(journal_number or ""),
        "journal_date_text": str(journal_date_text or ""),
        "journal_date_iso": str(journal_date_iso or ""),
        "journal_year": journal_year,

        # Aliases useful for temporal reranking.
        "publication_date": str(journal_date_iso or ""),
        "publication_year": journal_year,
        "source_year": journal_year,

        # Trace of what the extractor found in <<<DOCUMENT_METADATA>>>.
        "metadata_source_file": normalize_source_file(metadata_source_file),
    }


def apply_date_metadata_to_chunk(metadata: dict, document_metadata: dict) -> dict:
    """
    Ensures each chunk metadata contains the publication date fields.

    Priority:
    1. chunk metadata if already present
    2. top-level document metadata
    3. safe defaults
    """
    metadata = dict(metadata or {})

    metadata["journal_number"] = str(
        metadata.get("journal_number")
        or document_metadata.get("journal_number")
        or ""
    )

    metadata["journal_date_text"] = str(
        metadata.get("journal_date_text")
        or document_metadata.get("journal_date_text")
        or ""
    )

    metadata["journal_date_iso"] = str(
        metadata.get("journal_date_iso")
        or document_metadata.get("journal_date_iso")
        or ""
    )

    journal_year = safe_int(
        metadata.get("journal_year")
        or document_metadata.get("journal_year")
        or 0
    )

    metadata["journal_year"] = journal_year

    # Useful aliases for temporal reranking.
    metadata["publication_date"] = str(
        metadata.get("publication_date")
        or metadata.get("journal_date_iso")
        or ""
    )

    metadata["publication_year"] = safe_int(
        metadata.get("publication_year")
        or journal_year
    )

    metadata["source_year"] = safe_int(
        metadata.get("source_year")
        or journal_year
    )

    if not metadata.get("metadata_source_file"):
        metadata["metadata_source_file"] = document_metadata.get("metadata_source_file", "")

    return metadata


# ==============================================================================
# MODEL HELPERS
# ==============================================================================

def get_available_devices():
    if not torch.cuda.is_available():
        return []

    return [f"cuda:{i}" for i in range(torch.cuda.device_count())]


def load_embedding_model():
    print(f"🤖 Loading embedding model: {MODEL_NAME}")

    devices = get_available_devices()

    if devices:
        print(f"✅ CUDA available. Visible devices: {devices}")

        model = SentenceTransformer(
            MODEL_NAME,
            model_kwargs={
                "torch_dtype": torch.float16,
                "attn_implementation": "sdpa",
            },
        )
    else:
        print("⚠️ CUDA not available. Using CPU.")
        model = SentenceTransformer(MODEL_NAME)

    model.max_seq_length = MODEL_MAX_SEQ_LENGTH

    print(f"📏 Model max sequence length: {model.max_seq_length}")

    return model, devices


# ==============================================================================
# PAGE JSON LOADING
# ==============================================================================

def load_page_chunks_from_file(file_path: Path):
    """
    Loads JSON files generated by chunk_pages_for_vision.py.

    Expected structure:
    {
      "source_file": "F202009.pdf",
      "document_metadata": {
        "journal_number": "09",
        "journal_date_text": "6 janvier 2025",
        "journal_date_iso": "2025-01-06",
        "journal_year": 2025
      },
      "chunks": [
        {
          "id": "...",
          "text": "...",
          "metadata": {
            "source_file": "F202009.pdf",
            "page": 9,
            "page_id": "F202009_p9",
            "chunking_method": "page_full" or "page_window",
            "journal_date_iso": "2025-01-06",
            "journal_year": 2025
          }
        }
      ]
    }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "chunks" not in data:
        print(f"   ⚠️ Skipped {file_path.name}: no 'chunks' key")
        return [], [], []

    source_file = normalize_source_file(data.get("source_file", file_path.name))

    document_metadata = normalize_document_metadata(data)

    ids = []
    documents = []
    metadatas = []

    missing_date_count = 0

    for idx, chunk in enumerate(data.get("chunks", []), start=1):
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        metadata = dict(chunk.get("metadata", {}))

        metadata["source_file"] = normalize_source_file(
            metadata.get("source_file", source_file)
        )

        metadata = apply_date_metadata_to_chunk(
            metadata=metadata,
            document_metadata=document_metadata,
        )

        metadata.setdefault("source_json", file_path.name)
        metadata.setdefault("page", "Inconnu")
        metadata.setdefault("page_id", "")
        metadata.setdefault("chunking_method", "page_window")
        metadata.setdefault("chunk_format", "page_text")
        metadata.setdefault("chunk_index", chunk.get("chunk_index", idx))
        metadata.setdefault("embedding_strategy", "page_full_plus_page_window")

        if not metadata.get("journal_date_iso"):
            missing_date_count += 1

        chunk_id = chunk.get("id")

        if not chunk_id:
            page_id = metadata.get("page_id") or Path(source_file).stem
            chunk_method = metadata.get("chunking_method", "page_chunk")
            chunk_id = f"{page_id}_{chunk_method}_{idx}"

        chunk_id = make_safe_chroma_id(chunk_id)

        ids.append(chunk_id)
        documents.append(text)
        metadatas.append(clean_metadata(metadata))

    if missing_date_count:
        print(
            f"   ⚠️ {file_path.name}: {missing_date_count} chunks without journal_date_iso"
        )

    return ids, documents, metadatas


def collect_json_files():
    if not PAGE_JSON_DIR.exists():
        print(f"❌ PAGE_JSON_DIR not found: {PAGE_JSON_DIR}")
        return []

    json_files = sorted([
        p for p in PAGE_JSON_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".json"
    ])

    if not json_files:
        print(f"⚠️ No JSON files found in: {PAGE_JSON_DIR}")
        return []

    return json_files


# ==============================================================================
# CHROMA
# ==============================================================================

def create_or_reset_collection():
    print(f"🔄 Initializing ChromaDB at: {CHROMA_PATH}")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if RESET_COLLECTION:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"🗑️ Deleted old collection: {COLLECTION_NAME}")
        except Exception:
            print(f"ℹ️ No existing collection to delete: {COLLECTION_NAME}")

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print(f"📚 Using collection: {COLLECTION_NAME}")

    return collection


def add_encoded_batches(collection, model, pool, ids, documents, metadatas, label="chunks"):
    if not documents:
        return 0

    total_added = 0

    for start in range(0, len(documents), ADD_BATCH_SIZE):
        end = min(start + ADD_BATCH_SIZE, len(documents))

        batch_ids = ids[start:end]
        batch_docs = documents[start:end]
        batch_metas = metadatas[start:end]

        print(
            f"🧠 Encoding {label} batch "
            f"{start + 1}-{end}/{len(documents)}..."
        )

        if pool is not None:
            embeddings_array = model.encode(
                batch_docs,
                pool=pool,
                batch_size=ENCODE_BATCH_SIZE,
            )
        else:
            embeddings_array = model.encode(
                batch_docs,
                batch_size=ENCODE_BATCH_SIZE,
                show_progress_bar=False,
            )

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            embeddings=embeddings_array.tolist(),
            metadatas=batch_metas,
        )

        total_added += len(batch_docs)

        print(f"   ➕ Added {total_added}/{len(documents)} {label}")

    return total_added


# ==============================================================================
# MAIN INDEXING
# ==============================================================================

def main():
    print("=" * 100)
    print("🚀 PAGE-BASED EMBEDDING FOR FULL-VISION LEGAL RAG")
    print("=" * 100)
    print(f"Project root:       {BASE_DIR}")
    print(f"Input JSON folder:  {PAGE_JSON_DIR}")
    print(f"Chroma path:        {CHROMA_PATH}")
    print(f"Collection name:    {COLLECTION_NAME}")
    print(f"Embedding model:    {MODEL_NAME}")
    print(f"Reset collection:   {RESET_COLLECTION}")
    print("=" * 100)

    json_files = collect_json_files()

    if not json_files:
        print("❌ No chunks to embed. Stopping.")
        return

    print(f"📁 Found {len(json_files)} page JSON files.")

    collection = create_or_reset_collection()
    model, devices = load_embedding_model()

    pool = None
    global_total_added = 0
    global_page_full = 0
    global_page_window = 0
    global_missing_dates = 0
    errors = 0

    try:
        if len(devices) >= 2:
            print("🚀 Starting multi-GPU worker pool...")
            pool = model.start_multi_process_pool()
            print("✅ Multi-GPU pool started.")
        else:
            print("ℹ️ Using single-device encoding.")

        for index, file_path in enumerate(json_files, start=1):
            print("-" * 100)
            print(f"[{index}/{len(json_files)}] Loading {file_path.name}...")

            try:
                ids, documents, metadatas = load_page_chunks_from_file(file_path)

                if not documents:
                    print(f"   ⚠️ No valid chunks in {file_path.name}")
                    continue

                page_full_count = sum(
                    1 for m in metadatas
                    if m.get("chunking_method") == "page_full"
                )

                page_window_count = sum(
                    1 for m in metadatas
                    if m.get("chunking_method") == "page_window"
                )

                missing_date_count = sum(
                    1 for m in metadatas
                    if not m.get("journal_date_iso")
                )

                global_missing_dates += missing_date_count

                sample_date = ""
                sample_year = 0

                for m in metadatas:
                    if m.get("journal_date_iso"):
                        sample_date = m.get("journal_date_iso")
                        sample_year = m.get("journal_year", 0)
                        break

                print(
                    f"   ✅ loaded={len(documents)} "
                    f"| full={page_full_count} "
                    f"| windows={page_window_count} "
                    f"| date={sample_date or 'NOT_FOUND'} "
                    f"| year={sample_year or 'N/A'} "
                    f"| missing_dates={missing_date_count}"
                )

                added = add_encoded_batches(
                    collection=collection,
                    model=model,
                    pool=pool,
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    label=file_path.name,
                )

                global_total_added += added
                global_page_full += page_full_count
                global_page_window += page_window_count

            except Exception as e:
                errors += 1
                print(f"   ❌ ERROR in {file_path.name}: {e}")

    finally:
        if pool is not None:
            model.stop_multi_process_pool(pool)
            print("🛑 Multi-GPU pool shut down.")

    print("\n" + "=" * 100)
    print("🎉 INDEXING COMPLETE")
    print(f"Total vectors added: {global_total_added}")
    print(f"Page full chunks:    {global_page_full}")
    print(f"Page window chunks:  {global_page_window}")
    print(f"Missing dates:       {global_missing_dates}")
    print(f"Errors:              {errors}")
    print(f"Collection name:     {COLLECTION_NAME}")
    print(f"Chroma path:         {CHROMA_PATH}")
    print("=" * 100)


if __name__ == "__main__":
    main()
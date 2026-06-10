import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION BASE DE DONNÉES ---
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- PARAMÈTRES ---
# Kept for compatibility with old function calls.
# Not used in the new page-based retrieval strategy.
FALLBACK_DISTANCE_THRESHOLD = float(os.getenv("FALLBACK_DISTANCE_THRESHOLD", "1.05"))

# New retrieval balance:
# page_window = precise locator chunks
# page_full   = broad full-page chunks
PAGE_WINDOW_TOP_K_MULTIPLIER = float(
    os.getenv("PAGE_WINDOW_TOP_K_MULTIPLIER", "1.0")
)

PAGE_FULL_TOP_K_RATIO = float(
    os.getenv("PAGE_FULL_TOP_K_RATIO", "0.35")
)

MIN_PAGE_FULL_K = int(
    os.getenv("MIN_PAGE_FULL_K", "5")
)

# Final cap before reranking.
RETRIEVAL_CANDIDATE_MULTIPLIER = float(
    os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "1.2")
)


def normalize_source_file(source_file: str) -> str:
    """
    Ensures the source_file metadata stays compatible with the frontend PDF route.
    """
    source_file = str(source_file or "").strip()

    if not source_file:
        return ""

    base = os.path.basename(source_file)

    if base.lower().endswith(".txt"):
        base = os.path.splitext(base)[0] + ".pdf"

    if not base.lower().endswith(".pdf"):
        base += ".pdf"

    return base


def retrieve_vector_chunks(q_embed, chunk_type, collection, top_k):
    """
    Effectue une recherche vectorielle filtrée par chunking_method.

    New valid chunking_method values:
    - page_window
    - page_full
    """
    if top_k <= 0:
        return [], 999.0

    vec_res = collection.query(
        query_embeddings=q_embed,
        n_results=top_k,
        where={"chunking_method": chunk_type},
    )

    formatted_results = []
    top_distance = 999.0

    if vec_res["ids"] and len(vec_res["ids"][0]) > 0:
        top_distance = vec_res["distances"][0][0]

        for i in range(len(vec_res["ids"][0])):
            meta = vec_res["metadatas"][0][i] or {}

            # Enforce important fields for the next steps.
            meta["source_file"] = normalize_source_file(meta.get("source_file", ""))
            meta.setdefault("page", "Inconnu")
            meta.setdefault("page_id", "")
            meta.setdefault("chunking_method", chunk_type)
            meta.setdefault("chunk_format", "page_text")

            formatted_results.append({
                "id": vec_res["ids"][0][i],
                "text": vec_res["documents"][0][i],
                "meta": meta,
                "distance": vec_res["distances"][0][i],
                "retrieval_source": chunk_type,
            })

    return formatted_results, top_distance


def merge_and_deduplicate(results):
    """
    Merge results from page_window and page_full searches.
    If the same ID appears twice, keep the one with the best distance.
    """
    by_id = {}

    for doc in results:
        doc_id = doc["id"]

        if doc_id not in by_id:
            by_id[doc_id] = doc
        else:
            if doc.get("distance", 999.0) < by_id[doc_id].get("distance", 999.0):
                by_id[doc_id] = doc

    return sorted(by_id.values(), key=lambda x: x.get("distance", 999.0))


def count_unique_pages(results):
    """
    Debug helper only.
    Counts how many different PDF pages are represented in the retrieved chunks.
    """
    pages = set()

    for doc in results:
        meta = doc.get("meta", {}) or {}
        source_file = normalize_source_file(meta.get("source_file", ""))
        page = meta.get("page", "Inconnu")

        if source_file and page != "Inconnu":
            pages.add(f"{source_file}::page_{page}")

    return len(pages)


def get_retrieved_documents(
    query,
    model,
    collection,
    top_k=30,
    threshold=FALLBACK_DISTANCE_THRESHOLD,
):
    """
    Main retrieval function for the new full-vision version.

    Strategy:
    1. Search page_window chunks.
       These are precise chunks used to locate the best PDF page.

    2. Search page_full chunks.
       These help when the question is broad and the whole page is relevant.

    3. Merge and deduplicate.

    4. Return candidates to the reranker.

    Important:
    - The reranker will be updated separately.
    - The LLM vision route will be updated separately.
    - Here we only retrieve candidate chunks from Chroma.
    - threshold is kept only for compatibility with the old signature.
    """

    if not query or not str(query).strip():
        return [], "empty_query"

    q_embed = model.encode([query]).tolist()

    page_window_k = max(
        1,
        int(top_k * PAGE_WINDOW_TOP_K_MULTIPLIER),
    )

    page_full_k = max(
        MIN_PAGE_FULL_K,
        int(top_k * PAGE_FULL_TOP_K_RATIO),
    )

    # -----------------------------
    # PLAN A: precise page windows
    # -----------------------------

    page_window_results, page_window_top_dist = retrieve_vector_chunks(
        q_embed,
        "page_window",
        collection,
        page_window_k,
    )

    # -----------------------------
    # PLAN B: full page chunks
    # -----------------------------

    page_full_results, page_full_top_dist = retrieve_vector_chunks(
        q_embed,
        "page_full",
        collection,
        page_full_k,
    )

    # -----------------------------
    # Merge results
    # -----------------------------

    final_results = merge_and_deduplicate(
        page_window_results + page_full_results
    )

    max_candidates = max(
        top_k,
        int(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER),
    )

    final_results = final_results[:max_candidates]

    strategy_used = (
        f"page_window(k={page_window_k}, results={len(page_window_results)}, "
        f"best_dist={page_window_top_dist:.4f})"
        f"+page_full(k={page_full_k}, results={len(page_full_results)}, "
        f"best_dist={page_full_top_dist:.4f})"
        f"+unique_pages={count_unique_pages(final_results)}"
    )

    print(f"🔎 Retrieval strategy used: {strategy_used}")

    return final_results, strategy_used
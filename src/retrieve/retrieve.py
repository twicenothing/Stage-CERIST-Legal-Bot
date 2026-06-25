import os
import re
import math
import unicodedata
from collections import Counter, defaultdict

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ==============================================================================
# CONFIGURATION BASE DE DONNÉES
# ==============================================================================

CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


# ==============================================================================
# PARAMÈTRES GÉNÉRAUX
# ==============================================================================

# Kept for compatibility with old function calls.
FALLBACK_DISTANCE_THRESHOLD = float(os.getenv("FALLBACK_DISTANCE_THRESHOLD", "1.05"))

# Vector retrieval balance
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


# ==============================================================================
# HYBRID RETRIEVAL CONFIG
# ==============================================================================

USE_KEYWORD_RRF = os.getenv("USE_KEYWORD_RRF", "true").lower() in {
    "1", "true", "yes", "y", "on"
}

KEYWORD_TOP_K_MULTIPLIER = float(
    os.getenv("KEYWORD_TOP_K_MULTIPLIER", "1.0")
)

# RRF constant. Higher = smoother fusion, lower = top ranks matter more.
RRF_K = int(os.getenv("RRF_K", "60"))

# Main fusion weights
SEMANTIC_RRF_WEIGHT = float(
    os.getenv("SEMANTIC_RRF_WEIGHT", "0.7")
)

KEYWORD_RRF_WEIGHT = float(
    os.getenv("KEYWORD_RRF_WEIGHT", "0.3")
)

# How the semantic weight is split between page_window and page_full.
# page_window should usually be higher because it is more precise.
SEMANTIC_PAGE_WINDOW_WEIGHT = float(
    os.getenv("SEMANTIC_PAGE_WINDOW_WEIGHT", "0.75")
)

SEMANTIC_PAGE_FULL_WEIGHT = float(
    os.getenv("SEMANTIC_PAGE_FULL_WEIGHT", "0.25")
)

# BM25 parameters
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

KEYWORD_INDEX_BATCH_SIZE = int(
    os.getenv("KEYWORD_INDEX_BATCH_SIZE", "5000")
)

# Module-level keyword cache.
_KEYWORD_INDEX = None


# ==============================================================================
# BASIC HELPERS
# ==============================================================================

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

    if base.lower().endswith(".json"):
        base = os.path.splitext(base)[0] + ".pdf"

    if base.lower().endswith("_pages.pdf"):
        base = base.replace("_pages.pdf", ".pdf")

    if base.lower().endswith("_recursive.pdf"):
        base = base.replace("_recursive.pdf", ".pdf")

    if not base.lower().endswith(".pdf"):
        base += ".pdf"

    return base


def clean_metadata(meta: dict, chunk_type: str = "") -> dict:
    """
    Normalizes metadata expected by reranker, frontend, and PDF renderer.
    """
    meta = meta or {}

    meta["source_file"] = normalize_source_file(meta.get("source_file", ""))
    meta.setdefault("page", "Inconnu")
    meta.setdefault("page_id", "")
    meta.setdefault("chunking_method", chunk_type or meta.get("chunking_method", ""))
    meta.setdefault("chunk_format", "page_text")

    return meta


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


# ==============================================================================
# VECTOR RETRIEVAL
# ==============================================================================

def retrieve_vector_chunks(q_embed, chunk_type, collection, top_k):
    """
    Vector search filtered by chunking_method.

    Valid chunking_method values:
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
            meta = clean_metadata(
                vec_res["metadatas"][0][i] or {},
                chunk_type=chunk_type,
            )

            formatted_results.append({
                "id": vec_res["ids"][0][i],
                "text": vec_res["documents"][0][i],
                "meta": meta,
                "distance": vec_res["distances"][0][i],
                "retrieval_source": chunk_type,
                "vector_rank": i + 1,
            })

    return formatted_results, top_distance


# ==============================================================================
# KEYWORD / BM25 RETRIEVAL
# ==============================================================================

FRENCH_STOPWORDS = {
    "a", "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du",
    "elle", "en", "et", "eux", "il", "je", "la", "le", "les", "leur",
    "lui", "ma", "mais", "me", "meme", "mes", "moi", "mon", "ne",
    "nos", "notre", "nous", "on", "ou", "par", "pas", "pour", "qu",
    "que", "qui", "sa", "se", "ses", "son", "sur", "ta", "te", "tes",
    "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "est",
    "sont", "etre", "ete", "quel", "quelle", "quels", "quelles",
    "combien", "comment", "quoi", "dont", "ainsi", "plus", "moins",
}

# Keep these even if short.
IMPORTANT_SHORT_TERMS = {
    "ae", "cp", "da", "ht", "ttc", "tva", "jo", "vaep"
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text or "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def tokenize_for_keyword(text: str) -> list[str]:
    """
    Keyword tokenizer for Algerian legal text.

    Keeps:
    - acronyms: AE, CP, DA, VAEP
    - legal refs: 25-60
    - numbers: 48.300.000.000
    - normal legal words
    """
    text = strip_accents(text or "").lower()

    text = text.replace("’", "'")
    text = text.replace("°", "° ")

    tokens = re.findall(
        r"\b\d{1,4}[-–]\d+\b|"                  # 25-60, 24-440
        r"\b\d{1,3}(?:[.\s]\d{3})+(?:,\d+)?\b|" # 48.300.000.000
        r"\b\d+(?:,\d+)?\b|"                    # 2025, 42, 10
        r"\b[a-z0-9]{2,}\b",                    # words / acronyms lowered
        text,
        flags=re.IGNORECASE,
    )

    cleaned = []

    for tok in tokens:
        tok = tok.strip().lower()

        if not tok:
            continue

        if tok in FRENCH_STOPWORDS and tok not in IMPORTANT_SHORT_TERMS:
            continue

        if len(tok) < 2 and tok not in IMPORTANT_SHORT_TERMS:
            continue

        cleaned.append(tok)

    return cleaned


def load_chunks_for_keyword_index(collection, chunk_type: str) -> list[dict]:
    """
    Loads all chunks of one chunking_method from Chroma.
    Used to build the in-memory BM25 index.
    """
    all_docs = []
    offset = 0

    while True:
        batch = collection.get(
            where={"chunking_method": chunk_type},
            include=["documents", "metadatas"],
            limit=KEYWORD_INDEX_BATCH_SIZE,
            offset=offset,
        )

        ids = batch.get("ids", []) or []
        documents = batch.get("documents", []) or []
        metadatas = batch.get("metadatas", []) or []

        if not ids:
            break

        for i in range(len(ids)):
            meta = clean_metadata(
                metadatas[i] or {},
                chunk_type=chunk_type,
            )

            all_docs.append({
                "id": ids[i],
                "text": documents[i] or "",
                "meta": meta,
                "distance": 999.0,
                "retrieval_source": "keyword_bm25",
            })

        offset += len(ids)

        if len(ids) < KEYWORD_INDEX_BATCH_SIZE:
            break

    return all_docs


def build_keyword_index(collection):
    """
    Builds a dependency-free BM25 index over page_window + page_full chunks.

    This runs once per process and is cached in memory.
    """
    global _KEYWORD_INDEX

    if _KEYWORD_INDEX is not None:
        return _KEYWORD_INDEX

    print("🔤 Building keyword BM25 index over page chunks...")

    docs = []
    docs.extend(load_chunks_for_keyword_index(collection, "page_window"))
    docs.extend(load_chunks_for_keyword_index(collection, "page_full"))

    inverted = defaultdict(list)
    doc_lengths = []

    for idx, doc in enumerate(docs):
        meta = doc.get("meta", {}) or {}

        searchable_text = " ".join([
            doc.get("text", "") or "",
            str(meta.get("source_file", "")),
            str(meta.get("page", "")),
            str(meta.get("page_id", "")),
            str(meta.get("chunking_method", "")),
        ])

        tokens = tokenize_for_keyword(searchable_text)
        counts = Counter(tokens)

        doc_lengths.append(sum(counts.values()))

        for term, tf in counts.items():
            inverted[term].append((idx, tf))

    total_docs = len(docs)
    avgdl = sum(doc_lengths) / total_docs if total_docs else 0.0

    idf = {}

    for term, postings in inverted.items():
        df = len(postings)
        idf[term] = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))

    _KEYWORD_INDEX = {
        "docs": docs,
        "inverted": inverted,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "total_docs": total_docs,
    }

    print(
        f"✅ Keyword BM25 index ready: "
        f"{total_docs} chunks, {len(inverted)} unique terms."
    )

    return _KEYWORD_INDEX


def retrieve_keyword_chunks(query: str, collection, top_k: int):
    """
    BM25 keyword search over page chunks.
    """
    if top_k <= 0:
        return []

    index = build_keyword_index(collection)

    docs = index["docs"]
    inverted = index["inverted"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avgdl"]
    idf = index["idf"]

    if not docs or avgdl == 0:
        return []

    query_tokens = tokenize_for_keyword(query)
    query_counts = Counter(query_tokens)

    if not query_counts:
        return []

    scores = defaultdict(float)

    for term, qtf in query_counts.items():
        postings = inverted.get(term)

        if not postings:
            continue

        term_idf = idf.get(term, 0.0)

        for doc_idx, tf in postings:
            dl = doc_lengths[doc_idx] or 1

            denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))

            if denominator == 0:
                continue

            term_score = term_idf * ((tf * (BM25_K1 + 1.0)) / denominator)

            # Small boost if query repeats a term.
            scores[doc_idx] += term_score * min(2.0, float(qtf))

    if not scores:
        return []

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []

    for rank, (doc_idx, score) in enumerate(ranked, start=1):
        base_doc = docs[doc_idx]

        doc = {
            "id": base_doc["id"],
            "text": base_doc["text"],
            "meta": dict(base_doc["meta"]),
            "distance": base_doc.get("distance", 999.0),
            "retrieval_source": "keyword_bm25",
            "keyword_score": float(score),
            "keyword_rank": rank,
        }

        results.append(doc)

    return results


# ==============================================================================
# MERGE / WEIGHTED RRF
# ==============================================================================

def merge_and_deduplicate(results):
    """
    Simple fallback merge if hybrid RRF is disabled.
    If the same ID appears twice, keep the one with the best vector distance.
    """
    by_id = {}

    for doc in results:
        doc_id = doc["id"]

        if doc_id not in by_id:
            by_id[doc_id] = doc
        else:
            old = by_id[doc_id]

            old_distance = old.get("distance", 999.0)
            new_distance = doc.get("distance", 999.0)

            if new_distance < old_distance:
                merged = doc
                merged["retrieval_source"] = (
                    f"{old.get('retrieval_source', '')}+{doc.get('retrieval_source', '')}"
                )
                by_id[doc_id] = merged
            else:
                old["retrieval_source"] = (
                    f"{old.get('retrieval_source', '')}+{doc.get('retrieval_source', '')}"
                )

                if "keyword_score" in doc:
                    old["keyword_score"] = doc["keyword_score"]

                if "keyword_rank" in doc:
                    old["keyword_rank"] = doc["keyword_rank"]

    return sorted(by_id.values(), key=lambda x: x.get("distance", 999.0))


def get_effective_rrf_weights() -> dict:
    """
    Converts the global semantic/keyword weights into per-list weights.

    Example default:
    semantic = 0.7
      page_window = 0.7 * 0.75 = 0.525
      page_full   = 0.7 * 0.25 = 0.175

    keyword = 0.3
    """
    total_semantic_split = SEMANTIC_PAGE_WINDOW_WEIGHT + SEMANTIC_PAGE_FULL_WEIGHT

    if total_semantic_split <= 0:
        page_window_weight = SEMANTIC_RRF_WEIGHT
        page_full_weight = 0.0
    else:
        page_window_weight = SEMANTIC_RRF_WEIGHT * (
            SEMANTIC_PAGE_WINDOW_WEIGHT / total_semantic_split
        )
        page_full_weight = SEMANTIC_RRF_WEIGHT * (
            SEMANTIC_PAGE_FULL_WEIGHT / total_semantic_split
        )

    return {
        "vector_page_window": page_window_weight,
        "vector_page_full": page_full_weight,
        "keyword_bm25": KEYWORD_RRF_WEIGHT,
    }


def weighted_rrf_fuse(rank_lists: list[tuple[str, list[dict]]], max_results: int):
    """
    Weighted Reciprocal Rank Fusion.

    Normal RRF:
        score += 1 / (RRF_K + rank)

    Weighted RRF:
        score += list_weight * (1 / (RRF_K + rank))

    This lets us say:
    - semantic retrieval matters 70%
    - keyword retrieval matters 30%
    """
    weights = get_effective_rrf_weights()

    scores = defaultdict(float)
    docs_by_id = {}
    contributions = defaultdict(list)

    for list_name, docs in rank_lists:
        list_weight = float(weights.get(list_name, 1.0))

        if list_weight <= 0:
            continue

        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.get("id")

            if not doc_id:
                continue

            contribution = list_weight * (1.0 / (RRF_K + rank))

            scores[doc_id] += contribution
            contributions[doc_id].append(
                f"{list_name}:rank={rank}:weight={list_weight:.4f}:contribution={contribution:.6f}"
            )

            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = doc
            else:
                existing = docs_by_id[doc_id]

                # Keep the version with better vector distance when available.
                if doc.get("distance", 999.0) < existing.get("distance", 999.0):
                    docs_by_id[doc_id] = doc

                # Preserve keyword metadata if available.
                if "keyword_score" in doc:
                    docs_by_id[doc_id]["keyword_score"] = doc["keyword_score"]

                if "keyword_rank" in doc:
                    docs_by_id[doc_id]["keyword_rank"] = doc["keyword_rank"]

    ranked_ids = sorted(
        scores.keys(),
        key=lambda doc_id: scores[doc_id],
        reverse=True,
    )

    fused_results = []

    for fused_rank, doc_id in enumerate(ranked_ids[:max_results], start=1):
        doc = docs_by_id[doc_id]

        doc["rrf_score"] = float(scores[doc_id])
        doc["rrf_rank"] = fused_rank
        doc["rrf_contributions"] = contributions[doc_id]

        source_names = []

        for c in contributions[doc_id]:
            source_name = c.split(":")[0]

            if source_name not in source_names:
                source_names.append(source_name)

        doc["retrieval_source"] = "+".join(source_names)

        fused_results.append(doc)

    return fused_results


# ==============================================================================
# MAIN RETRIEVAL FUNCTION
# ==============================================================================

def get_retrieved_documents(
    query,
    model,
    collection,
    top_k=30,
    threshold=FALLBACK_DISTANCE_THRESHOLD,
):
    """
    Main retrieval function for the full-vision version.

    Hybrid strategy:
    1. Semantic vector search over page_window chunks.
    2. Semantic vector search over page_full chunks.
    3. Keyword BM25 search over page_window + page_full chunks.
    4. Weighted RRF fusion.
    5. Return candidates to the reranker.

    The reranker remains separate.
    The vision route remains separate.
    """

    if not query or not str(query).strip():
        return [], "empty_query"

    query = str(query).strip()

    q_embed = model.encode([query]).tolist()

    page_window_k = max(
        1,
        int(top_k * PAGE_WINDOW_TOP_K_MULTIPLIER),
    )

    page_full_k = max(
        MIN_PAGE_FULL_K,
        int(top_k * PAGE_FULL_TOP_K_RATIO),
    )

    keyword_k = max(
        1,
        int(top_k * KEYWORD_TOP_K_MULTIPLIER),
    )

    max_candidates = max(
        top_k,
        int(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER),
    )

    page_window_results, page_window_top_dist = retrieve_vector_chunks(
        q_embed,
        "page_window",
        collection,
        page_window_k,
    )

    page_full_results, page_full_top_dist = retrieve_vector_chunks(
        q_embed,
        "page_full",
        collection,
        page_full_k,
    )

    keyword_results = []

    if USE_KEYWORD_RRF:
        keyword_results = retrieve_keyword_chunks(
            query=query,
            collection=collection,
            top_k=keyword_k,
        )

    if USE_KEYWORD_RRF:
        final_results = weighted_rrf_fuse(
            rank_lists=[
                ("vector_page_window", page_window_results),
                ("vector_page_full", page_full_results),
                ("keyword_bm25", keyword_results),
            ],
            max_results=max_candidates,
        )
    else:
        final_results = merge_and_deduplicate(
            page_window_results + page_full_results
        )[:max_candidates]

    weights = get_effective_rrf_weights()

    strategy_used = (
        f"hybrid_weighted_rrf={USE_KEYWORD_RRF}"
        f"+weights("
        f"semantic={SEMANTIC_RRF_WEIGHT:.2f}, "
        f"keyword={KEYWORD_RRF_WEIGHT:.2f}, "
        f"page_window={weights['vector_page_window']:.3f}, "
        f"page_full={weights['vector_page_full']:.3f}"
        f")"
        f"+page_window(k={page_window_k}, results={len(page_window_results)}, "
        f"best_dist={page_window_top_dist:.4f})"
        f"+page_full(k={page_full_k}, results={len(page_full_results)}, "
        f"best_dist={page_full_top_dist:.4f})"
        f"+keyword(k={keyword_k}, results={len(keyword_results)})"
        f"+final={len(final_results)}"
        f"+unique_pages={count_unique_pages(final_results)}"
    )

    print(f"🔎 Retrieval strategy used: {strategy_used}")

    return final_results, strategy_used
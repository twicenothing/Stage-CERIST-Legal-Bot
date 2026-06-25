import os
import re
import sys
from datetime import datetime

# 1. Calculate the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to Python's module search path
sys.path.append(parent_dir)

# Import the retrieval function
from retrieve.retrieve import get_retrieved_documents


# ==============================================================================
# TEMPORAL RERANK CONFIG
# ==============================================================================

TEMPORAL_RERANK_ENABLED = os.getenv("TEMPORAL_RERANK_ENABLED", "true").lower() in {
    "1", "true", "yes", "y", "on"
}

# "Very very close similarity" threshold.
# If two docs have rerank scores within this delta, the newer one wins.
# Start with 0.01. Increase to 0.02 if recency is too weak.
TEMPORAL_TIE_DELTA = float(
    os.getenv("TEMPORAL_TIE_DELTA", "0.01")
)

# If query explicitly asks for a year or a specific legal act,
# do not prefer newer documents automatically.
TEMPORAL_DISABLE_ON_EXPLICIT_DATE = os.getenv(
    "TEMPORAL_DISABLE_ON_EXPLICIT_DATE",
    "true"
).lower() in {"1", "true", "yes", "y", "on"}


# ==============================================================================
# TEMPORAL HELPERS
# ==============================================================================

def safe_int(value, default=0):
    try:
        if value is None or value == "":
            return default

        return int(value)
    except Exception:
        return default


def parse_iso_date_to_ordinal(value: str) -> int:
    """
    Converts YYYY-MM-DD into an ordinal integer.
    Higher = more recent.

    Example:
    2025-01-06 -> 739257
    """
    value = str(value or "").strip()

    if not value:
        return 0

    try:
        return datetime.strptime(value, "%Y-%m-%d").date().toordinal()
    except Exception:
        return 0


def extract_year_from_source_file(source_file: str) -> int:
    """
    Fallback year extraction from filename.

    Examples:
    - F2025007.pdf -> 2025
    - F2010012.txt -> 2010
    - F2017006_2017.pdf -> 2017
    """
    source_file = str(source_file or "")

    match = re.search(r"F((?:19|20)\d{2})\d*", source_file, flags=re.IGNORECASE)

    if match:
        return int(match.group(1))

    match = re.search(r"\b((?:19|20)\d{2})\b", source_file)

    if match:
        return int(match.group(1))

    return 0


def extract_temporal_info(doc: dict) -> dict:
    """
    Extracts temporal metadata from a retrieved/reranked doc.

    Preferred metadata from your new pipeline:
    - journal_date_iso
    - publication_date
    - journal_year
    - publication_year
    - source_year

    Fallback:
    - source_file filename year
    """
    meta = doc.get("meta", {}) or {}

    date_iso = (
        meta.get("journal_date_iso")
        or meta.get("publication_date")
        or ""
    )

    date_ordinal = parse_iso_date_to_ordinal(date_iso)

    year = (
        safe_int(meta.get("journal_year"))
        or safe_int(meta.get("publication_year"))
        or safe_int(meta.get("source_year"))
        or extract_year_from_source_file(meta.get("source_file", ""))
    )

    # If exact date is missing but year exists, use January 1st of that year
    # as an approximate temporal ordering.
    if date_ordinal == 0 and year > 0:
        try:
            date_ordinal = datetime.strptime(f"{year}-01-01", "%Y-%m-%d").date().toordinal()
        except Exception:
            date_ordinal = 0

    return {
        "date_iso": str(date_iso or ""),
        "year": int(year or 0),
        "date_ordinal": int(date_ordinal or 0),
    }


def query_has_explicit_temporal_constraint(query: str) -> bool:
    """
    Detects cases where automatic recency preference should be disabled.

    Examples:
    - "décret exécutif n° 17-60"
    - "en 2017"
    - "loi n° 10-02"
    - "ancien texte"
    - "historique"
    """
    q = str(query or "").lower()

    # Explicit year
    if re.search(r"\b(19|20)\d{2}\b", q):
        return True

    # Explicit legal reference: n° 23-64, n° 17-60, etc.
    if re.search(r"\bn\s*°?\s*\d{2,4}[-–]\d+\b", q):
        return True

    temporal_words = [
        "historique",
        "ancien",
        "ancienne",
        "anciens",
        "anciennes",
        "avant",
        "après",
        "apres",
        "entre",
        "depuis",
        "jusqu",
        "année",
        "annee",
        "date",
        "abrogé",
        "abroge",
        "modifié",
        "modifie",
    ]

    if any(word in q for word in temporal_words):
        return True

    return False


def sort_close_docs_by_recency(query: str, reranked_docs: list[dict]) -> list[dict]:
    """
    Reorders only documents with very close rerank scores.

    Logic:
    1. Sort by CrossEncoder rerank_score.
    2. Split into groups where scores are very close.
    3. Inside each close group, prefer the most recent document.
    4. Keep relevance dominant.

    This avoids:
    - recent irrelevant document beating older correct document.

    But allows:
    - recent correct document beating older correct document when both are similarly relevant.
    """
    if not TEMPORAL_RERANK_ENABLED:
        for doc in reranked_docs:
            temporal = extract_temporal_info(doc)
            doc["temporal_date_iso"] = temporal["date_iso"]
            doc["temporal_year"] = temporal["year"]
            doc["temporal_date_ordinal"] = temporal["date_ordinal"]
            doc["temporal_applied"] = False

        return reranked_docs

    if TEMPORAL_DISABLE_ON_EXPLICIT_DATE and query_has_explicit_temporal_constraint(query):
        print("⏳ Temporal tie-break disabled: query contains explicit year/date/legal reference.")

        for doc in reranked_docs:
            temporal = extract_temporal_info(doc)
            doc["temporal_date_iso"] = temporal["date_iso"]
            doc["temporal_year"] = temporal["year"]
            doc["temporal_date_ordinal"] = temporal["date_ordinal"]
            doc["temporal_applied"] = False

        return reranked_docs

    if not reranked_docs:
        return reranked_docs

    # Ensure temporal fields exist.
    for doc in reranked_docs:
        temporal = extract_temporal_info(doc)
        doc["temporal_date_iso"] = temporal["date_iso"]
        doc["temporal_year"] = temporal["year"]
        doc["temporal_date_ordinal"] = temporal["date_ordinal"]
        doc["temporal_applied"] = True

    # Pure relevance first.
    docs_sorted = sorted(
        reranked_docs,
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )

    groups = []
    current_group = []
    group_anchor_score = None

    for doc in docs_sorted:
        score = float(doc.get("rerank_score", 0.0))

        if group_anchor_score is None:
            current_group = [doc]
            group_anchor_score = score
            continue

        # If score is very close to the best score in this group,
        # put it in the same group.
        if abs(group_anchor_score - score) <= TEMPORAL_TIE_DELTA:
            current_group.append(doc)
        else:
            groups.append(current_group)
            current_group = [doc]
            group_anchor_score = score

    if current_group:
        groups.append(current_group)

    final_docs = []

    for group_index, group in enumerate(groups, start=1):
        for doc in group:
            doc["temporal_group"] = group_index

        # Inside close-score group:
        # most recent first, then rerank_score.
        group_sorted = sorted(
            group,
            key=lambda x: (
                x.get("temporal_date_ordinal", 0),
                x.get("rerank_score", 0.0),
            ),
            reverse=True,
        )

        final_docs.extend(group_sorted)

    return final_docs


# ==============================================================================
# RERANKING
# ==============================================================================

def rerank_documents(query, retrieved_docs, reranker_model, top_k=4):
    """
    Reranks retrieved page chunks using a CrossEncoder.

    Then applies temporal tie-break:
    - CrossEncoder relevance remains the main criterion.
    - If scores are very close, newer Journal Officiel date wins.
    """

    if not retrieved_docs:
        return []

    query = str(query or "").strip()

    if not query:
        return retrieved_docs[:top_k]

    cross_inp = [
        [query, doc.get("text", "")]
        for doc in retrieved_docs
    ]

    scores = reranker_model.predict(
        cross_inp,
        batch_size=2,
    )

    for i, doc in enumerate(retrieved_docs):
        score = float(scores[i])

        doc["rerank_score"] = score
        doc["final_rerank_score"] = score
        doc["text_chars"] = len(doc.get("text", "") or "")

        temporal = extract_temporal_info(doc)
        doc["temporal_date_iso"] = temporal["date_iso"]
        doc["temporal_year"] = temporal["year"]
        doc["temporal_date_ordinal"] = temporal["date_ordinal"]
        doc["temporal_applied"] = False

    reranked_docs = sorted(
        retrieved_docs,
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
    )

    reranked_docs = sort_close_docs_by_recency(
        query=query,
        reranked_docs=reranked_docs,
    )

    for rank, doc in enumerate(reranked_docs, start=1):
        doc["rerank_rank"] = rank

    return reranked_docs[:top_k]


# ==============================================================================
# FULL RETRIEVAL + RERANK PIPELINE
# ==============================================================================

def get_best_documents_for_llm(
    retrieval_query,
    collection,
    bi_encoder,
    reranker,
    top_k_retrieve=30,
    top_k_rerank=4,
    rerank_query=None,
):
    """
    Executes the retrieval + rerank pipeline.

    Parameters:
    - retrieval_query:
        Query used for Chroma vector/BM25 search.
        This can be the optimized/enhanced query.

    - rerank_query:
        Query used by the CrossEncoder reranker.
        This should be the user's original query.
        If None, retrieval_query is used.

    Full-vision flow:
    1. Retrieve top_k_retrieve page chunks from Chroma.
    2. Rerank those chunks using the user's query.
    3. If several docs have very close rerank scores, prefer the most recent one.
    4. Return top_k_rerank best chunks.
    5. Later, another module will take source_file + page and render the PDF pages.
    """

    if rerank_query is None:
        rerank_query = retrieval_query

    print("=" * 90)
    print("🔎 PAGE RAG RETRIEVAL / RERANK DEBUG")
    print(f"📥 Retrieval query used for Chroma/vector/BM25 search:\n   {retrieval_query}")
    print(f"🎯 Rerank query used for CrossEncoder:\n   {rerank_query}")
    print(f"📌 top_k_retrieve={top_k_retrieve} | top_k_rerank={top_k_rerank}")
    print(
        f"⏳ temporal_enabled={TEMPORAL_RERANK_ENABLED} "
        f"| tie_delta={TEMPORAL_TIE_DELTA} "
        f"| disable_on_explicit_date={TEMPORAL_DISABLE_ON_EXPLICIT_DATE}"
    )
    print("=" * 90)

    initial_docs, strategy_used = get_retrieved_documents(
        query=retrieval_query,
        model=bi_encoder,
        collection=collection,
        top_k=top_k_retrieve,
    )

    print(f"🔎 Retrieval strategy used: {strategy_used}")
    print(f"📄 Retrieved candidate docs before rerank: {len(initial_docs)}")

    if not initial_docs:
        print("⚠️ No documents retrieved.")
        return []

    final_docs = rerank_documents(
        query=rerank_query,
        retrieved_docs=initial_docs,
        reranker_model=reranker,
        top_k=top_k_rerank,
    )

    print(f"✅ Final docs after rerank: {len(final_docs)}")

    for i, doc in enumerate(final_docs, start=1):
        meta = doc.get("meta", {}) or {}

        print(
            f"   #{i} "
            f"| rerank_score={doc.get('rerank_score')} "
            f"| temporal_date={doc.get('temporal_date_iso') or 'N/A'} "
            f"| temporal_year={doc.get('temporal_year') or 'N/A'} "
            f"| temporal_group={doc.get('temporal_group', 'N/A')} "
            f"| distance={doc.get('distance')} "
            f"| method={meta.get('chunking_method')} "
            f"| source={meta.get('source_file')} "
            f"| page={meta.get('page')} "
            f"| page_id={meta.get('page_id')}"
        )

    print("=" * 90)

    return final_docs
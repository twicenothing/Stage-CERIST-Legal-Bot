import os
import sys

# 1. Calculate the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to Python's module search path
sys.path.append(parent_dir) 

# Import the retrieval function we built previously
from retrieve.retrieve import get_retrieved_documents

def looks_like_tabular_text(text: str) -> bool:
    lower = (text or "").lower()

    signals = [
        "tableau annexe",
        "crédits ouverts",
        "credits ouverts",
        "crédits ouverts en da",
        "credits ouverts en da",
        "nos des chapitres",
        "libelles",
        "libellés",
        "répartition par chapitre",
        "repartition par chapitre",
        "effectifs selon la nature",
        "contrat à durée",
        "contrat a duree",
        "total général",
        "total general",
        "|---",
    ]

    return any(s in lower for s in signals)



HUGE_TABULAR_REGEX_CHAR_LIMIT = 18000
HUGE_TABULAR_REGEX_PENALTY = 0.25
MAX_HUGE_TABULAR_REGEX_KEEP_RANK = 2


def rerank_documents(query, retrieved_docs, reranker_model, top_k=4):
    """
    Takes loosely retrieved documents from ChromaDB and scores them using a CrossEncoder.

    Logic:
    - Keep raw rerank_score for debugging.
    - Penalize huge regex chunks that look like swallowed table annexes.
    - Sort by adjusted_rerank_score.
    - After sorting, skip penalized huge tabular regex chunks if they appear after rank 2.
    """
    if not retrieved_docs:
        return []

    cross_inp = [[query, doc["text"]] for doc in retrieved_docs]
    scores = reranker_model.predict(cross_inp, batch_size=2)

    for i in range(len(retrieved_docs)):
        raw_score = float(scores[i])
        retrieved_docs[i]["rerank_score"] = raw_score

        meta = retrieved_docs[i].get("meta", {}) or {}
        method = meta.get("chunking_method", "")
        text = retrieved_docs[i].get("text", "") or ""

        text_chars = len(text)
        is_huge_tabular_regex = (
            method == "regex"
            and text_chars > HUGE_TABULAR_REGEX_CHAR_LIMIT
            and looks_like_tabular_text(text)
        )

        adjusted_score = raw_score
        penalty_reason = ""

        if is_huge_tabular_regex:
            adjusted_score = raw_score * HUGE_TABULAR_REGEX_PENALTY
            penalty_reason = "huge_tabular_regex_penalty"

        retrieved_docs[i]["adjusted_rerank_score"] = adjusted_score
        retrieved_docs[i]["rerank_penalty_reason"] = penalty_reason
        retrieved_docs[i]["is_huge_tabular_regex"] = is_huge_tabular_regex
        retrieved_docs[i]["text_chars"] = text_chars

    reranked_docs = sorted(
        retrieved_docs,
        key=lambda x: x.get("adjusted_rerank_score", x.get("rerank_score", 0)),
        reverse=True,
    )

    final_docs = []

    for adjusted_rank, doc in enumerate(reranked_docs, start=1):
        is_huge_tabular_regex = doc.get("is_huge_tabular_regex", False)

        if (
            is_huge_tabular_regex
            and adjusted_rank > MAX_HUGE_TABULAR_REGEX_KEEP_RANK
        ):
            meta = doc.get("meta", {}) or {}
            print(
                "⚠️ Skipping penalized huge tabular regex after adjusted rank "
                f"{adjusted_rank} | source={meta.get('source_file')} "
                f"| page={meta.get('page')} "
                f"| chars={doc.get('text_chars')} "
                f"| raw={doc.get('rerank_score')} "
                f"| adjusted={doc.get('adjusted_rerank_score')}"
            )
            continue

        final_docs.append(doc)

        if len(final_docs) >= top_k:
            break

    return final_docs

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
    Exécute le pipeline complet:
    - retrieval_query : utilisé pour la recherche vectorielle Chroma
    - rerank_query    : utilisé pour le CrossEncoder reranker

    Usage recommandé:
    - retrieval_query = requête optimisée
    - rerank_query    = question originale utilisateur

    Backward compatible:
    - si rerank_query=None, on utilise retrieval_query pour le reranking aussi.
    """

    if rerank_query is None:
        rerank_query = retrieval_query

    print("=" * 90)
    print("🔎 RAG RETRIEVAL / RERANK DEBUG")
    print(f"📥 Retrieval query used for Chroma/vector search:\n   {retrieval_query}")
    print(f"🎯 Rerank query used for CrossEncoder:\n   {rerank_query}")
    print(f"📌 top_k_retrieve={top_k_retrieve} | top_k_rerank={top_k_rerank}")
    print("=" * 90)

    initial_docs, strategy_used = get_retrieved_documents(
        retrieval_query,
        bi_encoder,
        collection,
        top_k=top_k_retrieve
    )

    print(f"🔎 Retrieval strategy used: {strategy_used}")
    print(f"📄 Retrieved candidate docs before rerank: {len(initial_docs)}")

    if not initial_docs:
        print("⚠️ No documents retrieved.")
        return []

    final_docs = rerank_documents(
        rerank_query,
        initial_docs,
        reranker,
        top_k=top_k_rerank
    )

    print(f"✅ Final docs after rerank: {len(final_docs)}")

    for i, doc in enumerate(final_docs, start=1):
        meta = doc.get("meta", {}) or {}
        print(
            f"   #{i} | score={doc.get('rerank_score')} "
            f"| distance={doc.get('distance')} "
            f"| method={meta.get('chunking_method')} "
            f"| source={meta.get('source_file')} "
            f"| page={meta.get('page')}"
        )

    print("=" * 90)

    return final_docs
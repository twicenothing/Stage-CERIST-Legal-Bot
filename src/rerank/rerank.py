import os
import sys

# 1. Calculate the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to Python's module search path
sys.path.append(parent_dir)

# Import the retrieval function
from retrieve.retrieve import get_retrieved_documents


# ==============================================================================
# RERANKING
# ==============================================================================

def rerank_documents(query, retrieved_docs, reranker_model, top_k=4):
    """
    Reranks retrieved page chunks using a CrossEncoder.

    New full-vision logic:
    - Input docs come from page_window/page_full retrieval.
    - Reranking uses the user's original query.
    - No regex/table penalties.
    - No old fallback logic.
    - Output docs will later be converted to PDF pages for vision.
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
        doc["text_chars"] = len(doc.get("text", "") or "")

    reranked_docs = sorted(
        retrieved_docs,
        key=lambda x: x.get("rerank_score", 0.0),
        reverse=True,
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
        Query used for Chroma vector search.
        This can be the optimized/enhanced query.

    - rerank_query:
        Query used by the CrossEncoder reranker.
        This should be the user's original query.
        If None, retrieval_query is used.

    New full-vision flow:
    1. Retrieve top_k_retrieve page chunks from Chroma.
    2. Rerank those chunks using the user's query.
    3. Return top_k_rerank best chunks.
    4. Later, another module will take source_file + page and render the PDF pages.
    """

    if rerank_query is None:
        rerank_query = retrieval_query

    print("=" * 90)
    print("🔎 PAGE RAG RETRIEVAL / RERANK DEBUG")
    print(f"📥 Retrieval query used for Chroma/vector search:\n   {retrieval_query}")
    print(f"🎯 Rerank query used for CrossEncoder:\n   {rerank_query}")
    print(f"📌 top_k_retrieve={top_k_retrieve} | top_k_rerank={top_k_rerank}")
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
            f"| distance={doc.get('distance')} "
            f"| method={meta.get('chunking_method')} "
            f"| source={meta.get('source_file')} "
            f"| page={meta.get('page')} "
            f"| page_id={meta.get('page_id')}"
        )

    print("=" * 90)

    return final_docs
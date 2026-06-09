import os
import sys

# 1. Calculate the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to Python's module search path
sys.path.append(parent_dir) 

# Import the retrieval function we built previously
from retrieve.retrieve import get_retrieved_documents

def rerank_documents(query, retrieved_docs, reranker_model, top_k=4):
    """
    Takes the loosely retrieved documents from ChromaDB and scores them 
    using a Cross-Encoder for pinpoint accuracy.
    """
    if not retrieved_docs:
        return []

    # 1. Prepare the input pairs for the Cross-Encoder: [[query, doc1], [query, doc2], ...]
    cross_inp = [[query, doc["text"]] for doc in retrieved_docs]

    # 2. Predict the relevance scores
    scores = reranker_model.predict(cross_inp, batch_size=2)

    # 3. Attach the new scores to our documents
    for i in range(len(retrieved_docs)):
        retrieved_docs[i]["rerank_score"] = float(scores[i])

    # 4. Sort the documents based on the new Cross-Encoder score (Descending)
    reranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

    # 5. Return only the absolute best documents for the LLM
    return reranked_docs[:top_k]

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
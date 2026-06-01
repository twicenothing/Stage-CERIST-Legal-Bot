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

def get_best_documents_for_llm(query, collection, bi_encoder, reranker, top_k_retrieve=30, top_k_rerank=4):
    """
    Exécute le pipeline complet:
    regex + table retrieval -> optional recursive fallback -> reranking.
    """

    initial_docs, strategy_used = get_retrieved_documents(
        query,
        bi_encoder,
        collection,
        top_k=top_k_retrieve
    )

    print(f"🔎 Retrieval strategy used: {strategy_used}")

    if not initial_docs:
        return []

    final_docs = rerank_documents(
        query,
        initial_docs,
        reranker,
        top_k=top_k_rerank
    )

    return final_docs
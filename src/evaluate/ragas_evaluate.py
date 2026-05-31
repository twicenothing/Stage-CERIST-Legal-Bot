# ── MUST BE FIRST ────────────────────────────────────────────────────────────
import sys
from types import ModuleType
from ragas.run_config import RunConfig
# Patch 1: ragas imports ChatVertexAI from langchain_community (moved package)
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI as _CV
    except ImportError:
        class _CV: pass

    _mod = ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = _CV
    sys.modules["langchain_community.chat_models.vertexai"] = _mod

# Patch 2: langchain_core.exceptions missing ContextOverflowError in 0.2.x
try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:
    import langchain_core.exceptions as _lce
    class ContextOverflowError(Exception): pass
    _lce.ContextOverflowError = ContextOverflowError
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import pandas as pd
from datasets import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from generate.query_parse import rewrite_query
from rerank.rerank import get_best_documents_for_llm
from generate.llm_generate import init_rag_pipeline, generate_legal_response

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
JUDGE_MODEL = "llama3:70b"

judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_HOST))
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_HOST)
)


def run_evaluation(testset_path: str):
    print("🚀 Initialisation du pipeline de test...")
    collection, bi_encoder, reranker = init_rag_pipeline()

    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    print(f"🔄 Exécution de {len(test_data)} questions à travers le pipeline...")
    for item in test_data:
        question = item["question"]
        opt_query = rewrite_query(question)

        best_docs = get_best_documents_for_llm(
            opt_query, collection, bi_encoder, reranker,
            top_k_retrieve=8, top_k_rerank=2
        )

        contexts = [doc["text"] for doc in best_docs] if best_docs else ["Aucun contexte trouvé."]
        answer = (
            generate_legal_response(question, best_docs)
            if best_docs
            else "Je suis désolé, je n'ai pas la réponse à cette question."
        )

        results_data["user_input"].append(question)
        results_data["response"].append(answer)
        results_data["retrieved_contexts"].append(contexts)
        results_data["reference"].append(item["ground_truth"])

    dataset = Dataset.from_dict(results_data)

    print("⚖️ Lancement de l'évaluation RAGAS...")
    custom_run_config = RunConfig(max_workers=1, timeout=120)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=custom_run_config,
    )

    df_results = result.to_pandas()
    output_file = os.path.join(current_dir, "ragas_results.csv")
    df_results.to_csv(output_file, index=False)

    print("\n📊 Score Global :")
    print(result)
    print(f"\n✅ Résultats sauvegardés dans {output_file}")


if __name__ == "__main__":
    test_file = "../../data/ragas_dataset/test_set.json"
    run_evaluation(test_file)
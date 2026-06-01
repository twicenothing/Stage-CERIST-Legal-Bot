# ── MUST BE FIRST ────────────────────────────────────────────────────────────
import sys
from types import ModuleType
from ragas.run_config import RunConfig

# Patch 1: ragas imports ChatVertexAI from langchain_community (moved package)
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI as _CV
    except ImportError:
        class _CV:
            pass

    _mod = ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = _CV
    sys.modules["langchain_community.chat_models.vertexai"] = _mod

# Patch 2: langchain_core.exceptions missing ContextOverflowError in 0.2.x
try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:
    import langchain_core.exceptions as _lce

    class ContextOverflowError(Exception):
        pass

    _lce.ContextOverflowError = ContextOverflowError
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Load project-root .env explicitly so RAGAS uses the same config as the app.
load_dotenv(dotenv_path=Path(project_root) / ".env")

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
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "llama3:8b")
JUDGE_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text")

TOP_K_RETRIEVE = int(os.getenv("RAGAS_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAGAS_TOP_K_RERANK", "3"))

judge_llm = LangchainLLMWrapper(ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_HOST))
judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model=JUDGE_EMBEDDING_MODEL, base_url=OLLAMA_HOST)
)


def _json_dumps_safe(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def _doc_debug_row(doc):
    meta = doc.get("meta", {}) or {}
    return {
        "id": doc.get("id", ""),
        "chunking_method": meta.get("chunking_method", ""),
        "chunk_format": meta.get("chunk_format", ""),
        "source_file": meta.get("source_file", ""),
        "page": meta.get("page", ""),
        "parent_title": meta.get("parent_title", ""),
        "document_type": meta.get("document_type", ""),
        "table_id": meta.get("table_id", ""),
        "table_kind": meta.get("table_kind", ""),
        "row_index": meta.get("row_index", ""),
        "distance": doc.get("distance", None),
        "rerank_score": doc.get("rerank_score", None),
    }


def run_evaluation(testset_path: str):
    print("🚀 Initialisation du pipeline de test...")
    print(f"⚙️  Judge LLM RAGAS : {JUDGE_MODEL}")
    print(f"⚙️  Judge embeddings RAGAS : {JUDGE_EMBEDDING_MODEL}")
    print(f"⚙️  Retrieval top_k={TOP_K_RETRIEVE}, rerank top_k={TOP_K_RERANK}")

    collection, bi_encoder, reranker = init_rag_pipeline()

    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    ragas_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    debug_rows = []

    test_data = test_data[:50]  # Limit to 100 for faster evaluation; adjust as needed.
    print(f"🔄 Exécution de {len(test_data)} questions à travers le pipeline...")
    for idx, item in enumerate(test_data, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"   [{idx}/{len(test_data)}] {question[:90]}...")

        opt_query = rewrite_query(question)

        best_docs = get_best_documents_for_llm(
            opt_query,
            collection,
            bi_encoder,
            reranker,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_rerank=TOP_K_RERANK,
        )

        contexts = [doc.get("text", "") for doc in best_docs] if best_docs else ["Aucun contexte trouvé."]
        answer = (
            generate_legal_response(question, best_docs)
            if best_docs
            else "Je suis désolé, je n'ai pas la réponse à cette question."
        )

        ragas_data["user_input"].append(question)
        ragas_data["response"].append(answer)
        ragas_data["retrieved_contexts"].append(contexts)
        ragas_data["reference"].append(ground_truth)

        debug_rows.append({
            "user_input": question,
            "optimized_query": opt_query,
            "response": answer,
            "reference": ground_truth,
            "retrieved_contexts": contexts,
            "retrieved_docs_debug": [_doc_debug_row(doc) for doc in best_docs],
        })

    dataset = Dataset.from_dict(ragas_data)

    print("⚖️ Lancement de l'évaluation RAGAS...")
    custom_run_config = RunConfig(max_workers=1, timeout=600)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=custom_run_config,
    )

    df_results = result.to_pandas()

    # Add debug columns after RAGAS evaluation so the metrics receive only the
    # standard RAGAS fields, while your CSV still shows table/regex/recursive hits.
    if len(df_results) == len(debug_rows):
        df_results["optimized_query"] = [row["optimized_query"] for row in debug_rows]
        df_results["retrieved_docs_debug"] = [
            _json_dumps_safe(row["retrieved_docs_debug"]) for row in debug_rows
        ]

    output_file = os.path.join(current_dir, "ragas_results.csv")
    detailed_output_file = os.path.join(current_dir, "ragas_results_detailed.json")

    df_results.to_csv(output_file, index=False)

    with open(detailed_output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "judge_model": JUDGE_MODEL,
                "judge_embedding_model": JUDGE_EMBEDDING_MODEL,
                "top_k_retrieve": TOP_K_RETRIEVE,
                "top_k_rerank": TOP_K_RERANK,
            },
            "rows": debug_rows,
            "ragas_results": df_results.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2, default=str)

    print("\n📊 Score Global :")
    print(result)
    print(f"\n✅ Résultats CSV sauvegardés dans {output_file}")
    print(f"✅ Résultats détaillés sauvegardés dans {detailed_output_file}")


if __name__ == "__main__":
    default_test_file = os.path.join(project_root, "data", "ragas_dataset", "test_set.json")
    test_file = sys.argv[1] if len(sys.argv) > 1 else default_test_file
    run_evaluation(test_file)

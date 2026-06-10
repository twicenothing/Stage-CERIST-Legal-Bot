# ── MUST BE FIRST ────────────────────────────────────────────────────────────
import sys
from types import ModuleType

# Patch 1:
# RAGAS may still try to import ChatVertexAI from an old LangChain path.
# This fake compatibility module must be created BEFORE importing anything from ragas.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI as _CV
    except ImportError:
        class _CV:
            pass

    _mod = ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = _CV
    sys.modules["langchain_community.chat_models.vertexai"] = _mod

# Patch 2:
# Some RAGAS/LangChain combinations expect ContextOverflowError.
try:
    from langchain_core.exceptions import ContextOverflowError
except Exception:
    try:
        import langchain_core.exceptions as _lce

        class ContextOverflowError(Exception):
            pass

        _lce.ContextOverflowError = ContextOverflowError
    except Exception:
        pass

# Now it is safe to import RAGAS
from ragas.run_config import RunConfig
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

# Load project-root .env explicitly
load_dotenv(dotenv_path=Path(project_root) / ".env")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from generate.query_parse import rewrite_query
from rerank.rerank import get_best_documents_for_llm
from generate.llm_generate import init_rag_pipeline, generate_legal_response

# ---------------------------------------------------------------------------
# RAGAS imports
# ---------------------------------------------------------------------------
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# New RAGAS metric import style.
# If it fails, fallback to the old import style.
# ---------------------------------------------------------------------------
# RAGAS metrics
# Important:
# Do NOT use ragas.metrics.collections with classic evaluate().
# In RAGAS 0.4.x, collections metrics can trigger:
# TypeError: All metrics must be initialised metric objects
# ---------------------------------------------------------------------------
try:
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._answer_relevance import answer_relevancy
    from ragas.metrics._context_precision import context_precision
    from ragas.metrics._context_recall import context_recall
except Exception:
    # Fallback for older RAGAS versions
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )


RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

# ---------------------------------------------------------------------------
# Ollama / LangChain imports
# ---------------------------------------------------------------------------
# New official import path
try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    # Fallback for older environments
    try:
        from langchain_community.chat_models.ollama import ChatOllama
        from langchain_community.embeddings.ollama import OllamaEmbeddings
    except ImportError as e:
        raise ImportError(
            "Could not import ChatOllama/OllamaEmbeddings. "
            "Install the modern package with:\n\n"
            "    pip install langchain-ollama\n"
        ) from e


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "mistral:latest")
JUDGE_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text")

TOP_K_RETRIEVE = int(os.getenv("RAGAS_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAGAS_TOP_K_RERANK", "2"))

# Local-friendly context settings
RAGAS_NUM_CTX = int(os.getenv("RAGAS_NUM_CTX", "8192"))
RAGAS_NUM_PREDICT = int(os.getenv("RAGAS_NUM_PREDICT", "512"))

# Evaluation size for local testing
RAGAS_MAX_QUESTIONS = int(os.getenv("RAGAS_MAX_QUESTIONS", "50"))

# RAGAS execution settings
RAGAS_MAX_WORKERS = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
RAGAS_TIMEOUT = int(os.getenv("RAGAS_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# Judge LLM and embeddings
# ---------------------------------------------------------------------------
def build_judge_llm():
    """
    Builds the local Ollama judge model used by RAGAS.
    num_ctx controls the context window for the judge model.
    """
    return LangchainLLMWrapper(
        ChatOllama(
            model=JUDGE_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0,
            num_ctx=RAGAS_NUM_CTX,
            num_predict=RAGAS_NUM_PREDICT,
        )
    )


def build_judge_embeddings():
    """
    Builds the local Ollama embedding model used by RAGAS.
    """
    return LangchainEmbeddingsWrapper(
        OllamaEmbeddings(
            model=JUDGE_EMBEDDING_MODEL,
            base_url=OLLAMA_HOST,
        )
    )


judge_llm = build_judge_llm()
judge_embeddings = build_judge_embeddings()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def normalize_contexts(best_docs):
    """
    RAGAS expects retrieved_contexts to be a list of strings.
    """
    if not best_docs:
        return ["Aucun contexte trouvé."]

    contexts = []

    for doc in best_docs:
        text = doc.get("text", "")

        if text is None:
            text = ""

        text = str(text).strip()

        if text:
            contexts.append(text)

    return contexts if contexts else ["Aucun contexte trouvé."]


def get_ground_truth(item):
    """
    Supports both possible field names:
    - ground_truth
    - reponse
    """
    return item.get("ground_truth") or item.get("reponse") or ""


def get_question(item):
    return item.get("question") or item.get("user_input") or ""


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_evaluation(testset_path: str):
    print("🚀 Initialisation du pipeline de test...")
    print(f"⚙️  Ollama host: {OLLAMA_HOST}")
    print(f"⚙️  Judge LLM RAGAS: {JUDGE_MODEL}")
    print(f"⚙️  Judge embeddings RAGAS: {JUDGE_EMBEDDING_MODEL}")
    print(f"⚙️  RAGAS judge context window num_ctx={RAGAS_NUM_CTX}")
    print(f"⚙️  RAGAS judge max output num_predict={RAGAS_NUM_PREDICT}")
    print(f"⚙️  Retrieval top_k={TOP_K_RETRIEVE}, rerank top_k={TOP_K_RERANK}")
    print(f"⚙️  Max questions={RAGAS_MAX_QUESTIONS}")

    collection, bi_encoder, reranker = init_rag_pipeline()

    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if RAGAS_MAX_QUESTIONS > 0:
        test_data = test_data[:RAGAS_MAX_QUESTIONS]

    ragas_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    debug_rows = []

    print(f"🔄 Exécution de {len(test_data)} questions à travers le pipeline...")

    for idx, item in enumerate(test_data, start=1):
        question = get_question(item)
        ground_truth = get_ground_truth(item)

        if not question:
            print(f"⚠️  Question vide ignorée à l'index {idx}")
            continue

        print(f"   [{idx}/{len(test_data)}] {question[:90]}...")

        try:
            opt_query = rewrite_query(question)

            best_docs = get_best_documents_for_llm(
                opt_query,
                collection,
                bi_encoder,
                reranker,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_rerank=TOP_K_RERANK,
            )

            contexts = normalize_contexts(best_docs)

            if best_docs:
                answer = generate_legal_response(question, best_docs)
            else:
                answer = "Je suis désolé, je n'ai pas la réponse à cette question."

        except Exception as e:
            print(f"❌ Erreur sur la question {idx}: {e}")

            opt_query = ""
            best_docs = []
            contexts = ["Erreur pendant la récupération du contexte."]
            answer = "Erreur pendant la génération de la réponse."

        ragas_data["user_input"].append(question)
        ragas_data["response"].append(answer)
        ragas_data["retrieved_contexts"].append(contexts)
        ragas_data["reference"].append(ground_truth)

        debug_rows.append(
            {
                "user_input": question,
                "optimized_query": opt_query,
                "response": answer,
                "reference": ground_truth,
                "retrieved_contexts": contexts,
                "retrieved_docs_debug": [_doc_debug_row(doc) for doc in best_docs],
            }
        )

    dataset = Dataset.from_dict(ragas_data)

    print("⚖️ Lancement de l'évaluation RAGAS...")

    custom_run_config = RunConfig(
        max_workers=RAGAS_MAX_WORKERS,
        timeout=RAGAS_TIMEOUT,
    )

    result = evaluate(
        dataset=dataset,
       metrics=RAGAS_METRICS,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=custom_run_config,
    )

    df_results = result.to_pandas()

    # Add debug columns after RAGAS evaluation.
    if len(df_results) == len(debug_rows):
        df_results["optimized_query"] = [
            row["optimized_query"] for row in debug_rows
        ]
        df_results["retrieved_docs_debug"] = [
            _json_dumps_safe(row["retrieved_docs_debug"]) for row in debug_rows
        ]

    output_file = os.path.join(current_dir, "ragas_results.csv")
    detailed_output_file = os.path.join(current_dir, "ragas_results_detailed.json")

    df_results.to_csv(output_file, index=False, encoding="utf-8-sig")

    with open(detailed_output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "ollama_host": OLLAMA_HOST,
                    "judge_model": JUDGE_MODEL,
                    "judge_embedding_model": JUDGE_EMBEDDING_MODEL,
                    "ragas_num_ctx": RAGAS_NUM_CTX,
                    "ragas_num_predict": RAGAS_NUM_PREDICT,
                    "top_k_retrieve": TOP_K_RETRIEVE,
                    "top_k_rerank": TOP_K_RERANK,
                    "max_questions": RAGAS_MAX_QUESTIONS,
                    "max_workers": RAGAS_MAX_WORKERS,
                    "timeout": RAGAS_TIMEOUT,
                },
                "rows": debug_rows,
                "ragas_results": df_results.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n📊 Score Global :")
    print(result)

    print(f"\n✅ Résultats CSV sauvegardés dans {output_file}")
    print(f"✅ Résultats détaillés sauvegardés dans {detailed_output_file}")


if __name__ == "__main__":
    default_test_file = os.path.join(
        project_root,
        "data",
        "ragas_dataset",
        "test_set.json",
    )

    test_file = sys.argv[1] if len(sys.argv) > 1 else default_test_file

    run_evaluation(test_file)
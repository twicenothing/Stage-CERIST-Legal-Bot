# ── MUST BE FIRST ────────────────────────────────────────────────────────────
import sys
from types import ModuleType
from ragas.run_config import RunConfig

# Patch 1: ragas imports ChatVertexAI from langchain_community
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        from langchain_google_vertexai import ChatVertexAI as _CV
    except ImportError:
        class _CV:
            pass

    _mod = ModuleType("langchain_community.chat_models.vertexai")
    _mod.ChatVertexAI = _CV
    sys.modules["langchain_community.chat_models.vertexai"] = _mod

# Patch 2: langchain_core.exceptions missing ContextOverflowError
try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:
    import langchain_core.exceptions as _lce

    class ContextOverflowError(Exception):
        pass

    _lce.ContextOverflowError = ContextOverflowError
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from ollama import Client

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

load_dotenv(dotenv_path=Path(project_root) / ".env")

# ============================================================
# Force Mistral for this evaluation
# ============================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

ANSWER_MODEL = os.getenv("RAGAS_ANSWER_MODEL", "mistral-small3.1")
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "mistral-small3.1")
JUDGE_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text")

ANSWER_NUM_CTX = int(os.getenv("RAGAS_ANSWER_NUM_CTX", "32768"))
JUDGE_NUM_CTX = int(os.getenv("RAGAS_JUDGE_NUM_CTX", "32768"))

ANSWER_NUM_PREDICT = int(os.getenv("RAGAS_ANSWER_NUM_PREDICT", "800"))
JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "1024"))

TOP_K_RETRIEVE = int(os.getenv("RAGAS_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAGAS_TOP_K_RERANK", "4"))

MAX_CONTEXTS_FOR_LLM = int(os.getenv("RAGAS_MAX_CONTEXTS_FOR_LLM", "4"))
MAX_CHARS_PER_CONTEXT = int(os.getenv("RAGAS_MAX_CHARS_PER_CONTEXT", "3500"))
MAX_TOTAL_CONTEXT_CHARS = int(os.getenv("RAGAS_MAX_TOTAL_CONTEXT_CHARS", "18000"))

TEST_LIMIT = int(os.getenv("RAGAS_TEST_LIMIT", "0"))

# Make sure imported app code also sees Mistral if it reads LLM_MODEL.
os.environ["LLM_MODEL"] = ANSWER_MODEL

from rerank.rerank import get_best_documents_for_llm
from generate.llm_generate import init_rag_pipeline

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings


ollama_client = Client(host=OLLAMA_HOST)

judge_llm = LangchainLLMWrapper(
    ChatOllama(
        model=JUDGE_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0,
        num_ctx=JUDGE_NUM_CTX,
        num_predict=JUDGE_NUM_PREDICT,
        format="json",
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(
        model=JUDGE_EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )
)


def _json_dumps_safe(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def normalize_text(text: str) -> str:
    return " ".join(str(text or "").split())


def compact_context(text: str, question: str, max_chars: int = MAX_CHARS_PER_CONTEXT) -> str:
    """
    Keeps the most relevant window from a retrieved legal chunk.
    This prevents RAGAS and the answer LLM from receiving huge noisy chunks.
    """
    text = normalize_text(text)

    if len(text) <= max_chars:
        return text

    query_terms = [
        term.lower()
        for term in re.findall(r"\w+", question)
        if len(term) >= 4
    ]

    lower_text = text.lower()

    best_start = 0
    best_score = -1
    step = 500

    for start in range(0, len(text), step):
        window = lower_text[start:start + max_chars]
        score = sum(1 for term in query_terms if term in window)

        if score > best_score:
            best_score = score
            best_start = start

    start = max(0, best_start - 250)
    end = min(len(text), start + max_chars)

    compacted = text[start:end]

    if start > 0:
        compacted = "... " + compacted

    if end < len(text):
        compacted = compacted + " ..."

    return compacted


def build_contexts_for_llm(question: str, best_docs: list) -> list[str]:
    contexts = []

    for doc in best_docs[:MAX_CONTEXTS_FOR_LLM]:
        text = doc.get("text", "")
        compacted = compact_context(text, question)
        if compacted.strip():
            contexts.append(compacted)

    # Global safety cap
    final_contexts = []
    total_chars = 0

    for ctx in contexts:
        if total_chars + len(ctx) > MAX_TOTAL_CONTEXT_CHARS:
            remaining = MAX_TOTAL_CONTEXT_CHARS - total_chars
            if remaining > 500:
                final_contexts.append(ctx[:remaining])
            break

        final_contexts.append(ctx)
        total_chars += len(ctx)

    return final_contexts or ["Aucun contexte trouvé."]


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
        "text_chars": len(doc.get("text", "") or ""),
    }


def format_sources_for_prompt(best_docs: list, contexts: list[str]) -> str:
    blocks = []

    for i, doc in enumerate(best_docs[:len(contexts)], start=1):
        meta = doc.get("meta", {}) or {}

        source_file = meta.get("source_file", "document inconnu")
        page = meta.get("page", "Inconnue")
        parent_title = meta.get("parent_title", "Texte de loi inconnu")
        document_type = meta.get("document_type", "Extrait")
        chunking_method = meta.get("chunking_method", "")

        if chunking_method in ["table_row", "table_full"]:
            table_id = meta.get("table_id", "Tableau inconnu")
            source_label = f"{source_file}, page {page}, {table_id}"
        else:
            source_label = f"{parent_title}, page {page}, {document_type}"

        blocks.append(
            f"--- SOURCE {i}: {source_label} ---\n"
            f"{contexts[i - 1]}"
        )

    return "\n\n".join(blocks)


def generate_answer_with_mistral(question: str, best_docs: list, contexts: list[str]) -> str:
    context_text = format_sources_for_prompt(best_docs, contexts)

    system_prompt = """Tu es un assistant juridique strict spécialisé dans le Journal Officiel algérien.

Tu dois répondre UNIQUEMENT à partir des documents fournis.

Règles obligatoires :
- Ne donne aucune information générale ou externe.
- Si les documents ne contiennent pas explicitement la réponse exacte, réponds uniquement :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
- Réponds directement, sans introduction.
- Si la réponse contient une liste, cite tous les éléments présents dans la source.
- Cite toujours la source avec le type de texte, la page et l'article si disponible.
- Ne mentionne jamais un autre pays si les documents ne le mentionnent pas.
"""

    user_prompt = f"""<documents>
{context_text}
</documents>

<question>
{question}
</question>

Réponse directe :"""

    response = ollama_client.chat(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.0,
            "num_ctx": ANSWER_NUM_CTX,
            "num_predict": ANSWER_NUM_PREDICT,
        },
    )

    answer = response["message"]["content"].strip()

    if not answer:
        return "Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."

    return answer


def evaluate_safely(dataset: Dataset):
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    custom_run_config = RunConfig(
        max_workers=1,
        timeout=900,
        max_retries=3,
        max_wait=90,
    )

    try:
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=custom_run_config,
            raise_exceptions=False,
        )

    except TypeError:
        # Compatibility fallback for older RAGAS versions that do not expose raise_exceptions.
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=custom_run_config,
        )


def run_evaluation(testset_path: str):
    print("🚀 Initialisation du pipeline de test...")
    print(f"⚙️  Answer model       : {ANSWER_MODEL}")
    print(f"⚙️  Judge model        : {JUDGE_MODEL}")
    print(f"⚙️  Judge embeddings   : {JUDGE_EMBEDDING_MODEL}")
    print(f"⚙️  Answer num_ctx     : {ANSWER_NUM_CTX}")
    print(f"⚙️  Judge num_ctx      : {JUDGE_NUM_CTX}")
    print(f"⚙️  Retrieval top_k    : {TOP_K_RETRIEVE}")
    print(f"⚙️  Rerank top_k       : {TOP_K_RERANK}")
    print(f"⚙️  Max contexts       : {MAX_CONTEXTS_FOR_LLM}")
    print(f"⚙️  Max chars/context  : {MAX_CHARS_PER_CONTEXT}")
    print(f"⚙️  Max total context  : {MAX_TOTAL_CONTEXT_CHARS}")

    collection, bi_encoder, reranker = init_rag_pipeline()

    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if TEST_LIMIT > 0:
        test_data = test_data[:TEST_LIMIT]

    ragas_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    debug_rows = []

    print(f"🔄 Exécution de {len(test_data)} questions à travers le pipeline...")

    for idx, item in enumerate(test_data, start=1):
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{idx}/{len(test_data)}] {question[:100]}...")

        best_docs = get_best_documents_for_llm(
            question,
            collection,
            bi_encoder,
            reranker,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_rerank=TOP_K_RERANK,
        )

        contexts = build_contexts_for_llm(question, best_docs)

        if best_docs:
            answer = generate_answer_with_mistral(question, best_docs, contexts)
        else:
            answer = "Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."

        print(f"   Retrieved docs: {len(best_docs)}")
        print(f"   Context chars : {sum(len(c) for c in contexts)}")
        print(f"   Answer preview: {answer[:180]}")

        ragas_data["user_input"].append(question)
        ragas_data["response"].append(answer)
        ragas_data["retrieved_contexts"].append(contexts)
        ragas_data["reference"].append(ground_truth)

        debug_rows.append({
            "user_input": question,
            "optimized_query": question,
            "response": answer,
            "reference": ground_truth,
            "retrieved_contexts": contexts,
            "retrieved_docs_debug": [_doc_debug_row(doc) for doc in best_docs],
            "context_chars": sum(len(c) for c in contexts),
        })

    dataset = Dataset.from_dict(ragas_data)

    print("\n⚖️ Lancement de l'évaluation RAGAS...")
    result = evaluate_safely(dataset)

    df_results = result.to_pandas()

    if len(df_results) == len(debug_rows):
        df_results["optimized_query"] = [row["optimized_query"] for row in debug_rows]
        df_results["context_chars"] = [row["context_chars"] for row in debug_rows]
        df_results["retrieved_docs_debug"] = [
            _json_dumps_safe(row["retrieved_docs_debug"]) for row in debug_rows
        ]

    output_file = os.path.join(current_dir, "ragas_results.csv")
    detailed_output_file = os.path.join(current_dir, "ragas_results_detailed.json")

    df_results.to_csv(output_file, index=False)

    with open(detailed_output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "answer_model": ANSWER_MODEL,
                "judge_model": JUDGE_MODEL,
                "judge_embedding_model": JUDGE_EMBEDDING_MODEL,
                "answer_num_ctx": ANSWER_NUM_CTX,
                "judge_num_ctx": JUDGE_NUM_CTX,
                "top_k_retrieve": TOP_K_RETRIEVE,
                "top_k_rerank": TOP_K_RERANK,
                "max_contexts_for_llm": MAX_CONTEXTS_FOR_LLM,
                "max_chars_per_context": MAX_CHARS_PER_CONTEXT,
                "max_total_context_chars": MAX_TOTAL_CONTEXT_CHARS,
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
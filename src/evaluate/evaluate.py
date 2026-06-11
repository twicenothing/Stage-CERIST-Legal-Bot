import os
import re
import sys
import csv
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from ollama import Client


# ==============================================================================
# PATH SETUP
# ==============================================================================

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

load_dotenv(PROJECT_ROOT / ".env")


# ==============================================================================
# IMPORT YOUR CURRENT FULL-VISION PIPELINE
# ==============================================================================

"""
IMPORTANT:
Change this import if your interactive full-vision script has another name.

Expected file example:
    src/generate/llm_generate.py

It must expose:
    init_rag_pipeline
    rewrite_query
    get_best_documents_for_llm
    generate_legal_response_from_pdf_pages
    VISION_MODEL
    OLLAMA_HOST
"""

try:
    from generate.llm_generate import (
        init_rag_pipeline,
        rewrite_query,
        get_best_documents_for_llm,
        generate_legal_response_from_pdf_pages,
        VISION_MODEL,
        OLLAMA_HOST,
    )
except ImportError:
    # Fallback if your file is directly under src/generate.py or similar.
    from generate import (
        init_rag_pipeline,
        rewrite_query,
        get_best_documents_for_llm,
        generate_legal_response_from_pdf_pages,
        VISION_MODEL,
        OLLAMA_HOST,
    )


# ==============================================================================
# CONFIG
# ==============================================================================

GOLDEN_DATASET_PATH = Path(
    os.getenv(
        "GOLDEN_DATASET_PATH",
        str(PROJECT_ROOT / "data" / "golden_dataset" / "golden_dataset.json"),
    )
)

OUTPUT_DIR = Path(
    os.getenv(
        "EVAL_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "evaluation_results"),
    )
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

OUTPUT_JSON = OUTPUT_DIR / f"llm_judge_eval_{RUN_ID}.json"
OUTPUT_CSV = OUTPUT_DIR / f"llm_judge_eval_{RUN_ID}.csv"

JUDGE_MODEL = os.getenv("JUDGE_MODEL", VISION_MODEL)
GENERATION_MODEL = os.getenv("GENERATION_MODEL", VISION_MODEL)

RAG_TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
RAG_TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "4"))

JUDGE_NUM_CTX = int(os.getenv("JUDGE_NUM_CTX", "8192"))
JUDGE_TEMPERATURE = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))

# For quick tests:
# export EVAL_MAX_ROWS=10
EVAL_MAX_ROWS = int(os.getenv("EVAL_MAX_ROWS", "0"))

# Save after every N questions.
SAVE_EVERY = int(os.getenv("EVAL_SAVE_EVERY", "1"))

ollama_client = Client(host=OLLAMA_HOST)


# ==============================================================================
# HELPERS
# ==============================================================================

def normalize_stem(filename: str) -> str:
    """
    F2010012.txt -> f2010012
    F2010012.pdf -> f2010012
    """
    name = os.path.basename(str(filename or "").strip())
    stem = os.path.splitext(name)[0].lower()

    if stem.endswith("_pages"):
        stem = stem[:-6]

    if stem.endswith("_recursive"):
        stem = stem[:-10]

    return stem


def source_hit(expected_source: str, docs: list[dict]) -> bool:
    """
    Checks if expected source file appears in retrieved/reranked docs.
    """
    expected = normalize_stem(expected_source)

    if not expected:
        return False

    for doc in docs or []:
        meta = doc.get("meta", {}) or {}
        src = meta.get("source_file", "")
        got = normalize_stem(src)

        if got == expected:
            return True

    return False


def get_retrieved_sources(docs: list[dict]) -> list[dict]:
    sources = []

    for i, doc in enumerate(docs or [], start=1):
        meta = doc.get("meta", {}) or {}

        sources.append({
            "rank": i,
            "source_file": meta.get("source_file", ""),
            "page": meta.get("page", ""),
            "page_id": meta.get("page_id", ""),
            "chunking_method": meta.get("chunking_method", ""),
            "distance": doc.get("distance", None),
            "rerank_score": doc.get("rerank_score", None),
        })

    return sources


def has_required_citation_style(answer: str) -> bool:
    """
    Local simple check for your required source style.

    This does not judge correctness.
    It only checks whether the answer starts with / uses:
    - D'après le décret...
    - D'après la loi...
    - D'après l'arrêté...
    - D'après la décision...
    - D'après l'article...
    """
    answer = (answer or "").strip()

    refusal = (
        "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
        "ne contient pas cette information."
    )

    if answer == refusal:
        return True

    pattern = re.compile(
        r"(?i)\bD['’]après\s+"
        r"(l['’]article|le décret|la loi|la loi organique|l['’]arrêté|la décision)"
    )

    return bool(pattern.search(answer))


def extract_json_object(text: str) -> dict:
    """
    Robust JSON extraction from the judge response.
    """
    text = (text or "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in judge response:\n{text}")

    return json.loads(match.group(0))


def judge_answer(question: str, expected_answer: str, model_answer: str, expected_source: str, retrieved_sources: list[dict]) -> dict:
    """
    Uses the same local model as judge.
    It compares the generated answer to the golden answer and checks rule following.
    """
    judge_prompt = f"""Tu es un juge d'évaluation pour un système RAG juridique algérien.

Tu dois évaluer la réponse générée par rapport à la réponse attendue.

Tu dois vérifier deux choses principales :
1. CORRECTION FACTUELLE : est-ce que la réponse générée répond correctement à la question, en gardant le même sens que la réponse attendue ?
2. RESPECT DES RÈGLES : est-ce que la réponse respecte le style imposé :
   - réponse directe ;
   - citation juridique au début avec une formule du type "D'après le décret...", "D'après la loi...", "D'après l'arrêté...", "D'après la décision...", ou "D'après l'article..." ;
   - pas d'invention ;
   - pas de phrase d'introduction ;
   - refus exact si l'information n'est pas disponible.

Important :
- Ne sois pas trop strict sur la formulation exacte.
- Sois strict sur les nombres, noms, dates, montants, articles, conditions et listes.
- Si la réponse générée contient la bonne information mais sans la citation juridique demandée, la correction factuelle peut être bonne mais le respect des règles doit être pénalisé.
- Si la réponse générée cite une mauvaise source, pénalise le respect des règles et la fiabilité.
- Si la réponse attendue contient "[cite: ...]", ignore cette partie.

Question :
{question}

Source attendue dans le golden dataset :
{expected_source}

Réponse attendue :
{expected_answer}

Sources récupérées par le système :
{json.dumps(retrieved_sources, ensure_ascii=False, indent=2)}

Réponse générée :
{model_answer}

Réponds UNIQUEMENT avec un JSON valide, sans markdown, sans commentaire autour.

Format JSON obligatoire :
{{
  "answer_correct": true,
  "rules_followed": true,
  "citation_style_ok": true,
  "source_retrieval_ok": true,
  "factual_score": 0,
  "rules_score": 0,
  "overall_score": 0,
  "error_type": "none",
  "judge_notes": "explication courte"
}}

Barème :
- factual_score : entier de 0 à 5
  5 = totalement correct
  4 = correct mais légèrement incomplet
  3 = partiellement correct
  2 = très incomplet
  1 = presque faux
  0 = faux ou refus incorrect

- rules_score : entier de 0 à 5
  5 = respecte parfaitement le style et les citations
  4 = petit problème de style
  3 = citation présente mais faible ou incomplète
  2 = mauvaise citation ou style non respecté
  1 = presque aucune règle respectée
  0 = hallucination de source ou refus/style totalement incorrect

- overall_score : entier de 0 à 5, synthèse des deux.

error_type doit être l'un de :
"none", "wrong_answer", "incomplete_answer", "wrong_source", "missing_citation", "bad_refusal", "hallucination", "retrieval_failure", "format_failure"
"""

    response = ollama_client.chat(
        model=JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Tu es un juge strict. Tu réponds uniquement en JSON valide.",
            },
            {
                "role": "user",
                "content": judge_prompt,
            },
        ],
        think=False,
        options={
            "temperature": JUDGE_TEMPERATURE,
            "num_ctx": JUDGE_NUM_CTX,
        },
    )

    raw = response["message"]["content"]
    parsed = extract_json_object(raw)

    return parsed


def load_golden_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    # In case your file is wrapped like {"data": [...]}
    for key in ["data", "questions", "items", "examples"]:
        if key in data and isinstance(data[key], list):
            return data[key]

    raise ValueError("Unsupported golden dataset format. Expected list of objects.")


def save_results(results: list[dict], summary: dict):
    payload = {
        "summary": summary,
        "results": results,
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "index",
        "question",
        "expected_source",
        "source_hit",
        "optimized_query",
        "generated_answer",
        "expected_answer",
        "answer_correct",
        "rules_followed",
        "citation_style_ok",
        "source_retrieval_ok",
        "factual_score",
        "rules_score",
        "overall_score",
        "error_type",
        "judge_notes",
        "latency_seconds",
    ]

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })


def compute_summary(results: list[dict]) -> dict:
    total = len(results)

    if total == 0:
        return {
            "total": 0,
        }

    def avg(key):
        vals = []

        for r in results:
            try:
                vals.append(float(r.get(key, 0)))
            except Exception:
                pass

        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "total": total,
        "answer_correct_rate": round(sum(1 for r in results if r.get("answer_correct")) / total, 4),
        "rules_followed_rate": round(sum(1 for r in results if r.get("rules_followed")) / total, 4),
        "citation_style_local_rate": round(sum(1 for r in results if r.get("local_citation_style_ok")) / total, 4),
        "source_hit_rate": round(sum(1 for r in results if r.get("source_hit")) / total, 4),
        "avg_factual_score": avg("factual_score"),
        "avg_rules_score": avg("rules_score"),
        "avg_overall_score": avg("overall_score"),
        "avg_latency_seconds": avg("latency_seconds"),
        "judge_model": JUDGE_MODEL,
        "generation_model": GENERATION_MODEL,
        "golden_dataset_path": str(GOLDEN_DATASET_PATH),
        "output_json": str(OUTPUT_JSON),
        "output_csv": str(OUTPUT_CSV),
    }


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

def main():
    print("=" * 100)
    print("🚀 FULL-VISION RAG EVALUATION WITH LLM-AS-JUDGE")
    print("=" * 100)
    print(f"Golden dataset:   {GOLDEN_DATASET_PATH}")
    print(f"Output JSON:      {OUTPUT_JSON}")
    print(f"Output CSV:       {OUTPUT_CSV}")
    print(f"Generation model: {GENERATION_MODEL}")
    print(f"Judge model:      {JUDGE_MODEL}")
    print("=" * 100)

    golden_rows = load_golden_dataset(GOLDEN_DATASET_PATH)

    if EVAL_MAX_ROWS > 0:
        golden_rows = golden_rows[:EVAL_MAX_ROWS]

    print(f"📊 Rows to evaluate: {len(golden_rows)}")

    print("🛠️ Initializing RAG pipeline once...")
    collection, bi_encoder, reranker = init_rag_pipeline()

    results = []

    for idx, item in enumerate(golden_rows, start=1):
        question = str(item.get("question", "")).strip()
        expected_source = str(item.get("source", "")).strip()
        expected_answer = str(item.get("reponse", item.get("response", item.get("answer", "")))).strip()

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(golden_rows)}] Question:")
        print(question)
        print(f"Expected source: {expected_source}")
        print("=" * 100)

        start = time.time()

        result_row = {
            "index": idx,
            "question": question,
            "expected_source": expected_source,
            "expected_answer": expected_answer,
        }

        try:
            optimized_query = rewrite_query(question)

            result_row["optimized_query"] = optimized_query

            if not optimized_query or optimized_query.strip().upper() == "SKIP_OPTIMIZATION":
                generated_answer = (
                    "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
                    "ne contient pas cette information."
                )
                best_docs = []
            else:
                best_docs = get_best_documents_for_llm(
                    retrieval_query=optimized_query,
                    collection=collection,
                    bi_encoder=bi_encoder,
                    reranker=reranker,
                    top_k_retrieve=RAG_TOP_K_RETRIEVE,
                    top_k_rerank=RAG_TOP_K_RERANK,
                    rerank_query=question,
                )

                if best_docs:
                    generated_answer = generate_legal_response_from_pdf_pages(
                        question=question,
                        retrieved_docs=best_docs,
                        model_name=GENERATION_MODEL,
                    )
                else:
                    generated_answer = (
                        "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
                        "ne contient pas cette information."
                    )

            retrieved_sources = get_retrieved_sources(best_docs)
            hit = source_hit(expected_source, best_docs)
            local_citation_ok = has_required_citation_style(generated_answer)

            judge = judge_answer(
                question=question,
                expected_answer=expected_answer,
                model_answer=generated_answer,
                expected_source=expected_source,
                retrieved_sources=retrieved_sources,
            )

            latency = round(time.time() - start, 2)

            result_row.update({
                "generated_answer": generated_answer,
                "retrieved_sources": retrieved_sources,
                "source_hit": hit,
                "local_citation_style_ok": local_citation_ok,
                "latency_seconds": latency,

                "answer_correct": bool(judge.get("answer_correct", False)),
                "rules_followed": bool(judge.get("rules_followed", False)),
                "citation_style_ok": bool(judge.get("citation_style_ok", False)),
                "source_retrieval_ok": bool(judge.get("source_retrieval_ok", hit)),
                "factual_score": int(judge.get("factual_score", 0)),
                "rules_score": int(judge.get("rules_score", 0)),
                "overall_score": int(judge.get("overall_score", 0)),
                "error_type": judge.get("error_type", "unknown"),
                "judge_notes": judge.get("judge_notes", ""),
            })

            print("🤖 Generated answer:")
            print(generated_answer)
            print("⚖️ Judge:")
            print(json.dumps(judge, ensure_ascii=False, indent=2))
            print(f"🎯 Source hit: {hit} | Citation style local: {local_citation_ok} | Time: {latency}s")

        except Exception as e:
            latency = round(time.time() - start, 2)

            result_row.update({
                "generated_answer": "",
                "retrieved_sources": [],
                "source_hit": False,
                "local_citation_style_ok": False,
                "latency_seconds": latency,
                "answer_correct": False,
                "rules_followed": False,
                "citation_style_ok": False,
                "source_retrieval_ok": False,
                "factual_score": 0,
                "rules_score": 0,
                "overall_score": 0,
                "error_type": "exception",
                "judge_notes": str(e),
            })

            print(f"❌ ERROR: {e}")

        results.append(result_row)

        if idx % SAVE_EVERY == 0:
            summary = compute_summary(results)
            save_results(results, summary)
            print(f"💾 Partial results saved: {OUTPUT_JSON}")

    summary = compute_summary(results)
    save_results(results, summary)

    print("\n" + "=" * 100)
    print("🎉 EVALUATION COMPLETE")
    print("=" * 100)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 100)


if __name__ == "__main__":
    main()
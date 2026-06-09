# ── MUST BE FIRST ────────────────────────────────────────────────────────────
import sys
import re
from types import ModuleType
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI  # ← add this


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
import json
from pathlib import Path
from datetime import datetime

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
# Config
# ============================================================

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

ANSWER_MODEL = os.getenv("RAGAS_ANSWER_MODEL", os.getenv("LLM_MODEL", "qwen3:8b-q4_K_M"))
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "mistral-small3.1:latest")
JUDGE_EMBEDDING_MODEL = os.getenv("RAGAS_EMBEDDING_MODEL", "nomic-embed-text")
RAG_THINK = os.getenv("RAG_THINK", "false").lower() in ["1", "true", "yes", "on"]
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.0"))
ANSWER_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "32768"))
JUDGE_NUM_CTX = int(os.getenv("RAGAS_JUDGE_NUM_CTX", "32768"))

ANSWER_NUM_PREDICT = int(os.getenv("RAG_NUM_PREDICT", "2048"))
JUDGE_NUM_PREDICT = int(os.getenv("RAGAS_JUDGE_NUM_PREDICT", "1024"))

# Production equivalent defaults
TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "5"))

# 0 means evaluate all rows
TEST_LIMIT = int(os.getenv("RAGAS_TEST_LIMIT", "100"))

# Query rewrite enabled by default because production uses it
USE_QUERY_REWRITE = os.getenv("RAGAS_USE_QUERY_REWRITE", "1") == "1"

EXACT_REFUSAL = (
    "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
    "ne contient pas cette information."
)


JUDGE_OPENROUTER_MODEL = os.getenv("RAGAS_JUDGE_OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

# Make imported app code see the answer model if it reads LLM_MODEL
os.environ["LLM_MODEL"] = ANSWER_MODEL

from rerank.rerank import get_best_documents_for_llm
from generate.llm_generate import init_rag_pipeline
from retrieve.retrieve import get_retrieved_documents
from rerank.rerank import rerank_documents

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
    )
)

judge_embeddings = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(
        model=JUDGE_EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )
)


# ============================================================
# Query rewriting: production behavior + safety fixes
# ============================================================

def rewrite_query_for_eval(user_query: str, model_name: str = ANSWER_MODEL) -> str:
    """
    Same idea as production rewrite_query:
    raw user query -> optimized legal vector-search query.

    Fixes added:
    - no empty rewrite allowed
    - SKIP_OPTIMIZATION is handled cleanly
    - output forced to one line
    - num_predict limited so query rewrite does not ramble
    """

    original_query = (user_query or "").strip()

    if not original_query:
        return "SKIP_OPTIMIZATION"

    system_prompt = """Tu es un expert en optimisation de recherche juridique pour le Journal Officiel algérien.
Ta SEULE tâche est de reformuler la requête de l'utilisateur pour l'optimiser pour la recherche dans une base de données vectorielle.

RÈGLES STRICTES :
1. Clarifie les phrases ambiguës et utilise la terminologie juridique algérienne exacte.
2. Ajoute des synonymes pertinents dans une formulation fluide.
3. FORMAT EXIGÉ : Tu dois générer UNE SEULE ET UNIQUE PHRASE. Aucun saut de ligne, aucune liste, aucune puce.
4. Ne jamais remplacer, traduire ou développer les sigles/acronymes présents dans la question, sauf si leur signification est explicitement donnée par l'utilisateur. Conserve toujours le sigle original.

PORTE DE SORTIE (RÈGLE ABSOLUE) : 
Si l'entrée est une salutation (ex: "bonjour"), un test (ex: "test"), ou du charabia (ex: "blabla", "azerty"), renvoie UNIQUEMENT : SKIP_OPTIMIZATION

EXEMPLES DE COMPORTEMENT ATTENDU :
Entrée : "congé mat"
Sortie : Durée légale et conditions du congé de maternité pour les employées en Algérie

Entrée : "test"
Sortie : SKIP_OPTIMIZATION

Entrée : "loi sur le commerce"
Sortie : Législation et réglementation applicables aux sociétés commerciales et au droit des affaires en Algérie
"""

    user_prompt = f"""Reformule cette requête utilisateur pour la recherche en base de données :
{original_query}"""

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            think = RAG_THINK,
            options={
                "temperature": 0.0,
                "num_ctx": 4096,
                "num_predict": 120,
            },
        )

        rewritten = response["message"]["content"].strip()
        rewritten = " ".join(rewritten.split())

        print(f"✅ Requête reformulée par le LLM : {rewritten}")

        if not rewritten:
            print("⚠️ Requête reformulée vide. Utilisation de la requête originale.")
            return original_query

        if rewritten.upper() == "SKIP_OPTIMIZATION":
            return "SKIP_OPTIMIZATION"

        if len(rewritten) > 500:
            print("⚠️ Requête reformulée trop longue. Utilisation de la requête originale.")
            return original_query

        return rewritten

    except Exception as e:
        print(f"⚠️ Erreur lors de la reformulation : {e}. Utilisation de la requête originale.")
        return original_query


# ============================================================
# Helpers
# ============================================================

def _json_dumps_safe(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def estimate_tokens_from_chars(char_count: int) -> int:
    """
    Rough estimate for French/legal text.
    1 token ≈ 3.5 to 4 characters.
    """
    return int(char_count / 3.7)


def strip_cite_markers(text: str) -> str:
    """
    Removes markers like [cite: 2] from golden references.
    """
    if not text:
        return ""

    return re.sub(r"\[cite:\s*\d+\]", "", text).strip()


def normalize_source_file_to_pdf(source_file: str) -> str:
    """
    Converts internal source filenames to the real PDF filename.
    """
    if not source_file:
        return "document_inconnu.pdf"

    source_file = str(source_file)
    source_file = source_file.replace("_recursive.json", ".pdf")
    source_file = source_file.replace(".json", ".pdf")
    source_file = source_file.replace(".txt", ".pdf")

    return source_file


def build_contexts_for_ragas(best_docs: list) -> list[str]:
    """
    RAGAS expects a list of retrieved contexts.
    We pass full raw retrieved texts, no trimming.
    """
    contexts = []

    for doc in best_docs:
        text = str(doc.get("text", "") or "")
        if text.strip():
            contexts.append(text)

    return contexts or ["Aucun contexte trouvé."]


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


# ============================================================
# Production prompt copied into this evaluation script
# ============================================================

def _format_prod_llm_prompt_for_eval(query: str, best_docs: list):
    """
    This is the production prompt logic copied directly into the eval script.
    The final answer uses the ORIGINAL user question, while retrieval uses
    the optimized query.
    """

    date_du_jour = datetime.now().strftime("%d/%m/%Y")

    formatted_context = ""
    formatted_sources = []

    for i, doc in enumerate(best_docs):
        meta = doc.get("meta", {}) or {}
        text = doc.get("text", "") or ""

        source_file = meta.get("source_file", f"Document inconnu {i + 1}")
        source_file = normalize_source_file_to_pdf(source_file)

        chunking_method = meta.get("chunking_method", "")
        chunk_format = meta.get("chunk_format", "")
        page_num = meta.get("page", "Inconnu")

        if chunking_method in ["table_row", "table_full"]:
            table_id = meta.get("table_id", "Tableau inconnu")
            table_kind = meta.get("table_kind", "Tableau")
            titre_juridique = meta.get("parent_title") or table_id
            article = f"{table_kind} / {chunk_format}"
        else:
            titre_juridique = meta.get("parent_title", "Texte de loi inconnu")
            article = meta.get("document_type", "Extrait")

        raw_score = doc.get("rerank_score", 0)
        scaled_score = float(raw_score) * 100
        percentage_score = max(0, min(100, int(scaled_score)))

        formatted_sources.append({
            "doc_id": str(doc.get("id", i)),
            "score": percentage_score,
            "text": text,
            "title": source_file,
            "parent_title": titre_juridique,
            "page": page_num,
            "chunking_method": chunking_method,
            "chunk_format": chunk_format,
            "table_id": meta.get("table_id", ""),
            "table_kind": meta.get("table_kind", ""),
        })

        formatted_context += f"--- SOURCE : {titre_juridique} | PAGE : {page_num} ({article}) ---\n"
        formatted_context += f"{text}\n\n"

    system_prompt = f"""Tu es un assistant juridique strict. Aujourd'hui, nous sommes le {date_du_jour}. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.
4. Si plusieurs documents contiennent des réponses possibles ou contradictoires pour la même question, tu DOIS privilégier et formuler ta réponse en te basant EXCLUSIVEMENT sur le document le plus récent (en te fiant aux dates mentionnées dans les titres des sources).
5. Si la réponse implique une liste d'éléments, tu dois être EXHAUSTIF et n'omettre aucun élément mentionné dans la source.
6. Si la source est un tableau, exploite précisément la ligne ou le tableau fourni. Ne transforme pas les valeurs, les codes, les taux ou les libellés.
7. Si la question demande plusieurs éléments, conditions, délais, procédures, exceptions ou montants, structure la réponse en couvrant chaque élément demandé. Ne laisse aucune partie de la question sans réponse si elle est présente dans les documents.
8. Si les documents permettent de répondre seulement à une partie de la question, réponds à cette partie et précise clairement que le reste n'est pas indiqué dans les documents. N'utilise la phrase de rejet complète que si aucun élément utile de réponse n'est présent dans les documents.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.
Ne tente pas de deviner ou de déduire. Si les documents fournis parlent d'un sujet connexe mais ne répondent pas EXACTEMENT et FACTUELLEMENT à la question posée, applique la RÈGLE CRITIQUE DE REJET.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Utilise des listes à puces si nécessaire.
- Cite obligatoirement tes sources de manière naturelle (Type de texte, Numéro, Page, Article). Si la source indique "Texte de loi inconnu", utilise cette mention exacte suivie de la page et de l'article si disponible.
- Si la source est un tableau, cite le fichier ou l'identifiant du tableau, la page, et la ligne si elle est disponible.

=== EXEMPLES DE COMPORTEMENT ATTENDU ===

Exemple 1 (Information présente avec source complète) :
<documents>
--- SOURCE : Décret exécutif n° 23-64 du 14 Rajab 1444 correspondant au 5 février 2023 | PAGE : 3 (Décret) ---
Contenu : Art. 2. — La réalisation et l'exploitation d'un aérodrome destiné à l'usage privé, sont soumises à l'autorisation de l'autorité chargée de l'aviation civile.
</documents>
<question>Qui autorise la création d'un aérodrome privé ?</question>
Réponse directe :
La réalisation et l'exploitation d'un aérodrome à usage privé nécessitent l'autorisation de l'autorité chargée de l'aviation civile.
- [Source : Décret exécutif n° 23-64, Page 3, Art. 2]

Exemple 2 (Information absente) :
<documents>
--- SOURCE : Arrêté interministériel du 5 Rajab 1429 | PAGE : 5 (Arrêté) ---
Contenu : Art. 1. — Le présent arrêté fixe le tarif des redevances.
</documents>
<question>Quelle est la durée du congé maternité ?</question>
Réponse directe :
Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.

Exemple 3 (Information présente avec source inconnue) :
<documents>
--- SOURCE : Texte de loi inconnu | PAGE : 17 (Extrait) ---
Article 1er. — En application des dispositions de l'article 2 du décret exécutif n° 03-297 du 13 Rajab 1424 correspondant au 10 septembre 2003, modifié et complété, fixant les conditions et les modalités d'organisation des festivals culturels, est institutionnalisé à Adrar, le festival culturel international annuel du théâtre du Sahara.
</documents>
<question>Quelle ville a été choisie pour accueillir le festival culturel international annuel du théâtre du Sahara ?</question>
Réponse directe :
La ville choisie pour accueillir le festival culturel international annuel du théâtre du Sahara est Adrar.
- [Source : Texte de loi inconnu, Page 17, Art. 1er]
"""

    user_prompt = f"""<documents>
{formatted_context}
</documents>

<question>
{query}
</question>

Réponse directe :"""

    return system_prompt, user_prompt, formatted_sources


def generate_answer_with_prod_prompt(question: str, best_docs: list) -> str:
    """
    Generates the answer using the production prompt copied above.
    """

    system_prompt, user_prompt, _sources = _format_prod_llm_prompt_for_eval(
        question,
        best_docs,
    )

    total_prompt_chars = len(system_prompt) + len(user_prompt)
    estimated_tokens = estimate_tokens_from_chars(total_prompt_chars)

    print(f"   Answer prompt chars : {total_prompt_chars}")
    print(f"   Estimated tokens    : ~{estimated_tokens}")
    print(f"   Answer num_ctx      : {ANSWER_NUM_CTX}")

    response = ollama_client.chat(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        think=RAG_THINK,        
        options={
            "temperature": 0.0,
            "num_ctx": ANSWER_NUM_CTX,
            "num_predict": ANSWER_NUM_PREDICT,
        },
    )

    answer = response["message"]["content"].strip()

    if not answer:
        return EXACT_REFUSAL

    # Small cleanup for occasional empty source tags
    answer = re.sub(r"\s*<source>\s*</source>\s*", "", answer).strip()

    if not answer:
        return EXACT_REFUSAL

    return answer


# ============================================================
# RAGAS evaluation
# ============================================================

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
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=custom_run_config,
        )


def get_question_and_reference(item: dict):
    """
    Supports both dataset formats:
    - golden_dataset: question + reponse
    - test_set: question + ground_truth
    """

    question = item.get("question") or item.get("user_input")
    reference = (
        item.get("reponse")
        or item.get("ground_truth")
        or item.get("reference")
        or ""
    )

    if not question:
        raise ValueError(f"Missing question/user_input field in item: {item}")

    return question, strip_cite_markers(reference)


def run_evaluation(testset_path: str):
    print("🚀 Initialisation du pipeline de test...")
    print(f"⚙️  Answer model       : {ANSWER_MODEL}")
    print(f"⚙️  Judge model        : {JUDGE_MODEL}")
    print(f"⚙️  Judge embeddings   : {JUDGE_EMBEDDING_MODEL}")
    print(f"⚙️  Answer num_ctx     : {ANSWER_NUM_CTX}")
    print(f"⚙️  Judge num_ctx      : {JUDGE_NUM_CTX}")
    print(f"⚙️  Retrieval top_k    : {TOP_K_RETRIEVE}")
    print(f"⚙️  Rerank top_k       : {TOP_K_RERANK}")
    print(f"⚙️  Query rewrite      : {USE_QUERY_REWRITE}")
    print("⚙️  Prompt             : production prompt copied in this file")
    print("⚙️  Context mode       : FULL CONTEXT, no trimming")

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
        question, ground_truth = get_question_and_reference(item)
        source_file = item.get("source", "")

        print(f"\n[{idx}/{len(test_data)}] {question[:120]}...")
        if source_file:
            print(f"   Golden source file  : {source_file}")

        # ------------------------------------------------------------
        # 1. Production-style query rewrite
        # ------------------------------------------------------------
        if USE_QUERY_REWRITE:
            optimized_query = rewrite_query_for_eval(question)
        else:
            optimized_query = question

        print(f"   Optimized query     : {optimized_query}")

        # ------------------------------------------------------------
        # 2. Skip greetings/tests/gibberish before retrieval
        # ------------------------------------------------------------
        if optimized_query == "SKIP_OPTIMIZATION":
            best_docs = []
            contexts = ["Aucun contexte trouvé."]
            answer = EXACT_REFUSAL
            strategy_note = "skip_optimization"

            context_chars = sum(len(c) for c in contexts)
            estimated_context_tokens = estimate_tokens_from_chars(context_chars)

        else:
            # --------------------------------------------------------
            # 3. Production-style retrieval/rerank with optimized query
            # --------------------------------------------------------
            best_docs = get_best_documents_for_llm(
                optimized_query,
                collection,
                bi_encoder,
                reranker,
                top_k_retrieve=TOP_K_RETRIEVE,
                top_k_rerank=TOP_K_RERANK,
                rerank_query=question,
            )

            contexts = build_contexts_for_ragas(best_docs)
            context_chars = sum(len(c) for c in contexts)
            estimated_context_tokens = estimate_tokens_from_chars(context_chars)

            print(f"   Retrieved docs      : {len(best_docs)}")
            print(f"   Full context chars  : {context_chars}")
            print(f"   Est. context tokens : ~{estimated_context_tokens}")

            # --------------------------------------------------------
            # 4. Production-style generation with ORIGINAL question
            # --------------------------------------------------------
            if best_docs:
                answer = generate_answer_with_prod_prompt(question, best_docs)
            else:
                answer = EXACT_REFUSAL

            strategy_note = "prod_rewrite_retrieve_rerank_prompt"

        print(f"   Answer preview      : {answer[:220]}")

        ragas_data["user_input"].append(question)
        ragas_data["response"].append(answer)
        ragas_data["retrieved_contexts"].append(contexts)
        ragas_data["reference"].append(ground_truth)

        debug_rows.append({
            "user_input": question,
            "source_file": source_file,
            "optimized_query": optimized_query,
            "strategy_note": strategy_note,
            "response": answer,
            "reference": ground_truth,
            "retrieved_contexts": contexts,
            "retrieved_docs_debug": [_doc_debug_row(doc) for doc in best_docs],
            "context_chars": context_chars,
            "estimated_context_tokens": estimated_context_tokens,
        })

    dataset = Dataset.from_dict(ragas_data)

    print("\n⚖️ Lancement de l'évaluation RAGAS...")
    result = evaluate_safely(dataset)

    df_results = result.to_pandas()

    if len(df_results) == len(debug_rows):
        df_results["source_file"] = [row["source_file"] for row in debug_rows]
        df_results["optimized_query"] = [row["optimized_query"] for row in debug_rows]
        df_results["strategy_note"] = [row["strategy_note"] for row in debug_rows]
        df_results["context_chars"] = [row["context_chars"] for row in debug_rows]
        df_results["estimated_context_tokens"] = [
            row["estimated_context_tokens"] for row in debug_rows
        ]
        df_results["retrieved_docs_debug"] = [
            _json_dumps_safe(row["retrieved_docs_debug"]) for row in debug_rows
        ]

    output_file = os.path.join(current_dir, "ragas_results.csv")
    detailed_output_file = os.path.join(current_dir, "ragas_results_detailed.json")

    df_results.to_csv(output_file, index=False)

    with open(detailed_output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "context_mode": "full_context_no_trimming",
                    "prompt_mode": "production_prompt_copied_in_eval_file",
                    "query_rewrite": USE_QUERY_REWRITE,
                    "answer_model": ANSWER_MODEL,
                    "judge_model": JUDGE_MODEL,
                    "judge_embedding_model": JUDGE_EMBEDDING_MODEL,
                    "answer_num_ctx": ANSWER_NUM_CTX,
                    "judge_num_ctx": JUDGE_NUM_CTX,
                    "answer_num_predict": ANSWER_NUM_PREDICT,
                    "judge_num_predict": JUDGE_NUM_PREDICT,
                    "top_k_retrieve": TOP_K_RETRIEVE,
                    "top_k_rerank": TOP_K_RERANK,
                    "test_limit": TEST_LIMIT,
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
    print(f"\n✅ Résultats CSV sauvegardés dans       {output_file}")
    print(f"✅ Résultats détaillés sauvegardés dans {detailed_output_file}")


if __name__ == "__main__":
    default_test_file = os.path.join(
        project_root,
        "data",
        "golden_dataset",
        "golden_dataset.json",
    )

    test_file = sys.argv[1] if len(sys.argv) > 1 else default_test_file
    run_evaluation(test_file)
import os
import sys
import json
import csv
import argparse
from datetime import date
from pathlib import Path
from statistics import mean

# ---------------------------------------------------------
# 1. Add project paths to Python path
# ---------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from src.rerank.rerank import get_best_documents_for_llm

# Optional imports for loading models and ChromaDB
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


DEFAULT_BI_ENCODER_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
DEFAULT_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


def resolve_path(path_value):
    """
    Resolves paths passed from any working directory.
    """
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


# ---------------------------------------------------------
# 2. System prompt
# ---------------------------------------------------------
def build_system_prompt():
    date_du_jour = date.today().strftime("%d/%m/%Y")

    return f"""Tu es un assistant juridique strict. Aujourd'hui, nous sommes le {date_du_jour}. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.
4. Si plusieurs documents contiennent des réponses possibles ou contradictoires pour la même question, tu DOIS privilégier et formuler ta réponse en te basant EXCLUSIVEMENT sur le document le plus récent (en te fiant aux dates mentionnées dans les titres des sources).
5. Si la réponse implique une liste d'éléments, tu dois être EXHAUSTIF et n'omettre aucun élément mentionné dans la source.
6. Si la source est un tableau, exploite précisément la ligne ou le tableau fourni. Ne transforme pas les valeurs, les codes, les taux ou les libellés.

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


# ---------------------------------------------------------
# 3. Token estimation
# ---------------------------------------------------------
def count_tokens(text: str) -> int:
    """
    Uses tiktoken if available.
    Otherwise falls back to a rough estimation: 1 token ≈ 4 characters.
    """
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ---------------------------------------------------------
# 4. Formatting retrieved documents
# ---------------------------------------------------------
def get_doc_field(doc, key, default=""):
    """
    Safely reads fields from either doc directly or doc['metadata'].
    """
    if key in doc:
        return doc.get(key, default)

    metadata = doc.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get(key, default)

    return default


def format_documents_for_prompt(docs):
    formatted_docs = []

    for i, doc in enumerate(docs, start=1):
        text = doc.get("text") or doc.get("content") or doc.get("document") or ""

        source = (
            get_doc_field(doc, "source")
            or get_doc_field(doc, "file")
            or get_doc_field(doc, "filename")
            or "Source inconnue"
        )

        page = get_doc_field(doc, "page", "Page inconnue")
        article = get_doc_field(doc, "article", "")
        chunk_type = get_doc_field(doc, "chunk_type", "")
        rerank_score = doc.get("rerank_score", "")

        header_parts = [
            f"SOURCE : {source}",
            f"PAGE : {page}",
        ]

        if article:
            header_parts.append(f"ARTICLE : {article}")

        if chunk_type:
            header_parts.append(f"TYPE : {chunk_type}")

        if rerank_score != "":
            header_parts.append(f"RERANK_SCORE : {rerank_score:.4f}")

        header = " | ".join(header_parts)

        formatted_docs.append(
            f"--- DOCUMENT {i} | {header} ---\n"
            f"Contenu : {text}"
        )

    return "\n\n".join(formatted_docs)


def build_user_prompt(question, formatted_context):
    return f"""<documents>
{formatted_context}
</documents>

<question>
{question}
</question>
"""


# ---------------------------------------------------------
# 5. Evaluation logic
# ---------------------------------------------------------
def source_found_in_docs(expected_source, docs):
    if not expected_source:
        return False

    expected_source = expected_source.lower().strip()

    for doc in docs:
        source = (
            get_doc_field(doc, "source")
            or get_doc_field(doc, "file")
            or get_doc_field(doc, "filename")
            or ""
        )

        if expected_source in str(source).lower():
            return True

    return False


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)
    index = int((p / 100) * (len(values) - 1))
    return values[index]


def evaluate_golden_dataset(
    golden_path,
    collection,
    bi_encoder,
    reranker,
    top_k_retrieve=30,
    top_k_rerank=4,
    context_windows=None,
    output_csv="context_window_eval.csv"
):
    if context_windows is None:
        context_windows = [4096, 8192, 12000, 16000, 24000, 32000, 64000, 128000]

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)

    system_prompt = build_system_prompt()
    system_prompt_tokens = count_tokens(system_prompt)

    results = []

    for index, item in enumerate(golden_data, start=1):
        question = item["question"]
        expected_source = item.get("source", "")
        expected_answer = item.get("reponse", "")

        print(f"\n[{index}/{len(golden_data)}] Question: {question}")

        final_docs = get_best_documents_for_llm(
            query=question,
            collection=collection,
            bi_encoder=bi_encoder,
            reranker=reranker,
            top_k_retrieve=top_k_retrieve,
            top_k_rerank=top_k_rerank
        )

        formatted_context = format_documents_for_prompt(final_docs)
        user_prompt = build_user_prompt(question, formatted_context)
        full_prompt = system_prompt + "\n\n" + user_prompt

        docs_chars = len(formatted_context)
        docs_tokens = count_tokens(formatted_context)
        user_prompt_tokens = count_tokens(user_prompt)
        total_tokens = count_tokens(full_prompt)

        row = {
            "question_id": index,
            "question": question,
            "expected_source": expected_source,
            "expected_answer": expected_answer,
            "retrieved_docs_count": len(final_docs),
            "source_found_in_top_reranked": source_found_in_docs(expected_source, final_docs),
            "docs_chars": docs_chars,
            "docs_tokens_estimated": docs_tokens,
            "system_prompt_tokens_estimated": system_prompt_tokens,
            "user_prompt_tokens_estimated": user_prompt_tokens,
            "total_request_tokens_estimated": total_tokens,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
        }

        for window in context_windows:
            row[f"fits_{window}"] = total_tokens <= window
            row[f"remaining_tokens_{window}"] = window - total_tokens

        results.append(row)

        print(f"  Retrieved docs: {len(final_docs)}")
        print(f"  Context chars: {docs_chars}")
        print(f"  Context tokens estimated: {docs_tokens}")
        print(f"  Total request tokens estimated: {total_tokens}")

    save_results_to_csv(results, output_csv)

    print_summary(results, context_windows)

    return results


def save_results_to_csv(results, output_csv):
    if not results:
        print("No results to save.")
        return

    fieldnames = list(results[0].keys())

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDetailed results saved to: {output_csv}")


def print_summary(results, context_windows):
    if not results:
        return

    total_tokens = [r["total_request_tokens_estimated"] for r in results]
    docs_tokens = [r["docs_tokens_estimated"] for r in results]
    docs_chars = [r["docs_chars"] for r in results]

    source_found_count = sum(1 for r in results if r["source_found_in_top_reranked"])

    print("\n==================== SUMMARY ====================")
    print(f"Total questions: {len(results)}")
    print(f"Source found in top reranked docs: {source_found_count}/{len(results)}")
    print(f"Source recall@rerank_top_k: {source_found_count / len(results):.2%}")

    print("\n--- Context size ---")
    print(f"Average retrieved context chars: {mean(docs_chars):.0f}")
    print(f"Average retrieved context tokens: {mean(docs_tokens):.0f}")

    print("\n--- Full request size: system prompt + user prompt + documents ---")
    print(f"Average total tokens/request: {mean(total_tokens):.0f}")
    print(f"Minimum total tokens/request: {min(total_tokens)}")
    print(f"Maximum total tokens/request: {max(total_tokens)}")
    print(f"P95 total tokens/request: {percentile(total_tokens, 95)}")

    print("\n--- Context window fit ---")
    for window in context_windows:
        fit_count = sum(1 for r in results if r[f"fits_{window}"])
        fit_rate = fit_count / len(results)
        print(f"{window:>6} tokens: {fit_count}/{len(results)} fit = {fit_rate:.2%}")

    print("=================================================\n")


# ---------------------------------------------------------
# 6. Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--golden_path", required=True, help="Path to golden dataset JSON file.")
    parser.add_argument("--chroma_path", required=True, help="Path to Chroma persistent directory.")
    parser.add_argument("--collection_name", required=True, help="Chroma collection name.")

    parser.add_argument(
        "--bi_encoder_model",
        default=DEFAULT_BI_ENCODER_MODEL,
        help="Same embedding model used for indexing."
    )
    parser.add_argument(
        "--reranker_model",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-Encoder reranker model."
    )

    parser.add_argument("--top_k_retrieve", type=int, default=30)
    parser.add_argument("--top_k_rerank", type=int, default=4)

    parser.add_argument(
        "--context_windows",
        type=str,
        default="4096,8192,12000,16000,24000,32000,64000,128000",
        help="Comma-separated context windows to test."
    )

    parser.add_argument(
        "--output_csv",
        default="context_window_eval.csv",
        help="Output CSV file path."
    )

    args = parser.parse_args()

    context_windows = [int(x.strip()) for x in args.context_windows.split(",")]
    golden_path = resolve_path(args.golden_path)
    chroma_path = resolve_path(args.chroma_path)
    output_csv = resolve_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Chroma collection from: {chroma_path}")
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=args.collection_name)

    print(f"Loading bi-encoder: {args.bi_encoder_model}")
    bi_encoder = SentenceTransformer(args.bi_encoder_model)

    print(f"Loading reranker: {args.reranker_model}")
    reranker = CrossEncoder(args.reranker_model)

    evaluate_golden_dataset(
        golden_path=str(golden_path),
        collection=collection,
        bi_encoder=bi_encoder,
        reranker=reranker,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        context_windows=context_windows,
        output_csv=str(output_csv)
    )


if __name__ == "__main__":
    main()

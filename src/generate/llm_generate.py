import os
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import torch
from dotenv import load_dotenv
from ollama import Client
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==============================================================================
# 🔐 SÉCURITÉ DES CHEMINS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

ENV_PATH = Path(project_root) / ".env"
load_dotenv(dotenv_path=ENV_PATH)

try:
    from rerank.rerank import get_best_documents_for_llm
except ImportError:
    from rerank import get_best_documents_for_llm

try:
    from generate.query_parse import rewrite_query
except ImportError:
    from query_parse import rewrite_query

from retrieve.retrieve import is_table_query

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
ENV_CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")

if not os.path.isabs(ENV_CHROMA_PATH):
    clean_path = ENV_CHROMA_PATH[2:] if ENV_CHROMA_PATH.startswith("./") else ENV_CHROMA_PATH
    ABSOLUTE_CHROMA_PATH = os.path.join(project_root, clean_path)
else:
    ABSOLUTE_CHROMA_PATH = ENV_CHROMA_PATH

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral-small3.1:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

RAG_TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
RAG_TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "4"))
RAG_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "32768"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.0"))

VISION_TABLE_MODEL = os.getenv("VISION_TABLE_MODEL", LLM_MODEL)
USE_PDF_VISION_FOR_TABLES = os.getenv("USE_PDF_VISION_FOR_TABLES", "true").lower() == "true"
VISION_MAX_PAGES = int(os.getenv("VISION_MAX_PAGES", "3"))
VISION_PAGE_ZOOM = float(os.getenv("VISION_PAGE_ZOOM", "3.0"))
VISION_NUM_CTX = int(os.getenv("VISION_NUM_CTX", "32768"))

PDF_BASE_DIRS = [
    Path(project_root) / "data" / "pdf",
    Path(project_root) / "data" / "pdf_old",
]

ollama_client = Client(host=OLLAMA_HOST)


# ==============================================================================
# 🛠️ INITIALISATION DU PIPELINE RAG
# ==============================================================================
def init_rag_pipeline():
    print(f"⏳ Connexion à la base ChromaDB existante : {ABSOLUTE_CHROMA_PATH}")

    if not os.path.exists(ABSOLUTE_CHROMA_PATH):
        print(f"❌ ERREUR CRITIQUE : Le dossier de la base est introuvable à {ABSOLUTE_CHROMA_PATH}.")
        sys.exit(1)

    db_client = chromadb.PersistentClient(path=ABSOLUTE_CHROMA_PATH)

    try:
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' trouvée.")
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : Impossible de trouver la collection '{COLLECTION_NAME}'.\n{e}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"⚙️  Hardware détecté : {device.upper()}")

    print(f"⏳ Chargement du Bi-Encoder ({EMBEDDING_MODEL})...")
    bi_encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    print(f"⏳ Chargement du Cross-Encoder ({RERANKER_MODEL})...")
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )

    return collection, bi_encoder, reranker


# ==============================================================================
# 🧠 FORMATAGE DES SOURCES TEXTUELLES
# ==============================================================================
def _format_source_header(doc: dict) -> str:
    meta = doc.get("meta", {}) or {}
    chunking_method = meta.get("chunking_method", "")
    chunk_format = meta.get("chunk_format", "")
    page = meta.get("page", "Inconnu")

    if chunking_method in ["table_row", "table_full"]:
        table_id = meta.get("table_id", "Tableau inconnu")
        table_kind = meta.get("table_kind", "Tableau")
        row_index = meta.get("row_index", "")
        titre_juridique = meta.get("parent_title") or table_id

        if row_index != "":
            source_type = f"{table_kind} / {chunk_format} / ligne {row_index}"
        else:
            source_type = f"{table_kind} / {chunk_format}"
    else:
        titre_juridique = meta.get("parent_title", "Texte de loi inconnu")
        source_type = meta.get("document_type", "Extrait")

    return f"--- SOURCE : {titre_juridique} | PAGE : {page} ({source_type}) ---"


# ==============================================================================
# 👁️ OUTILS PDF VISION
# ==============================================================================
def normalize_source_stem(source_file: str) -> str:
    """
    Examples:
      F202009.txt -> f202009
      F202009.pdf -> f202009
      data/txt/F202009.txt -> f202009
    """
    name = os.path.basename(str(source_file or "").strip())
    return os.path.splitext(name)[0].lower()


def find_pdf_for_source(source_file: str):
    """
    Finds the original PDF corresponding to source_file.

    Searches recursively inside:
      data/pdf/<YEAR>/*.pdf
      data/pdf_old/<YEAR>/*.pdf

    It ignores the year folders and matches only by PDF stem.
    """
    expected_stem = normalize_source_stem(source_file)

    if not expected_stem:
        return None

    for base_dir in PDF_BASE_DIRS:
        if not base_dir.exists():
            print(f"⚠️ Dossier PDF introuvable: {base_dir}")
            continue

        for pdf_path in base_dir.rglob("*"):
            if not pdf_path.is_file():
                continue

            if pdf_path.suffix.lower() != ".pdf":
                continue

            if pdf_path.stem.lower() == expected_stem:
                return pdf_path

    return None


def render_pdf_page_to_png(pdf_path: Path, page_num: int, output_dir: Path, zoom: float = 3.0):
    """
    Renders a 1-indexed PDF page number to PNG.
    """
    doc = fitz.open(str(pdf_path))

    try:
        page_index = int(page_num) - 1

        if page_index < 0 or page_index >= len(doc):
            print(f"⚠️ Page invalide: {page_num} pour {pdf_path.name} ({len(doc)} pages)")
            return None

        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        output_path = output_dir / f"{pdf_path.stem}_page_{page_num}.png"
        pix.save(str(output_path))

        return output_path

    finally:
        doc.close()


def get_unique_source_pages_from_docs(docs, max_pages=3):
    """
    Extracts unique source_file/page pairs from reranked docs.
    Keeps reranker order.
    """
    pages = []
    seen = set()

    for doc in docs:
        meta = doc.get("meta", {}) or {}
        source_file = meta.get("source_file", "")
        page = meta.get("page", "")

        try:
            page_int = int(page)
        except Exception:
            continue

        key = (normalize_source_stem(source_file), page_int)

        if key in seen:
            continue

        seen.add(key)
        pages.append({
            "source_file": source_file,
            "page": page_int,
            "chunking_method": meta.get("chunking_method", ""),
        })

        if len(pages) >= max_pages:
            break

    return pages


def should_use_pdf_vision_route(question: str, docs) -> bool:
    """
    Use PDF-page vision route only for table-like questions or when
    final docs contain table chunks / huge tabular regex.
    """
    if not USE_PDF_VISION_FOR_TABLES:
        return False

    if is_table_query(question):
        return True

    for doc in docs:
        meta = doc.get("meta", {}) or {}
        method = meta.get("chunking_method", "")

        if method in ["table_row", "table_full"]:
            return True

        if doc.get("is_huge_tabular_regex", False):
            return True

    return False


def generate_table_answer_from_pdf_pages(question, retrieved_docs, model_name=VISION_TABLE_MODEL):
    """
    Uses retrieved docs only to locate PDF pages.
    Then sends rendered original PDF pages to the vision model.
    """
    source_pages = get_unique_source_pages_from_docs(
        retrieved_docs,
        max_pages=VISION_MAX_PAGES,
    )

    if not source_pages:
        print("⚠️ Aucune page source exploitable pour la vision. Fallback texte.")
        return generate_legal_response(question, retrieved_docs)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        image_paths = []
        page_labels = []

        for item in source_pages:
            source_file = item["source_file"]
            page_num = item["page"]

            pdf_path = find_pdf_for_source(source_file)

            if pdf_path is None:
                print(f"⚠️ PDF introuvable pour source_file={source_file}")
                continue

            print(f"📄 PDF trouvé: {pdf_path}")
            print(f"🖼️ Rendu de la page {page_num} en image...")

            image_path = render_pdf_page_to_png(
                pdf_path,
                page_num,
                tmp_dir,
                zoom=VISION_PAGE_ZOOM,
            )

            if image_path is None:
                print(f"⚠️ Impossible de rendre la page {page_num} de {pdf_path}")
                continue

            image_paths.append(str(image_path))
            page_labels.append(f"- Image {len(image_paths)} : {pdf_path.name}, page {page_num}")

        if not image_paths:
            print("⚠️ Aucune image générée. Fallback texte.")
            return generate_legal_response(question, retrieved_docs)

        page_context = "\n".join(page_labels)

        system_prompt = """Tu es un assistant juridique strict spécialisé dans la lecture de tableaux du Journal Officiel algérien.

Tu dois répondre uniquement à partir des images fournies.

Règles :
- Lis attentivement les tableaux, les lignes, les colonnes, les en-têtes et les notes.
- Préserve exactement les nombres, montants, taux, unités, dates, noms, codes et libellés.
- Si la réponse est visible, donne une réponse directe et cite le fichier et la page.
- Si l'information n'est pas visible dans les images fournies, réponds exactement :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
"""

        user_prompt = f"""Les images suivantes sont des pages originales du Journal Officiel :

{page_context}

Question :
{question}

Réponse directe :"""

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": image_paths,
                    },
                ],
                think=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": VISION_NUM_CTX,
                },
            )

            return response["message"]["content"]

        except Exception as e:
            return f"❌ Erreur vision Ollama ({OLLAMA_HOST}) : {e}"


# ==============================================================================
# 🧠 GÉNÉRATION LLM TEXTUELLE
# ==============================================================================
def generate_legal_response(question, retrieved_docs, model_name=LLM_MODEL):
    formatted_context = ""

    for doc in retrieved_docs:
        formatted_context += f"{_format_source_header(doc)}\n"
        formatted_context += f"{doc.get('text', '')}\n\n"

    date_du_jour = datetime.now().strftime("%d/%m/%Y")

    system_prompt = f"""Tu es un assistant juridique strict. Aujourd'hui, nous sommes le {date_du_jour}. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.
4. Si plusieurs documents contiennent des réponses possibles ou contradictoires pour la même question, tu DOIS privilégier et formuler ta réponse en te basant EXCLUSIVEMENT sur le document le plus récent.
5. Si la réponse implique une liste d'éléments, tu dois être EXHAUSTIF et n'omettre aucun élément mentionné dans la source.
6. Si la source est un tableau, exploite précisément la ligne ou le tableau fourni. Ne transforme pas les valeurs, les codes, les taux ou les libellés.
7. Si la question demande plusieurs éléments, conditions, délais, procédures, exceptions ou montants, structure la réponse en couvrant chaque élément demandé.
8. Si les documents permettent de répondre seulement à une partie de la question, réponds à cette partie et précise clairement que le reste n'est pas indiqué dans les documents.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.
Ne tente pas de deviner ou de déduire.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Utilise des listes à puces si nécessaire.
- Cite obligatoirement tes sources de manière naturelle.
- Si la source est un tableau, cite le fichier ou l'identifiant du tableau, la page, et la ligne si elle est disponible.
"""

    user_prompt = f"""<documents>
{formatted_context}
</documents>

<question>
{question}
</question>

Réponse directe :"""

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            think=False,
            options={
                "temperature": RAG_TEMPERATURE,
                "num_ctx": RAG_NUM_CTX,
            },
        )

        return response["message"]["content"]

    except Exception as e:
        return f"❌ Erreur de connexion à Ollama ({OLLAMA_HOST}) : {e}\nAssurez-vous qu'Ollama est lancé."


# ==============================================================================
# 🚀 MAIN
# ==============================================================================
def main():
    print("🚀 Démarrage du Legal Bot CERIST...")
    print(f"⚙️  Modèle LLM configuré : {LLM_MODEL}")
    print(f"⚙️  Modèle vision tableaux : {VISION_TABLE_MODEL}")
    print(f"⚙️  Hôte Ollama : {OLLAMA_HOST}")
    print(f"⚙️  Vision tableaux activée : {USE_PDF_VISION_FOR_TABLES}")
    print(f"📁 Dossiers PDF recherchés :")
    for base in PDF_BASE_DIRS:
        print(f"   - {base}")

    collection, bi_encoder, reranker = init_rag_pipeline()

    print("\n✅ Système prêt ! Posez vos questions.")
    print("=" * 60)

    while True:
        original_question = input("\n❓ Question juridique (ou 'q' pour quitter) : ").strip()

        if original_question.lower() == "q":
            break

        start_time = time.time()

        print("🪄  Optimisation de la requête pour la base de données...")
        optimized_question = rewrite_query(original_question)
        print(f"optimized_question: {optimized_question}")

        if not optimized_question or optimized_question.strip().upper() == "SKIP_OPTIMIZATION":
            print("\n🤖 Réponse : Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.")
            continue

        print("🔍 Recherche et reclassement des documents en cours...")
        best_docs = get_best_documents_for_llm(
            optimized_question,
            collection,
            bi_encoder,
            reranker,
            top_k_retrieve=RAG_TOP_K_RETRIEVE,
            top_k_rerank=RAG_TOP_K_RERANK,
            rerank_query=original_question,
        )

        if not best_docs:
            print("\n🤖 Réponse : Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.")
            continue

        print(f"🤖 Analyse juridique en cours par {LLM_MODEL}...")

        vision_query_check = f"{original_question} {optimized_question}"

        if should_use_pdf_vision_route(vision_query_check, best_docs):
            print(f"👁️ Analyse tabulaire depuis les pages PDF originales avec {VISION_TABLE_MODEL}...")
            reponse_llm = generate_table_answer_from_pdf_pages(original_question, best_docs)
        else:
            print(f"🤖 Analyse juridique textuelle en cours par {LLM_MODEL}...")
            reponse_llm = generate_legal_response(original_question, best_docs)

        end_time = time.time()

        print("\n" + "=" * 80)
        print("⚖️ RÉPONSE DU BOT JURIDIQUE :")
        print("=" * 80)
        print(reponse_llm)
        print("\n" + "-" * 80)
        print(f"⏱️ Temps total de traitement : {end_time - start_time:.2f} secondes")
        print("-" * 80)


if __name__ == "__main__":
    main()
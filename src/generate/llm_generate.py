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
VISION_MODEL = os.getenv("VISION_MODEL", os.getenv("VISION_TABLE_MODEL", LLM_MODEL))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

RAG_TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
RAG_TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "4"))

RAG_NUM_CTX = int(os.getenv("RAG_NUM_CTX", "32768"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.0"))

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
    bi_encoder.max_seq_length = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "8192"))

    print(f"⏳ Chargement du Cross-Encoder ({RERANKER_MODEL})...")
    reranker = CrossEncoder(
        RERANKER_MODEL,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        },
    )

    return collection, bi_encoder, reranker


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
    stem = os.path.splitext(name)[0].lower()

    return stem


def candidate_source_stems(source_file: str):
    """
    Creates possible stems for matching PDFs.

    Useful if a TXT/JSON source was renamed because of duplicate filenames:
      F2017006_2017.pdf -> also try F2017006.pdf
    """
    stem = normalize_source_stem(source_file)

    if not stem:
        return []

    candidates = [stem]

    # If duplicate output names were created like F2017006_2017,
    # also try the original stem before the final _YYYY.
    parts = stem.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        candidates.append(parts[0])

    return list(dict.fromkeys(candidates))


def find_pdf_for_source(source_file: str):
    """
    Finds the original PDF corresponding to source_file.

    Searches recursively inside:
      data/pdf/<YEAR>/*.pdf
      data/pdf_old/<YEAR>/*.pdf

    It ignores year folders and matches by PDF stem.
    """
    expected_stems = candidate_source_stems(source_file)

    if not expected_stems:
        return None

    for base_dir in PDF_BASE_DIRS:
        if not base_dir.exists():
            print(f"⚠️ Dossier PDF introuvable: {base_dir}")
            continue

        for pdf_path in base_dir.rglob("*.pdf"):
            if pdf_path.stem.lower() in expected_stems:
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
            "page_id": meta.get("page_id", ""),
            "chunking_method": meta.get("chunking_method", ""),
            "rerank_score": doc.get("rerank_score"),
            "distance": doc.get("distance"),
        })

        if len(pages) >= max_pages:
            break

    return pages


# ==============================================================================
# 🧠 PROMPT SYSTÈME
# ==============================================================================

def build_system_prompt():
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
9. Site le décret ou la loi ou la décision ainsi que l'article exacte dans ta réponse, en utilisant les informations du document. Ne cite pas d'articles ou de lois qui ne sont pas explicitement mentionnés dans les documents fournis.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.
Ne tente pas de deviner ou de déduire.

RÈGLE CRITIQUE DE CITATION DES SOURCES :
Tu dois TOUJOURS citer la source juridique au début de la phrase ou de chaque point important.

Utilise obligatoirement l'une des formules suivantes selon le type d'acte trouvé dans la page :

- Pour un décret exécutif numéroté :
  "D'après le décret exécutif n° ... du ..., ..."

- Pour un décret présidentiel numéroté :
  "D'après le décret présidentiel n° ... du ..., ..."

- Pour un décret sans numéro :
  "D'après le décret présidentiel du ..., ..."
  ou
  "D'après le décret exécutif du ..., ..."

- Pour une loi :
  "D'après la loi n° ... du ..., ..."

- Pour une loi organique :
  "D'après la loi organique n° ... du ..., ..."

- Pour un arrêté :
  "D'après l'arrêté du ..., ..."

- Pour une décision :
  "D'après la décision n° ... du ..., ..."

- Pour un article précis :
  "D'après l'article ... du décret exécutif n° ... du ..., ..."
  "D'après l'article ... de la loi n° ... du ..., ..."
  "D'après l'article ... de l'arrêté du ..., ..."

NE DIS PAS :
- "Selon le document"
- "Selon les sources"
- "Le texte indique"
- "Dans le document fourni"
- "D'après le contexte"

Tu dois citer l'acte juridique réel : décret, loi, arrêté ou décision.

Si le numéro de l'acte n'est pas visible dans la page, ne l'invente pas. Cite seulement le type et la date :
"D'après le décret présidentiel du 29 Rajab 1446 correspondant au 29 janvier 2025, ..."

Si plusieurs sources sont utilisées, chaque paragraphe ou puce doit commencer par sa propre citation.

EXEMPLES DE STYLE OBLIGATOIRE :

Exemple 1 :
Question : Quel montant est transféré au profit de la Présidence de la République ?
Réponse correcte :
D'après le décret présidentiel n° 24-440 du 29 Joumada Ethania 1446 correspondant au 31 décembre 2024, il est ouvert au profit du portefeuille de programmes de la Présidence de la République un montant de quarante-huit milliards trois cent millions de dinars (48.300.000.000 DA) en autorisations d'engagement et un montant de vingt milliards de dinars (20.000.000.000 DA) en crédits de paiement.

Exemple 2 :
Question : Que fixe le décret exécutif n° 25-60 ?
Réponse correcte :
D'après le décret exécutif n° 25-60 du 28 Rajab 1446 correspondant au 28 janvier 2025, ce texte fixe les modalités d’élaboration et d’exécution des plans de confortement priorisés visant à préserver les infrastructures et les bâtiments à valeur stratégique ou patrimoniale contre les risques de catastrophes.

Exemple 3 :
Question : Qui préside la commission nationale des plans de confortement ?
Réponse correcte :
D'après l'article 10 du décret exécutif n° 25-60 du 28 Rajab 1446 correspondant au 28 janvier 2025, la commission nationale est présidée par le ministre chargé de l'habitat ou son représentant.

Exemple 4 :
Question : Quels sont les secteurs chargés des plans de confortement ?
Réponse correcte :
D'après les articles 4 à 7 du décret exécutif n° 25-60 du 28 Rajab 1446 correspondant au 28 janvier 2025, les plans de confortement sont élaborés et exécutés par les ministères chargés de l'habitat, de la culture, des travaux publics et de l'hydraulique, chacun pour les infrastructures ou ouvrages relevant de son secteur.

Exemple 5 :
Question : Que prévoit le décret exécutif n° 25-61 ?
Réponse correcte :
D'après le décret exécutif n° 25-61 du 28 Rajab 1446 correspondant au 28 janvier 2025, ce texte fixe les missions, la composition et le fonctionnement du comité intersectoriel chargé de l’évaluation des dégâts occasionnés par la catastrophe.

Exemple 6 :
Question : Que doit contenir le plan de gestion des déchets de catastrophe ?
Réponse correcte :
D'après l'article 4 du décret exécutif n° 25-62 du 28 Rajab 1446 correspondant au 28 janvier 2025, le plan de gestion des déchets de catastrophe doit contenir notamment la classification des déchets, l’estimation de leur quantité, les procédures de prévention, l’identification des services responsables, les mesures de communication, les points de regroupement, le suivi et le contrôle des opérations de traitement, ainsi que la remise en état des points de regroupement.

Exemple 7 :
Question : Quand le plan particulier d’intervention est-il déclenché ?
Réponse correcte :
D'après l'article 23 du décret exécutif n° 25-63 du 28 Rajab 1446 correspondant au 28 janvier 2025, le plan particulier d’intervention est déclenché par le wali en cas de survenance d’une catastrophe définie par ce plan ou lorsque les moyens du plan interne d’intervention sont insuffisants pour faire face à l’accident.

Exemple 8 :
Question : Qui a été nommé directeur de l’éducation à la wilaya de Ghardaïa ?
Réponse correcte :
D'après le décret exécutif du 30 Rajab 1446 correspondant au 30 janvier 2025, M. Khatir Ghali est nommé directeur de l’éducation à la wilaya de Ghardaïa.

Exemple 9 :
Question : Qui a été nommé délégué national à la sécurité routière ?
Réponse correcte :
D'après le décret présidentiel du 29 Rajab 1446 correspondant au 29 janvier 2025, M. Djamal Younsi est nommé délégué national à la sécurité routière.

Exemple 10 :
Question : Quelles sont les conditions pour bénéficier de la VAEP ?
Réponse correcte :
D'après l'article 4 de l'arrêté du 6 Joumada Ethania 1446 correspondant au 8 décembre 2024, tout candidat à la validation des acquis de l’expérience professionnelle doit être inscrit maritime et avoir exercé une navigation effective à bord des navires de commerce et/ou des navires auxiliaires.

Exemple 11 :
Question : Quels certificats sont concernés par la VAEP ?
Réponse correcte :
D'après l'article 3 de l'arrêté du 6 Joumada Ethania 1446 correspondant au 8 décembre 2024, les certificats concernés sont le certificat d’aptitude de matelot faisant partie d’une équipe de quart à la passerelle, le certificat d’aptitude de matelot faisant partie d’une équipe de quart à la machine, et le certificat d’aptitude de matelot électrotechnicien faisant partie d’une équipe de quart à la machine.

Exemple 12 :
Question : Quelle banque est agréée par la décision n° 25-02 ?
Réponse correcte :
D'après la décision n° 25-02 du 16 Rajab 1446 correspondant au 16 janvier 2025, « T.C Ziraat Bankasi-Algeria » est agréée en qualité de succursale de banque.

Exemple 13 :
Question : Quel est le siège de T.C Ziraat Bankasi-Algeria ?
Réponse correcte :
D'après l'article 1er de la décision n° 25-02 du 16 Rajab 1446 correspondant au 16 janvier 2025, le siège de la succursale « T.C Ziraat Bankasi-Algeria » est sis au 7 rue Larbi Alik, Hydra-Alger.

Exemple 14 :
Question : Que prévoit l'arrêté du 21 janvier 2025 ?
Réponse correcte :
D'après l'arrêté du 21 Rajab 1446 correspondant au 21 janvier 2025, le festival culturel international annuel du théâtre du Sahara est institutionnalisé à Adrar.

Exemple 15 :
Question : Quelle est la réponse si l'information n'existe pas dans les pages fournies ?
Réponse correcte :
Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Commence chaque réponse par "D'après..." suivi du type exact de l'acte juridique.
- Cite le numéro de l'acte lorsqu'il est visible.
- Cite la date complète lorsqu'elle est visible.
- Cite l'article lorsqu'il est visible et pertinent.
- Utilise des listes à puces si nécessaire, mais chaque puce importante doit garder une citation claire.
"""

    return system_prompt


# ==============================================================================
# 👁️ GÉNÉRATION LLM PAR PAGES PDF
# ==============================================================================

def generate_legal_response_from_pdf_pages(question, retrieved_docs, model_name=VISION_MODEL):
    """
    Uses retrieved/reranked chunks only to locate PDF pages.
    Then sends only the rendered original PDF pages to the vision model.
    """
    source_pages = get_unique_source_pages_from_docs(
        retrieved_docs,
        max_pages=VISION_MAX_PAGES,
    )

    if not source_pages:
        print("⚠️ Aucune page source exploitable pour la vision.")
        return "Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."

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
                pdf_path=pdf_path,
                page_num=page_num,
                output_dir=tmp_dir,
                zoom=VISION_PAGE_ZOOM,
            )

            if image_path is None:
                print(f"⚠️ Impossible de rendre la page {page_num} de {pdf_path}")
                continue

            image_paths.append(str(image_path))
            page_labels.append(f"- Document : {pdf_path.name} | page : {page_num}")

        if not image_paths:
            print("⚠️ Aucune image générée.")
            return "Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."

        documents_metadata = "\n".join(page_labels)

        system_prompt = build_system_prompt()

        user_prompt = f"""<documents>
Les documents fournis sont uniquement les images des pages originales du Journal Officiel algérien.

Pages fournies :
{documents_metadata}
</documents>

<question>
{question}
</question>

Réponse directe :"""

        try:
            response = ollama_client.chat(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                        "images": image_paths,
                    },
                ],
                think=False,
                options={
                    "temperature": RAG_TEMPERATURE,
                    "num_ctx": VISION_NUM_CTX,
                },
            )

            return response["message"]["content"]

        except Exception as e:
            return f"❌ Erreur vision Ollama ({OLLAMA_HOST}) : {e}"


# ==============================================================================
# 🚀 MAIN
# ==============================================================================

def main():
    print("🚀 Démarrage du Legal Bot CERIST - Full Vision RAG...")
    print(f"⚙️  Modèle LLM configuré : {LLM_MODEL}")
    print(f"👁️  Modèle vision configuré : {VISION_MODEL}")
    print(f"⚙️  Hôte Ollama : {OLLAMA_HOST}")
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

        print("🔍 Recherche et reclassement des pages candidates en cours...")

        best_docs = get_best_documents_for_llm(
            retrieval_query=optimized_question,
            collection=collection,
            bi_encoder=bi_encoder,
            reranker=reranker,
            top_k_retrieve=RAG_TOP_K_RETRIEVE,
            top_k_rerank=RAG_TOP_K_RERANK,
            rerank_query=original_question,
        )

        if not best_docs:
            print("\n🤖 Réponse : Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.")
            continue

        print(f"👁️ Analyse des pages PDF originales avec {VISION_MODEL}...")

        reponse_llm = generate_legal_response_from_pdf_pages(
            question=original_question,
            retrieved_docs=best_docs,
            model_name=VISION_MODEL,
        )

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
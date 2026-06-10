import os
import sys
import math
import tempfile
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator

import torch
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import CrossEncoder, SentenceTransformer
from ollama import AsyncClient
from core.config import settings

# ==============================================================================
# 🔐 SÉCURITÉ DES CHEMINS
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================================================================
# 📦 IMPORTS RAG
# ==============================================================================
from src.rerank.rerank import get_best_documents_for_llm
from src.generate.query_parse import rewrite_query
from src.retrieve.retrieve import is_table_query

# ==============================================================================
# GLOBALS
# ==============================================================================
collection = None
bi_encoder = None
reranker = None


# ==============================================================================
# ⚙️ SETTINGS HELPERS
# ==============================================================================
def _get_setting(name: str, default=None):
    """
    Safe settings getter.
    Uses settings.NAME if it exists, otherwise os.getenv(NAME), otherwise default.
    This avoids breaking production if your Settings class does not yet define
    the new vision variables.
    """
    value = getattr(settings, name, None)
    if value is not None:
        return value
    return os.getenv(name, default)


def _get_bool_setting(name: str, default: bool = False) -> bool:
    value = _get_setting(name, str(default).lower())

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in ["1", "true", "yes", "y", "on"]


def _get_int_setting(name: str, default: int) -> int:
    value = _get_setting(name, str(default))

    try:
        return int(value)
    except Exception:
        return default


def _get_float_setting(name: str, default: float) -> float:
    value = _get_setting(name, str(default))

    try:
        return float(value)
    except Exception:
        return default


# ==============================================================================
# 👁️ PDF VISION CONFIG
# ==============================================================================
VISION_TABLE_MODEL = str(_get_setting("VISION_TABLE_MODEL", _get_setting("LLM_MODEL", settings.LLM_MODEL)))
USE_PDF_VISION_FOR_TABLES = _get_bool_setting("USE_PDF_VISION_FOR_TABLES", True)
VISION_MAX_PAGES = _get_int_setting("VISION_MAX_PAGES", 3)
VISION_PAGE_ZOOM = _get_float_setting("VISION_PAGE_ZOOM", 3.0)
VISION_NUM_CTX = _get_int_setting("VISION_NUM_CTX", getattr(settings, "RAG_NUM_CTX", 32768))
VISION_NUM_PREDICT = _get_int_setting("VISION_NUM_PREDICT", getattr(settings, "RAG_NUM_PREDICT", 800))


def _resolve_backend_path(path_value: str) -> Path:
    """
    Resolves paths the same way your backend usually does.

    Supports values like:
    - data/pdf
    - ./data/pdf
    - ../data/pdf
    - absolute paths
    """
    raw = Path(str(path_value))

    if raw.is_absolute():
        return raw

    candidates = [
        Path(PROJECT_ROOT) / raw,
        Path(CURRENT_DIR) / raw,
        Path(CURRENT_DIR) / ".." / raw,
        Path("../") / raw,
    ]

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    # Return the most likely path even if it does not exist, for debug output.
    return (Path(PROJECT_ROOT) / raw).resolve()


PDF_BASE_DIRS = [
    _resolve_backend_path(_get_setting("PDF_PATH", "data/pdf")),
    _resolve_backend_path(_get_setting("PDF_OLD_PATH", "data/pdf_old")),
]


# ==============================================================================
# 🛠️ INITIALISATION DU PIPELINE RAG
# ==============================================================================
async def init_rag():
    """
    Initializes the RAG pipeline components: DB, Bi-Encoder, and Cross-Encoder.
    Dynamically allocates to hardware, avoiding hardcoded device IDs.
    """
    global collection, bi_encoder, reranker

    print("Loading RAG pipeline...")

    chroma_path = os.path.join(PROJECT_ROOT, settings.CHROMA_PATH)
    client = chromadb.PersistentClient(chroma_path)
    collection = client.get_collection(settings.COLLECTION_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware detected: {device.upper()}")

    print("Loading Bi-Encoder...")
    bi_encoder = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)

    print("Loading Cross-Encoder Reranker...")
    reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker = CrossEncoder(
        reranker_model_name,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
    )

    print("Orchestrator: Pipeline ready.")
    print(f"PDF vision enabled: {USE_PDF_VISION_FOR_TABLES}")
    print(f"PDF vision model: {VISION_TABLE_MODEL}")
    print("PDF search dirs:")
    for directory in PDF_BASE_DIRS:
        print(f"  - {directory}")


# ==============================================================================
# 📄 SOURCE / PDF NAME HELPERS
# ==============================================================================
def normalize_source_file_to_pdf(source_file: str) -> str:
    """
    Converts internal chunk source filenames to the real PDF filename
    used by the frontend PDF route.

    Examples:
    - F2017006.json -> F2017006.pdf
    - F2017006.txt -> F2017006.pdf
    - F2017006_recursive.json -> F2017006.pdf
    - F2017006.pdf -> F2017006.pdf
    """
    if not source_file:
        return "document_inconnu.pdf"

    source_file = os.path.basename(str(source_file))

    source_file = source_file.replace("_recursive.json", ".pdf")
    source_file = source_file.replace(".json", ".pdf")
    source_file = source_file.replace(".txt", ".pdf")

    if not source_file.lower().endswith(".pdf"):
        source_file += ".pdf"

    return source_file


def normalize_source_stem(source_file: str) -> str:
    """
    Examples:
    - F202009.txt -> f202009
    - F202009.pdf -> f202009
    - F202009_recursive.json -> f202009
    """
    pdf_name = normalize_source_file_to_pdf(source_file)
    return Path(pdf_name).stem.lower()


def find_pdf_for_source(source_file: str):
    """
    Finds the original PDF corresponding to source_file.

    Searches recursively inside:
    - settings.PDF_PATH
    - settings.PDF_OLD_PATH

    The year folders are ignored. It matches by filename stem.
    """
    expected_stem = normalize_source_stem(source_file)

    if not expected_stem:
        return None

    for directory in PDF_BASE_DIRS:
        if not directory.exists():
            print(f"⚠️ PDF directory does not exist: {directory}")
            continue

        for pdf_path in directory.rglob("*"):
            if not pdf_path.is_file():
                continue

            if pdf_path.suffix.lower() != ".pdf":
                continue

            if pdf_path.stem.lower() == expected_stem:
                return pdf_path

    return None


def render_pdf_page_to_png(pdf_path: Path, page_num: int, output_dir: Path, zoom: float = 3.0):
    """
    Renders a 1-indexed PDF page to PNG.
    """
    try:
        page_num = int(page_num)
    except Exception:
        return None

    doc = fitz.open(str(pdf_path))

    try:
        page_index = page_num - 1

        if page_index < 0 or page_index >= len(doc):
            print(f"⚠️ Invalid page {page_num} for {pdf_path.name}. PDF has {len(doc)} pages.")
            return None

        page = doc.load_page(page_index)
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        output_path = output_dir / f"{pdf_path.stem}_page_{page_num}.png"
        pix.save(str(output_path))

        return output_path

    finally:
        doc.close()


def get_unique_source_pages_from_docs(docs, max_pages: int = 3):
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


def should_use_pdf_vision_route(query: str, optimized_query: str, best_docs) -> bool:
    """
    Uses PDF-page vision only for table-like questions or table-like final docs.
    Normal legal questions keep the existing text-generation logic.
    """
    if not USE_PDF_VISION_FOR_TABLES:
        return False

    combined_query = f"{query} {optimized_query}"

    if is_table_query(combined_query):
        return True

    for doc in best_docs:
        meta = doc.get("meta", {}) or {}
        method = meta.get("chunking_method", "")

        if method in ["table_row", "table_full"]:
            return True

        if doc.get("is_huge_tabular_regex", False):
            return True

        text = (doc.get("text", "") or "").lower()
        if "|---" in text or "tableau annexe" in text or "crédits ouverts" in text or "credits ouverts" in text:
            return True

    return False


# ==============================================================================
# 🧠 FORMATAGE DES PROMPTS TEXTUELS EXISTANTS
# ==============================================================================
def _format_llm_prompt(query, best_docs):
    """
    Constructs the prompt using your exact system prompt logic,
    including page numbers, table metadata, and natural legal titles for the LLM.

    IMPORTANT:
    This function still returns formatted_sources for the frontend.
    """
    date_du_jour = datetime.now().strftime("%d/%m/%Y")

    formatted_context = ""
    formatted_sources = []

    for i, doc in enumerate(best_docs):
        meta = doc.get("meta", {})
        text = doc.get("text", "")

        source_file = meta.get("source_file", f"Document inconnu {i+1}")
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


def _format_pdf_vision_prompt(query: str, page_labels: list[str]):
    """
    Prompt used only for the PDF-page vision route.
    Does not modify the normal text-generation prompt.
    """
    page_context = "\n".join(page_labels)

    system_prompt = """Tu es un assistant juridique strict spécialisé dans la lecture de tableaux du Journal Officiel algérien.

Tu dois répondre uniquement à partir des images fournies.

Règles :
1. Lis attentivement les tableaux, les lignes, les colonnes, les en-têtes, les titres et les notes.
2. Préserve exactement les nombres, montants, taux, unités, dates, noms, codes, libellés et signes.
3. Si plusieurs pages sont fournies, cherche la réponse dans toutes les pages, mais ne mélange pas deux lignes différentes.
4. Si la réponse est visible, donne une réponse directe et cite le fichier et la page.
5. Si l'information n'est pas visible dans les images fournies, réponds exactement :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
6. N'explique pas ton raisonnement. Commence directement la réponse.
7. Site le décret ou la loi ou la décision ou l'arrêté en citant le numéro, la page, et l'article si possible. Si c'est un tableau, cite le nom du tableau, la page, et la ligne si possible.
"""

    user_prompt = f"""Les images suivantes sont des pages originales du Journal Officiel :

{page_context}

Question :
{query}

Réponse directe :"""

    return system_prompt, user_prompt


async def _stream_pdf_vision_answer(query: str, best_docs, client: AsyncClient) -> AsyncGenerator[dict, None]:
    """
    Uses retrieved docs only to locate source pages.
    Then streams an answer from rendered original PDF page images.
    """
    source_pages = get_unique_source_pages_from_docs(
        best_docs,
        max_pages=VISION_MAX_PAGES,
    )

    if not source_pages:
        print("⚠️ Vision route selected, but no source pages found. Falling back to text route.")
        system_prompt, user_prompt, _ = _format_llm_prompt(query, best_docs)

        async for part in await client.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            think=settings.RAG_THINK,
            options={
                "temperature": settings.RAG_TEMPERATURE,
                "num_ctx": settings.RAG_NUM_CTX,
                "num_predict": settings.RAG_NUM_PREDICT,
            },
        ):
            token = part["message"]["content"]
            if token:
                yield {"type": "chunk", "text": token}

        return

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
            page_labels.append(f"- Image {len(image_paths)} : {pdf_path.name}, page {page_num}")

        if not image_paths:
            print("⚠️ Vision route selected, but no images generated. Falling back to text route.")
            system_prompt, user_prompt, _ = _format_llm_prompt(query, best_docs)

            async for part in await client.chat(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=True,
                think=settings.RAG_THINK,
                options={
                    "temperature": settings.RAG_TEMPERATURE,
                    "num_ctx": settings.RAG_NUM_CTX,
                    "num_predict": settings.RAG_NUM_PREDICT,
                },
            ):
                token = part["message"]["content"]
                if token:
                    yield {"type": "chunk", "text": token}

            return

        system_prompt, user_prompt = _format_pdf_vision_prompt(query, page_labels)

        print(f"👁️ PDF vision route active. Sending {len(image_paths)} page image(s) to {VISION_TABLE_MODEL}.")

        async for part in await client.chat(
            model=VISION_TABLE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": image_paths,
                },
            ],
            stream=True,
            think=settings.RAG_THINK,
            options={
                "temperature": settings.RAG_TEMPERATURE,
                "num_ctx": VISION_NUM_CTX,
                "num_predict": VISION_NUM_PREDICT,
            },
        ):
            token = part["message"]["content"]
            if token:
                yield {"type": "chunk", "text": token}


# ==============================================================================
# 🚀 STREAM MAIN
# ==============================================================================
async def stream_legal_answer(query: str) -> AsyncGenerator[dict, None]:
    """
    The main generator called by the FastAPI router.

    Frontend contract preserved:
    1. optimized_query
    2. sources
    3. chunk stream
    """

    refusal_message = (
        "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
        "ne contient pas cette information."
    )

    optimized_query = rewrite_query(query)

    yield {"type": "optimized_query", "text": optimized_query}

    if not optimized_query or optimized_query.strip().upper() == "SKIP_OPTIMIZATION":
        print("⛔ SKIP_OPTIMIZATION detected. Skipping retrieval and LLM generation.")
        yield {"type": "sources", "sources": []}
        yield {"type": "chunk", "text": refusal_message}
        return

    best_docs = get_best_documents_for_llm(
        optimized_query,
        collection,
        bi_encoder,
        reranker,
        top_k_retrieve=settings.RAG_TOP_K_RETRIEVE,
        top_k_rerank=settings.RAG_TOP_K_RERANK,
        rerank_query=query,
    )

    if not best_docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "chunk", "text": refusal_message}
        return

    # Always build sources the same way for the frontend.
    system_prompt, user_prompt, sources = _format_llm_prompt(query, best_docs)

    # Emit sources first, exactly like before.
    yield {"type": "sources", "sources": sources}

    client = AsyncClient(host=settings.OLLAMA_HOST)

    use_pdf_vision = should_use_pdf_vision_route(
        query=query,
        optimized_query=optimized_query,
        best_docs=best_docs,
    )

    if use_pdf_vision:
        async for event in _stream_pdf_vision_answer(query, best_docs, client):
            yield event
        return

    print(settings.LLM_MODEL)

    async for part in await client.chat(
        model=settings.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
        think=settings.RAG_THINK,
        options={
            "temperature": settings.RAG_TEMPERATURE,
            "num_ctx": settings.RAG_NUM_CTX,
            "num_predict": settings.RAG_NUM_PREDICT,
        },
    ):
        token = part["message"]["content"]
        if token:
            yield {"type": "chunk", "text": token}
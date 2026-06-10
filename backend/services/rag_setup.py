import os
import sys
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
# 👁️ FULL PDF VISION CONFIG
# ==============================================================================

VISION_MODEL = str(
    _get_setting(
        "VISION_MODEL",
        _get_setting("VISION_TABLE_MODEL", _get_setting("LLM_MODEL", settings.LLM_MODEL)),
    )
)

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
    bi_encoder.max_seq_length = _get_int_setting("EMBEDDING_MAX_SEQ_LENGTH", 8192)

    print("Loading Cross-Encoder Reranker...")
    reranker_model_name = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

    reranker = CrossEncoder(
        reranker_model_name,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32
        },
    )

    print("Orchestrator: Pipeline ready.")
    print("PDF full-vision mode: enabled")
    print(f"PDF vision model: {VISION_MODEL}")
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
    - F2017006_pages.json -> F2017006.pdf
    - F2017006.pdf -> F2017006.pdf
    """
    if not source_file:
        return "document_inconnu.pdf"

    source_file = os.path.basename(str(source_file))

    source_file = source_file.replace("_recursive.json", ".pdf")
    source_file = source_file.replace("_pages.json", ".pdf")
    source_file = source_file.replace(".json", ".pdf")
    source_file = source_file.replace(".txt", ".pdf")

    if not source_file.lower().endswith(".pdf"):
        source_file += ".pdf"

    return source_file


def normalize_source_stem(source_file: str) -> str:
    pdf_name = normalize_source_file_to_pdf(source_file)
    return Path(pdf_name).stem.lower()


def candidate_source_stems(source_file: str):
    """
    Allows matching duplicate exported names like:
    F2017006_2017.pdf -> F2017006.pdf
    """
    stem = normalize_source_stem(source_file)

    if not stem:
        return []

    candidates = [stem]

    parts = stem.rsplit("_", 1)

    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        candidates.append(parts[0])

    return list(dict.fromkeys(candidates))


def find_pdf_for_source(source_file: str):
    """
    Finds the original PDF corresponding to source_file.

    Searches recursively inside:
    - settings.PDF_PATH
    - settings.PDF_OLD_PATH

    The year folders are ignored. It matches by filename stem.
    """
    expected_stems = candidate_source_stems(source_file)

    if not expected_stems:
        return None

    for directory in PDF_BASE_DIRS:
        if not directory.exists():
            print(f"⚠️ PDF directory does not exist: {directory}")
            continue

        for pdf_path in directory.rglob("*.pdf"):
            if pdf_path.stem.lower() in expected_stems:
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
            "page_id": meta.get("page_id", ""),
            "rerank_score": doc.get("rerank_score", ""),
            "distance": doc.get("distance", ""),
        })

        if len(pages) >= max_pages:
            break

    return pages


# ==============================================================================
# 🧠 SYSTEM PROMPT
# ==============================================================================

def _build_system_prompt():
    date_du_jour = datetime.now().strftime("%d/%m/%Y")

    return f"""Tu es un assistant juridique strict. Aujourd'hui, nous sommes le {date_du_jour}. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES :
1. Ne fais aucune introduction.
2. N'explique pas ton raisonnement.
3. Commence directement la réponse.
4. Si plusieurs documents répondent à la même question, privilégie le document le plus récent.
5. Si la réponse implique une liste, sois exhaustif et n'omets aucun élément visible dans la source.
6. Si la source est un tableau, respecte exactement les valeurs, codes, montants, taux, unités et libellés.
7. Si les documents ne répondent qu'à une partie de la question, réponds à cette partie et précise clairement que le reste n'est pas indiqué.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents fournis, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."

RÈGLE CRITIQUE DE CITATION :
Tu dois TOUJOURS citer l'acte juridique utilisé au début de la réponse ou au début de chaque point important.

Utilise obligatoirement ces formes :

- "D'après le décret exécutif n° ... du ..., ..."
- "D'après le décret présidentiel n° ... du ..., ..."
- "D'après le décret exécutif du ..., ..."
- "D'après le décret présidentiel du ..., ..."
- "D'après la loi n° ... du ..., ..."
- "D'après la loi organique n° ... du ..., ..."
- "D'après l'arrêté du ..., ..."
- "D'après la décision n° ... du ..., ..."
- "D'après l'article ... du/de la ..., ..."

Si le numéro de l'acte est visible, cite-le.
Si l'article est visible et pertinent, cite-le.
Si le numéro n'est pas visible, cite seulement le type d'acte et la date.
N'invente jamais un numéro, une date ou un article.

INTERDIT :
- "Selon le document"
- "Selon les sources"
- "Le texte indique"
- "Dans le document fourni"
- "D'après le contexte"

EXEMPLES DE STYLE :

Question : Que fixe le décret exécutif n° 25-60 ?
Réponse correcte :
D'après le décret exécutif n° 25-60 du 28 Rajab 1446 correspondant au 28 janvier 2025, ce texte fixe les modalités d’élaboration et d’exécution des plans de confortement priorisés visant à préserver les infrastructures et les bâtiments à valeur stratégique ou patrimoniale contre les risques de catastrophes.

Question : Qui préside la commission nationale ?
Réponse correcte :
D'après l'article 10 du décret exécutif n° 25-60 du 28 Rajab 1446 correspondant au 28 janvier 2025, la commission nationale est présidée par le ministre chargé de l'habitat ou son représentant.

Question : Quelles sont les conditions pour bénéficier de la VAEP ?
Réponse correcte :
D'après l'article 4 de l'arrêté du 6 Joumada Ethania 1446 correspondant au 8 décembre 2024, tout candidat à la validation des acquis de l’expérience professionnelle doit être inscrit maritime et avoir exercé une navigation effective à bord des navires de commerce et/ou des navires auxiliaires.

Question : Quelle banque est agréée ?
Réponse correcte :
D'après la décision n° 25-02 du 16 Rajab 1446 correspondant au 16 janvier 2025, « T.C Ziraat Bankasi-Algeria » est agréée en qualité de succursale de banque.

Question : Qui a été nommé à une fonction ?
Réponse correcte :
D'après le décret présidentiel du 29 Rajab 1446 correspondant au 29 janvier 2025, M. Djamal Younsi est nommé délégué national à la sécurité routière.

FORMAT FINAL :
- Réponse directe.
- Citation juridique dès le début.
- Pas d'introduction.
- Pas de raisonnement.
- Pas d'information hors documents.
"""


# ==============================================================================
# 🧠 FORMATAGE DES SOURCES POUR FRONTEND
# ==============================================================================

def _format_llm_prompt(query, best_docs):
    """
    Builds:
    - system prompt
    - fallback text prompt
    - formatted_sources for the frontend

    IMPORTANT:
    In the new full-vision version, formatted_sources is still required
    for the frontend. The LLM itself receives only rendered PDF pages.
    """
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

        titre_juridique = meta.get("parent_title") or meta.get("page_id") or "Page du Journal Officiel"
        article = meta.get("document_type") or chunking_method or "Extrait"

        raw_score = doc.get("rerank_score", 0)

        try:
            scaled_score = float(raw_score) * 100
            percentage_score = max(0, min(100, int(scaled_score)))
        except Exception:
            percentage_score = 0

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

    system_prompt = _build_system_prompt()

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
    Prompt used for the full PDF-page vision route.
    """
    page_context = "\n".join(page_labels)

    system_prompt = _build_system_prompt()

    user_prompt = f"""<documents>
Les documents fournis sont uniquement les images des pages originales du Journal Officiel algérien.

Pages fournies :
{page_context}
</documents>

<question>
{query}
</question>

Réponse directe :"""

    return system_prompt, user_prompt


# ==============================================================================
# 👁️ STREAM PDF VISION ANSWER
# ==============================================================================

async def _stream_pdf_vision_answer(query: str, best_docs, client: AsyncClient) -> AsyncGenerator[dict, None]:
    """
    Uses retrieved/reranked chunks only to locate source pages.
    Then streams an answer from rendered original PDF page images.

    The LLM receives only:
    - system prompt
    - page labels
    - rendered PDF page images
    - user question
    """
    refusal_message = (
        "Je suis désolé, je n'ai pas la réponse à cette question car la base de données "
        "ne contient pas cette information."
    )

    source_pages = get_unique_source_pages_from_docs(
        best_docs,
        max_pages=VISION_MAX_PAGES,
    )

    if not source_pages:
        print("⚠️ No source pages found for full-vision route.")
        yield {"type": "chunk", "text": refusal_message}
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
            print("⚠️ No images generated for full-vision route.")
            yield {"type": "chunk", "text": refusal_message}
            return

        system_prompt, user_prompt = _format_pdf_vision_prompt(query, page_labels)

        print(f"👁️ Full PDF vision route active. Sending {len(image_paths)} page image(s) to {VISION_MODEL}.")

        async for part in await client.chat(
            model=VISION_MODEL,
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
        retrieval_query=optimized_query,
        collection=collection,
        bi_encoder=bi_encoder,
        reranker=reranker,
        top_k_retrieve=settings.RAG_TOP_K_RETRIEVE,
        top_k_rerank=settings.RAG_TOP_K_RERANK,
        rerank_query=query,
    )

    if not best_docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "chunk", "text": refusal_message}
        return

    # Still build sources the same way for the frontend.
    # The LLM will NOT receive this text in the full-vision route.
    _, _, sources = _format_llm_prompt(query, best_docs)

    # Emit sources first, exactly like before.
    yield {"type": "sources", "sources": sources}

    client = AsyncClient(host=settings.OLLAMA_HOST)

    async for event in _stream_pdf_vision_answer(query, best_docs, client):
        yield event
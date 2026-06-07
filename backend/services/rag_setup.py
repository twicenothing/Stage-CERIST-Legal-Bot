import os
import sys
import math
import torch
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import AsyncGenerator
from ollama import AsyncClient
from core.config import settings
from datetime import datetime
# 1. Path Setup: Ensure the backend can see the 'src' directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 2. Modular Imports from your workstation 'src'
from src.rerank.rerank import get_best_documents_for_llm
from src.generate.query_parse import rewrite_query

# Global variables for models
collection = None
bi_encoder = None
reranker = None

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
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    )
    
    print("Orchestrator: Pipeline ready.")



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

    source_file = str(source_file)

    # Recursive chunks can come from files like F2017006_recursive.json
    source_file = source_file.replace("_recursive.json", ".pdf")

    # Regex chunks can come from files like F2017006.json
    source_file = source_file.replace(".json", ".pdf")

    # Regex/recursive metadata can also come from TXT source files
    source_file = source_file.replace(".txt", ".pdf")

    return source_file


def _format_llm_prompt(query, best_docs):
    """
    Constructs the prompt using your exact system prompt logic,
    now including page numbers, table metadata, and natural legal titles for the LLM.
    """
    # Must be defined before system_prompt, even if best_docs is empty or loop changes.
    date_du_jour = datetime.now().strftime("%d/%m/%Y")

    formatted_context = ""
    formatted_sources = []

    for i, doc in enumerate(best_docs):
        meta = doc.get("meta", {})
        text = doc.get("text", "")

        # 1. Frontend info:
        # Keep the real PDF filename for the /pdf endpoint routing.
        source_file = meta.get("source_file", f"Document inconnu {i+1}")
        source_file = normalize_source_file_to_pdf(source_file)

        # 2. Common metadata
        chunking_method = meta.get("chunking_method", "")
        chunk_format = meta.get("chunk_format", "")
        page_num = meta.get("page", "Inconnu")

        # 3. LLM-readable source label
        # Table chunks usually do not have parent_title/document_type,
        # so we expose table_id/table_kind instead of "Texte de loi inconnu".
        if chunking_method in ["table_row", "table_full"]:
            table_id = meta.get("table_id", "Tableau inconnu")
            table_kind = meta.get("table_kind", "Tableau")
            titre_juridique = meta.get("parent_title") or table_id
            article = f"{table_kind} / {chunk_format}"
        else:
            titre_juridique = meta.get("parent_title", "Texte de loi inconnu")
            article = meta.get("document_type", "Extrait")

        # Convert BAAI/bge-reranker-v2-m3 scores to percentage.
        # This model often outputs probabilities (~0.0 to ~1.0).
        raw_score = doc.get("rerank_score", 0)
        scaled_score = float(raw_score) * 100
        percentage_score = max(0, min(100, int(scaled_score)))

        # The frontend receives the PDF filename in "title" for PDF routing.
        # parent_title stays available for prettier display.
        formatted_sources.append({
            "doc_id": str(doc.get("id", i)),
            "score": percentage_score,
            "text": text,
            "title": source_file,  # Must stay PDF filename for the frontend router.
            "parent_title": titre_juridique,
            "page": page_num,
            "chunking_method": chunking_method,
            "chunk_format": chunk_format,
            "table_id": meta.get("table_id", ""),
            "table_kind": meta.get("table_kind", ""),
        })

        # Context passed to the LLM.
        formatted_context += f"--- SOURCE : {titre_juridique} | PAGE : {page_num} ({article}) ---\n"
        formatted_context += f"{text}\n\n"

    # 🔥 Prompt système mis à jour avec les exemples Few-Shot naturels
    system_prompt = f"""Tu es un assistant juridique strict. Aujourd'hui, nous sommes le {date_du_jour}. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.
4. Si plusieurs documents contiennent des réponses possibles ou contradictoires pour la même question, tu DOIS privilégier et formuler ta réponse en te basant EXCLUSIVEMENT sur le document le plus récent (en te fiant aux dates mentionnées dans les titres des sources).
5. Si la réponse implique une liste d'éléments, tu dois être EXHAUSTIF et n'omettre aucun élément mentionné dans la source.
6. Si la réponse contien plusieurs éléments, tu dois les citer tous n'omettre aucun élément mentionné dans la source.
7. Si la question est : "SKIP_OPTIMIZATION" alors tu dois rien ecrire d'ature que "Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.".

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Utilise des listes à puces si nécessaire.
- Cite obligatoirement tes sources de manière naturelle (Type de texte, Numéro, Page, Article). Si la source indique "Texte de loi inconnu", utilise cette mention exacte suivie de la page et de l'article si disponible.
- Si la source est un tableau, cite le fichier, la page et le tableau ou la ligne concernée si elle est disponible.

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


async def stream_legal_answer(query: str) -> AsyncGenerator[dict, None]:
    """
    The main generator called by the FastAPI router.
    """

    optimized_query = rewrite_query(query)

    yield {"type": "optimized_query", "text": optimized_query}
    # 1. Modular Retrieval & Reranking using workstation src
    best_docs = get_best_documents_for_llm(
        optimized_query, 
        collection, 
        bi_encoder, 
        reranker,
        top_k_retrieve=30, 
        top_k_rerank=4
    )

    if not best_docs:
        yield {"type": "sources", "sources": []}
        yield {"type": "chunk", "text": "Les documents fournis ne contiennent pas cette information."}
        return

    # 2. Build the exact prompts
    system_prompt, user_prompt, sources = _format_llm_prompt(query, best_docs)

    # 3. Emit sources first (Frontend requirement)
    yield {"type": "sources", "sources": sources}

    # 4. Stream tokens from Ollama
    client = AsyncClient(host=settings.OLLAMA_HOST)
    print(settings.LLM_MODEL)
    async for part in await client.chat(
        model=settings.LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        stream=True,
        think=False,
            options={
                "temperature": 0.0, # 0.0 empêche le modèle d'être "créatif" et le force à respecter les règles
                "num_ctx": 32768
            }
    ):
        token = part["message"]["content"]
        if token:
            yield {"type": "chunk", "text": token}
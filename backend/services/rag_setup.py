import os
import sys
import math
import torch
import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from typing import AsyncGenerator
from ollama import AsyncClient
from core.config import settings

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


def _format_llm_prompt(query, best_docs):
    """
    Constructs the prompt using your exact system prompt logic,
    now including page numbers for both the frontend and the LLM.
    """
    formatted_context = ""
    formatted_sources = []
    
    for i, doc in enumerate(best_docs):
        meta = doc.get('meta', {})
        text = doc.get('text', '')
        
        # Prepare source info for frontend and LLM context
        source_title = meta.get('source_file', f'Document inconnu {i+1}')
        source_title = source_title.replace('.json', '.pdf')
        article = meta.get('document_type', 'Extrait')
        
        # 🎯 RÉCUPÉRATION DE LA PAGE DEPUIS LES MÉTADONNÉES
        page_num = meta.get('page', 'Inconnu')
        
        # Convert rerank score to % using sigmoid
        SCALING_FACTOR = 2.5
        raw_score = doc.get('rerank_score', 0)
        calibrated_score = raw_score * SCALING_FACTOR
        percentage_score = min(100, int((1 / (1 + math.exp(-calibrated_score))) * 100))
        
        formatted_sources.append({
            "doc_id": str(doc.get("id", i)),
            "score": percentage_score,
            "text": text,
            "title": source_title,
            "page": page_num,  # 👈 AJOUT POUR LE FRONTEND
        })
        
        # 👈 AJOUT DE LA PAGE DANS LE CONTEXTE TEXTUEL POUR LE LLM
        formatted_context += f"--- SOURCE : {source_title} | PAGE : {page_num} ({article}) ---\n"
        formatted_context += f"{text}\n\n"

    # 🔥 Prompt système mis à jour pour EXIGER la citation de la page
    system_prompt = """Tu es un assistant juridique strict. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Utilise des listes à puces si nécessaire.
- Cite obligatoirement tes sources ET LA PAGE à la fin de l'affirmation (ex: [Source : F2002002.pdf, Page 3, Art. 4])."""

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

    # 1. Modular Retrieval & Reranking using workstation src
    best_docs = get_best_documents_for_llm(
        optimized_query, 
        collection, 
        bi_encoder, 
        reranker,
        top_k_retrieve=20, 
        top_k_rerank=8
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
    async for part in await client.chat(
        model=settings.LLM_MODEL,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ],
        stream=True,
        options={"temperature": 0.0,
        "num_ctx": 8192}
    ):
        token = part["message"]["content"]
        if token:
            yield {"type": "chunk", "text": token}
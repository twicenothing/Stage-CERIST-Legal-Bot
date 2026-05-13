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

    chroma_path = os.path.join("..", settings.CHROMA_PATH)
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
    Constructs the prompt using your exact system prompt logic.
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
        
        # Convert rerank score to % using sigmoid
        raw_score = doc.get('rerank_score', 0)
        percentage_score = min(100, int((1 / (1 + math.exp(-raw_score))) * 100))
        
        formatted_sources.append({
            "doc_id": str(doc.get("id", i)),
            "score": percentage_score,
            "text": text,
            "title": source_title,
        })
        
        formatted_context += f"--- SOURCE : {source_title} ({article}) ---\n"
        formatted_context += f"{text}\n\n"

    system_prompt = """Tu es un assistant juridique expert en droit administratif algérien. Ta seule et unique mission est de répondre aux questions de l'utilisateur en te basant STRICTEMENT et EXCLUSIVEMENT sur les documents de référence fournis.

MÉTHODOLOGIE OBLIGATOIRE :
Tu es un modèle de raisonnement. Tu vas automatiquement utiliser ta balise <think> pour réfléchir avant de répondre. 
Dans cette réflexion :
1. Identifie les concepts clés de la question.
2. Cherche attentivement ces concepts dans les documents fournis.
3. Détermine si l'information s'y trouve réellement.

RÈGLES POUR LA RÉPONSE FINALE (Après la balise <think>) :
1. RÈGLE CRITIQUE : Si ton analyse conclut que la réponse ne figure pas dans les documents, ta réponse finale DOIT être EXACTEMENT : "Les documents fournis ne contiennent pas cette information."
2. Si la réponse s'y trouve, réponds de manière directe, factuelle et précise.
3. Cite toujours le numéro de l'Article ou l'intitulé du texte de loi justifiant ta réponse."""

    user_prompt = f"Voici les documents de référence :\n\n{formatted_context}\nQuestion : {query}\n\nRéponse :"
    
    return system_prompt, user_prompt, formatted_sources


async def stream_legal_answer(query: str) -> AsyncGenerator[dict, None]:
    """
    The main generator called by the FastAPI router.
    """
    # 1. Modular Retrieval & Reranking using workstation src
    best_docs = get_best_documents_for_llm(
        query, 
        collection, 
        bi_encoder, 
        reranker,
        top_k_retrieve=20, 
        top_k_rerank=3
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
        options={"temperature": 0.0}
    ):
        token = part["message"]["content"]
        if token:
            yield {"type": "chunk", "text": token}
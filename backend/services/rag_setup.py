import chromadb
from sentence_transformers import SentenceTransformer
from core.config import settings
import string
from rank_bm25 import BM25Okapi
import numpy as np
from ollama import AsyncClient

chroma_collection = None
embedding_model = None
bm25_index = None
corpus_ids = []
corpus_docs = []
corpus_metadatas = []


def normalize_text(text: str):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def reciprocal_rank_fusion(results_dict, weights=None, k=60):
    if weights is None:
        weights = {system: 1.0 for system in results_dict.keys()}
        
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        weight = weights.get(system_name, 1.0)
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "text": doc_text, "meta": metadata}
            fused_scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
            
    return sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)


async def init_rag():
    global chroma_collection, embedding_model, bm25_index, corpus_docs, corpus_ids, corpus_metadatas
    print("Connexion a ChromaDB")
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    chroma_collection = client.get_collection(settings.COLLECTION_NAME)
    print("Chargement du model d'embedding")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL, device = 'cuda')
    print("Chargement de l'index BM25...")
    all_docs = chroma_collection.get()
    corpus_docs = all_docs['documents']
    corpus_ids = all_docs['ids']
    corpus_metadatas = all_docs['metadatas']
    
    tokenized_corpus = [normalize_text(doc) for doc in corpus_docs]
    bm25_index = BM25Okapi(tokenized_corpus)
    print("Systeme pret")



async def get_legal_answer(query:str):
    q_embed = embedding_model.encode([query]).tolist()
    vec_res = chroma_collection.query(query_embeddings=q_embed, n_results=10)
    vec_list = []
    if vec_res['ids']:
        for i in range(len(vec_res['ids'][0])):
            vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))
    # 2. Keyword Search (BM25)
    tokenized_query = normalize_text(query)
    doc_scores = bm25_index.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:10]
    
    kw_list = []
    for idx in top_n:
        if doc_scores[idx] > 0:
            kw_list.append((corpus_ids[idx], corpus_docs[idx], corpus_metadatas[idx]))

    # 3. Fusion
    best_weights = {"keyword": 0.5, "vector": 0.5}
    final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list}, weights=best_weights)
    if not final_results:
        return {"answer": "Désolé, je n'ai trouvé aucun document pertinent dans la base de données.", "sources": []}
     # 4. Prompt Preparation
    context_pieces = []
    formatted_sources = []
    
    for rank, (doc_id, data) in enumerate(final_results[:3]):
        meta = data['meta']
        text = data['text']
        source_title = meta.get('parent_title', meta.get('title', 'Document sans titre'))
        
        context_pieces.append(f"DOCUMENT {rank+1} (Source: {source_title})\nCONTENU: {text}")
        
        formatted_sources.append({
            "doc_id": str(doc_id),
            "score": data['score'],
            "text": text,
            "title": source_title
        })

    full_context = "\n\n---\n\n".join(context_pieces)
    prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.
 RÈGLES STRICTES DE RÉDACTION :
1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement, puis justifie en citant la base légale.
3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant.
4. PRÉCISION CHIRURGICALE : Reproduis fidèlement les noms propres, montants, et utilise des listes si nécessaire.
5. NETTOYAGE VISUEL : Ignore les pointillés ("....") dans le texte brut.

CONTEXTE FOURNI :
{full_context}

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE :
"""
    client = AsyncClient(host=settings.OLLAMA_HOST)
    response = await client.chat(model=settings.LLM_MODEL, messages=[{'role': 'user', 'content': prompt}])
    
    return {
        "answer": response['message']['content'],
        "sources": formatted_sources
    } 
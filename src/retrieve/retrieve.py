import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 🏆 MODULE DE RETRIEVAL OPTIMISÉ (PRÊT POUR RERANKER)
# ==============================================================================
# Ce fichier est conçu pour être importé comme un module.
# Il exécute la logique de routage séquentiel (Regex -> Fallback) 
# et retourne les Top-K documents qui seront ensuite traités par le Cross-Encoder.
# ==============================================================================

# --- CONFIGURATION MATÉRIELLE ---
# Masque le GPU 0, utilise les GPU 1, 2 ou 3 pour l'embedding
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

# --- CONFIGURATION BASE DE DONNÉES ---
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- ⚙️ PARAMÈTRES OPTIMISÉS ---
FALLBACK_DISTANCE_THRESHOLD = 1.05 


def retrieve_vector_chunks(q_embed, chunk_type, collection, top_k):
    """Effectue une recherche purement vectorielle filtrée par métadonnée."""
    
    vec_res = collection.query(
        query_embeddings=q_embed, 
        n_results=top_k, # 🔥 Le paramètre modifiable est appliqué ici
        where={"chunking_method": chunk_type} # L'astuce du routage est ici
    )
    
    formatted_results = []
    top_distance = 999.0 
    
    if vec_res['ids'] and len(vec_res['ids'][0]) > 0:
        top_distance = vec_res['distances'][0][0] 
        for i in range(len(vec_res['ids'][0])):
            formatted_results.append({
                "id": vec_res['ids'][0][i],
                "text": vec_res['documents'][0][i],
                "meta": vec_res['metadatas'][0][i],
                "distance": vec_res['distances'][0][i]
            })
            
    return formatted_results, top_distance


def get_retrieved_documents(query, model, collection, top_k=20, threshold=FALLBACK_DISTANCE_THRESHOLD):
    """
    Fonction principale à importer dans ton script de Reranking.
    
    Args:
        query (str): La question de l'utilisateur.
        model (SentenceTransformer): Le modèle BGE-M3 déjà chargé.
        collection (chromadb.Collection): La collection ChromaDB déjà connectée.
        top_k (int): Le nombre de documents à récupérer (ex: 20 pour le Reranker).
        threshold (float): Le seuil de distance pour déclencher le fallback.
        
    Returns:
        list: La liste des documents formatés prêts à être rerankés.
        str: La stratégie utilisée ("regex" ou "recursive").
    """
    # 1. Encodage de la question
    q_embed = model.encode([query]).tolist()
    
    # 2. PLAN A : Recherche dans les chunks REGEX
    final_results, top_dist = retrieve_vector_chunks(q_embed, "regex", collection, top_k)
    
    # 3. ROUTAGE SÉQUENTIEL (Le Fallback)
    if top_dist > threshold or not final_results:
        # ⚠️ Distance trop élevée, basculement sur le filet de sécurité (Chunks Récursifs)
        final_results, _ = retrieve_vector_chunks(q_embed, "recursive", collection, top_k)
        strategy_used = "recursive"
    else:
        # 🎯 Document structuré valide trouvé
        strategy_used = "regex"
        
    return final_results, strategy_used
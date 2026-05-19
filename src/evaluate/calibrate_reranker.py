import os
import sys
import json
import math
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ==============================================================================
# 🔐 CONFIGURATION DES CHEMINS & IMPORTS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir) 

# Import de vos fonctions existantes
try:
    from rerank.rerank import get_best_documents_for_llm
except ImportError:
    from rerank import get_best_documents_for_llm

try:
    from generate.query_parse import rewrite_query
except ImportError:
    from query_parse import rewrite_query

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db" # Ajustez si besoin
COLLECTION_NAME = "legal_algeria"
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DATASET_PATH = "../../data/golden_dataset/golden_dataset.json" # Ajustez si besoin

def pure_sigmoid(score):
    """Calcule le pourcentage mathématique pur"""
    clamped = max(-10.0, min(10.0, float(score)))
    return int((1 / (1 + math.exp(-clamped))) * 100)

def main():
    print("⚙️ Chargement des modèles pour l'analyse des scores...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL, max_length=1024)

    # Chargement d'un petit échantillon du dataset (5 questions suffisent)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)[:5] 

    print("\n" + "="*80)
    print("📊 ANALYSE DE LA DISTRIBUTION DES SCORES DU RERANKER")
    print("="*80)

    for idx, item in enumerate(dataset):
        original_query = item["question"]
        print(f"\n❓ Q{idx+1} : {original_query}")
        
        optimized_query = rewrite_query(original_query)
        
        # On demande un Top 10 pour voir la dégradation des scores entre le 1er et le 10ème
        docs = get_best_documents_for_llm(
            optimized_query, 
            collection, 
            bi_encoder, 
            reranker, 
            top_k_retrieve=15, 
            top_k_rerank=10 
        )

        print(f"{'Rang':<6} | {'Score Brut':<12} | {'Sigmoïde %':<12} | {'Source / Titre':<40}")
        print("-" * 80)
        
        for i, doc in enumerate(docs):
            raw_score = doc.get("rerank_score", 0.0)
            sig_pct = pure_sigmoid(raw_score)
            
            # Récupération des infos pour voir de quel document on parle
            meta = doc.get("meta", {})
            titre = meta.get("parent_title", meta.get("source_file", "Inconnu"))
            titre_court = (titre[:37] + '...') if len(titre) > 37 else titre
            
            # Formatage de l'affichage
            print(f"#{i+1:<4} | {raw_score:>10.4f} | {sig_pct:>8}%   | {titre_court}")
            
            # Si vous voulez lire un bout du texte pour confirmer s'il est pertinent ou non, décommentez la ligne ci-dessous :
            # print(f"      Extrait : {doc['text'][:100]}...\n")

if __name__ == "__main__":
    main()
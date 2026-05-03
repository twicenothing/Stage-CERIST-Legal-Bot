import os
import json
import string
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
load_dotenv()
# --- CONFIGURATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
GOLDEN_DATASET_PATH = "../../data/golden_dataset/golden_dataset.json" # Remplace par le chemin de ton fichier

# --- GRILLES DE PARAMÈTRES À TESTER ---
# Le script va tester TOUTES les combinaisons possibles de ces listes
FALLBACK_THRESHOLDS = [0.85, 0.90, 0.95, 1.0, 1.05]
WEIGHT_COMBINATIONS = [
    {"keyword": 0.8, "vector": 0.2}, # Fort sur le mot-clé exact
    {"keyword": 0.6, "vector": 0.4}, # Hybride équilibré (orienté lexique)
    {"keyword": 0.5, "vector": 0.5}, # Égalité parfaite
    {"keyword": 0.3, "vector": 0.7}, # Fort sur le sémantique (sens global)
    {"keyword": 0.0, "vector": 1.0}  # 100% Vectoriel (pour comparer l'impact du BM25)
]

def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def reciprocal_rank_fusion(results_dict, weights, k=60):
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        weight = weights.get(system_name, 1.0) 
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "text": doc_text, "meta": metadata}
            fused_scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
    return sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)

def retrieve_chunks(query, q_embed, chunk_type, collection, bm25_index, data_dict, weights):
    # Vector Search
    vec_res = collection.query(
        query_embeddings=q_embed, 
        n_results=10,
        where={"chunking_method": chunk_type}
    )
    vec_list = []
    top_distance = 999.0 
    if vec_res['ids'] and len(vec_res['ids'][0]) > 0:
        top_distance = vec_res['distances'][0][0] 
        for i in range(len(vec_res['ids'][0])):
            vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

    # Keyword Search (BM25)
    kw_list = []
    if bm25_index:
        tokenized_query = normalize_text(query)
        doc_scores = bm25_index.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10] 
        for idx in top_n:
            if doc_scores[idx] > 0:
                kw_list.append((data_dict['ids'][idx], data_dict['docs'][idx], data_dict['metas'][idx]))

    # Fusion
    final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list}, weights=weights)
    return final_results, top_distance

def evaluate_configuration(dataset, collection, model, bm25_regex, bm25_rec, regex_data, rec_data, threshold, weights):
    """Teste une combinaison de paramètres sur tout le dataset et renvoie le Hit Rate."""
    hits = 0
    total_queries = len(dataset)
    
    for item in dataset:
        query = item["question"]
        expected_source = item["source"]
        
        q_embed = model.encode([query]).tolist()
        
        # Plan A: Regex
        results, top_dist = retrieve_chunks(query, q_embed, "regex", collection, bm25_regex, regex_data, weights)
        
        # Plan B: Fallback
        if top_dist > threshold or not results:
            results, _ = retrieve_chunks(query, q_embed, "recursive", collection, bm25_rec, rec_data, weights)
            
        # Évaluation : Est-ce que le fichier source attendu est dans le TOP 3 ?
        # Évaluation : Est-ce que le fichier source attendu est dans le TOP 3 ?
        success = False
        
        # 1. On nettoie la source attendue (on enlève l'extension)
        expected_base = expected_source.split('.')[0] 
        
        for _, data in results[:3]:
            # 2. On récupère la source dans la BDD et on la nettoie aussi
            db_source = data['meta'].get('source_file', '')
            db_base = db_source.split('.')[0]
            
            if expected_base == db_base:
                success = True
                break
                
        if success:
            hits += 1
        # --- OPTIONNEL : MODE DÉBOGAGE ---
        # Décommente ces 3 lignes si tu as toujours 0.0% pour voir ce que le script compare réellement
        # else:
        #     db_sources_found = [d[1]['meta'].get('source_file') for d in results[:3]]
        #     print(f"   [Erreur] Attendu: {expected_source} | Trouvé dans le Top 3: {db_sources_found}")
            
    hit_rate = (hits / total_queries) * 100
    return hit_rate

def main():
    print("🔄 Initialisation du système d'évaluation...")
    
    # 1. Chargement du Golden Dataset
    try:
        with open(GOLDEN_DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"📖 Golden Dataset chargé : {len(dataset)} questions à tester.")
    except Exception as e:
        print(f"❌ Erreur de chargement du dataset : {e}")
        return

    # 2. Setup Base de données et Modèle
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    all_docs = collection.get()

    # 3. Préparation BM25
    regex_data = {'docs': [], 'ids': [], 'metas': []}
    rec_data = {'docs': [], 'ids': [], 'metas': []}
    for doc, doc_id, meta in zip(all_docs['documents'], all_docs['ids'], all_docs['metadatas']):
        if meta.get('chunking_method') == 'regex':
            regex_data['docs'].append(doc); regex_data['ids'].append(doc_id); regex_data['metas'].append(meta)
        else:
            rec_data['docs'].append(doc); rec_data['ids'].append(doc_id); rec_data['metas'].append(meta)

    bm25_regex = BM25Okapi([normalize_text(d) for d in regex_data['docs']]) if regex_data['docs'] else None
    bm25_rec = BM25Okapi([normalize_text(d) for d in rec_data['docs']]) if rec_data['docs'] else None
    
    print("\n🚀 Démarrage du Grid Search (Recherche des meilleurs paramètres)...")
    print("="*60)
    
    best_score = -1
    best_config = {}
    results_log = []

    # 4. Boucle de Grid Search
    total_tests = len(FALLBACK_THRESHOLDS) * len(WEIGHT_COMBINATIONS)
    current_test = 1
    
    for threshold in FALLBACK_THRESHOLDS:
        for weights in WEIGHT_COMBINATIONS:
            print(f"[{current_test}/{total_tests}] Test -> Seuil: {threshold:.2f} | Poids: {weights}")
            
            score = evaluate_configuration(
                dataset, collection, model, bm25_regex, bm25_rec, 
                regex_data, rec_data, threshold, weights
            )
            
            print(f"   🎯 Hit Rate@3 : {score:.1f}%")
            
            results_log.append({
                "threshold": threshold,
                "weights": weights,
                "hit_rate": score
            })
            
            if score > best_score:
                best_score = score
                best_config = {"threshold": threshold, "weights": weights}
                
            current_test += 1

    # 5. Affichage du Résultat Final
    print("\n" + "="*60)
    print("🏆 RÉSULTAT OPTIMAL TROUVÉ 🏆")
    print("="*60)
    print(f"Meilleur Hit Rate : {best_score:.1f}%")
    print(f"Seuil de Fallback idéal : {best_config['threshold']}")
    print(f"Poids RRF idéaux        : {best_config['weights']}")
    print("="*60)
    
    # (Optionnel) Sauvegarder les logs pour analyse graphique
    with open("grid_search_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4)
    print("💾 Résultats complets sauvegardés dans 'grid_search_results.json'")

if __name__ == "__main__":
    main()
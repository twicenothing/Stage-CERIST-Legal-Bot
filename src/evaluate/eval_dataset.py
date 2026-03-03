import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# --- LE GOLDEN DATASET ---
golden_dataset = [
    {"query": "À quelle date a été signé le décret exécutif n° 25-74 fixant les attributions du ministre de la jeunesse ?", "expected_doc_id": "F2025010.json"},
    {"query": "Qui est le signataire du décret exécutif n° 25-74 ?", "expected_doc_id": "F2025010.json"},
    {"query": "D'après l'article 10 du décret exécutif n° 25-74, quel ancien décret est explicitement abrogé ?", "expected_doc_id": "F2025010.json"},
    {"query": "Quelles sont les missions principales de la sous-direction de la protection et de l'insertion des jeunes mentionnées dans l'Article 3 du décret n° 25-75 ?", "expected_doc_id": "F2025010.json"},
    {"query": "Quelles sont les trois sous-directions qui composent la direction du développement des compétences, des initiatives des jeunes et de l’insertion ?", "expected_doc_id": "F2025010.json"},
    {"query": "Quel est le montant exact en dinars annulé sur les crédits ouverts selon l'Article 1er du décret présidentiel n° 24-441 ?", "expected_doc_id": "F2025009.json"},
    {"query": "À la disposition de quel ministre le transfert de crédits du 31 décembre 2024 a-t-il été mis ?", "expected_doc_id": "F2025009.json"},
    {"query": "Qui est la représentante de l'association nationale des affaires de la femme divorcée, de la veuve et de l'enfance ?", "expected_doc_id": "F2025009.json"},
    {"query": "De quel laboratoire de recherche Monsieur Abdelhalim Bouchekioua est-il le représentant ?", "expected_doc_id": "F2025009.json"},
    {"query": "Quelle est la durée de validité minimale exigée pour les passeports diplomatiques ou de service avant d'entrer sur le territoire de l'autre partie, selon l'accord entre l'Algérie et l'Indonésie ?", "expected_doc_id": "F2025008.json"},
    {"query": "Pour quelle durée maximale les détenteurs de passeports diplomatiques et de service sont-ils exemptés de visa pour séjourner dans l'autre pays selon l'accord algéro-indonésien ?", "expected_doc_id": "F2025008.json"},
    {"query": "Quel est le délai maximum pour transmettre les documents officiels après une arrestation provisoire dans le cadre de la convention d'extradition entre l'Algérie et la Tunisie ?", "expected_doc_id": "F2025008.json"},
    {"query": "Quelle est la durée minimale de la peine privative de liberté requise pour qu'une infraction donne lieu à une extradition selon l'Article 2 de la convention algéro-tunisienne ?", "expected_doc_id": "F2025008.json"},
    {"query": "Quel est le montant exact en dinars ouvert en crédits de paiement au profit du portefeuille de programmes de la Présidence de la République pour l'année 2024 ?", "expected_doc_id": "F2025007.json"},
    {"query": "Quel ministre est représenté par Saïda Malek au sein du conseil d'administration du commissariat aux énergies renouvelables et à l'efficacité énergétique ?", "expected_doc_id": "F2025007.json"},
    {"query": "Quel ministère est chargé d'élaborer et d'exécuter les plans de confortement spécifiquement pour les ouvrages à valeur patrimoniale ?", "expected_doc_id": "F2025007.json"},
    {"query": "D'après les dispositions relatives aux plans de confortement, pour quelle durée les membres de la commission nationale sont-ils nommés ?", "expected_doc_id": "F2025007.json"},
    {"query": "Quelle autorité préside le comité intersectoriel chargé de l'évaluation des dégâts occasionnés par une catastrophe ?", "expected_doc_id": "F2025007.json"},
    {
        "query": "Selon l'Article 18 de l'accord ratifié par le décret présidentiel n° 25-57, pour quelle durée l'accord de coopération entre l'Algérie et l'Allemagne est-il initialement conclu ?",
        "expected_doc_id": "F2025006.json"
    },
    {
        "query": "Qui sont les représentants respectifs de l'Algérie et de l'Allemagne qui ont signé l'accord de coopération culturelle et scientifique le 13 juin 2022 ?",
        "expected_doc_id": "F2025006.json"
    },
    {
        "query": "Quel pays est concerné par le mémorandum d'entente de coopération ratifié par le décret présidentiel 25-58 du 23 janvier 2025 ?",
        "expected_doc_id": "F2025006.json"
    },
    {
        "query": "Comment les deux Etats prévoient-ils de traiter la question de l'équivalence et de la reconnaissance des grades ou diplômes universitaires délivrés par l'autre pays ?",
        "expected_doc_id": "F2025006.json"
    },
    {
        "query": "Quel est le nombre total d'effectifs prévus pour les agents exerçant des activités d’entretien et de maintenance au sein de la direction générale du budget ?",
        "expected_doc_id": "F2025005.json"
    },
    {
        "query": "Quelles sont les conditions de quorum requises pour que la commission sectorielle de tutelle pédagogique puisse valablement délibérer lors de sa première convocation ?",
        "expected_doc_id": "F2025005.json"
    },
    {
        "query": "Comment la Cour justifie-t-elle que l'obligation d'être représenté par un avocat en cassation ne viole pas le droit d'accès à la justice pour les citoyens démunis ?",
        "expected_doc_id": "F2025005.json"
    },

]

# --- FONCTIONS UTILITAIRES ---
def normalize_text(text):
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

# --- FONCTION D'ÉVALUATION ---
# J'ai ajouté les paramètres model, collection, etc. pour éviter les erreurs de variables globales
def evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 1.0, "vector": 1.0}, top_k=3):
    print(f"\n📊 RUNNING EVALUATION (Weights: {weights})")
    print("-" * 50)
    
    total_queries = len(golden_dataset)
    hits = 0
    mrr_sum = 0.0
    
    for item in golden_dataset:
        query = item["query"]
        expected_id = item["expected_doc_id"]
        
        # 1. Vector Search
        q_embed = model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        vec_list = [(vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]) for i in range(len(vec_res['ids'][0]))]
        
        # 2. Keyword Search
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10]
        kw_list = [(ids[idx], documents[idx], metadatas[idx]) for idx in top_n if doc_scores[idx] > 0]
        
        # 3. Fuse Results
        final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list}, weights=weights)
        
        # 4. Check Rankings
        found_rank = None
        for rank, (doc_id, data) in enumerate(final_results[:top_k]):
            if expected_id in doc_id: # On vérifie si expected_id est contenu dans le vrai doc_id
                found_rank = rank + 1
                break
                
        # 5. Metrics
        if found_rank:
            hits += 1
            mrr_sum += (1.0 / found_rank)
            print(f"✅ [HIT at Rank {found_rank}] {query[:50]}...")
        else:
            print(f"❌ [MISS] {query[:50]}...")
            
    hit_rate = (hits / total_queries) * 100
    mrr = mrr_sum / total_queries
    
    print("\n" + "=" * 50)
    print(f"📈 RESULTS (Top {top_k})")
    print(f"Hit Rate: {hit_rate:.1f}% (Found in top {top_k})")
    print(f"MRR:      {mrr:.3f} (Closer to 1.0 is better)")
    print("=" * 50)
    
    return hit_rate, mrr

# --- SCRIPT PRINCIPAL ---
# --- SCRIPT PRINCIPAL ---
def main():
    print("🔄 Initialisation du système pour l'évaluation...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    print("🤖 Chargement du modèle d'embedding...")
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})

    print("📚 Préparation de l'index BM25...")
    all_docs = collection.get()
    documents = all_docs['documents']
    ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    tokenized_corpus = [normalize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print("✅ Système prêt. Lancement des tests de performance...\n")

    # TEST 1 : Baseline (50% Mots-clés / 50% Vecteur)
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 1.0, "vector": 1.0})

    # TEST 2 : Pure Keyword (100% Mots-clés / 0% Vecteur) - Pour voir si la sémantique sert vraiment à quelque chose
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 1.0, "vector": 0.0})

    # TEST 3 : Pure Vecteur (0% Mots-clés / 100% Vecteur) - Pour voir si BM25 est indispensable
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 0.0, "vector": 1.0})

    # TEST 4 : Forte priorité Mots-clés (80% / 20%)
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 0.8, "vector": 0.2})

    # TEST 5 : Priorité modérée Mots-clés (60% / 40%)
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 0.6, "vector": 0.4})

    # TEST 6 : Forte priorité Vecteur (20% / 80%)
    evaluate_system(golden_dataset, model, collection, bm25, ids, documents, metadatas, weights={"keyword": 0.2, "vector": 0.8})

if __name__ == "__main__":
    main()
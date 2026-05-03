import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# ==============================================================================
# 🏆 SCRIPT DE RETRIEVAL OPTIMISÉ (RÉSULTATS DU GRID SEARCH)
# ==============================================================================
#
# Suite à l'évaluation par Grid Search (Hit Rate: 81.0%), voici les conclusions :
# 1. Seuil de Fallback idéal : 1.05
# 2. Poids RRF idéaux        : {'keyword': 0.0, 'vector': 1.0}
#
# 🛑 POURQUOI LA RECHERCHE PAR MOTS-CLÉS (BM25) A-T-ELLE ÉTÉ SUPPRIMÉE ?
# Le langage juridique utilise énormément de synonymes ou de périphrases 
# (ex: "amende" vs "pénalité financière", ou "loi fondamentale" vs "constitution").
# Le système BM25 cherche des correspondances de mots exactes, ce qui génère des 
# échecs fréquents. Le modèle d'embedding (BGE-M3), en revanche, cartographie le 
# "sens" des phrases. Le Grid Search a prouvé qu'une approche 100% sémantique 
# était la plus performante.
# -> Avantage majeur : La suppression du BM25 élimine le besoin de charger 
#    l'intégralité des documents en mémoire vive au lancement du script.
#
# ⚙️ EXPLICATION DU FLUX DE TRAVAIL (ROUTAGE SÉQUENTIEL)
# 1. L'utilisateur pose une question.
# 2. BGE-M3 encode cette question en un vecteur mathématique.
# 3. PLAN A (Regex) : ChromaDB cherche les documents les plus proches sémantiquement,
#    en filtrant UNIQUEMENT sur les chunks structurés (Titre + Préambule + Article).
# 4. L'ÉVALUATION (Le Seuil à 1.05) : ChromaDB renvoie une "Distance L2". 
#    - Si Distance <= 1.05 : Le document est jugé très pertinent. On s'arrête là.
#    - Si Distance > 1.05 : Le modèle doute fortement de la pertinence de sa trouvaille.
# 5. PLAN B (Récursif) : Le filet de sécurité s'active. On refait une recherche, 
#    mais cette fois uniquement dans les chunks découpés à l'aveugle (recursive), 
#    au cas où la réponse aurait été ratée par le script d'extraction Regex.
# ==============================================================================


# --- CONFIGURATION MATÉRIELLE ---
# Masque le GPU 0, utilise les GPU 1, 2 ou 3 pour l'embedding
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

# --- CONFIGURATION BASE DE DONNÉES ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- ⚙️ PARAMÈTRES OPTIMISÉS ---
FALLBACK_DISTANCE_THRESHOLD = 1.05 


def retrieve_vector_chunks(q_embed, chunk_type, collection):
    """Effectue une recherche purement vectorielle filtrée par métadonnée."""
    
    vec_res = collection.query(
        query_embeddings=q_embed, 
        n_results=3, # On ne récupère que le Top 3 directement
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


def display_results(results, search_type):
    """Affiche les résultats formatés dans le terminal."""
    print(f"\n" + "="*80)
    print(f"📊 RÉSULTATS DU RETRIEVER | STRATÉGIE UTILISÉE : {search_type.upper()}")
    print("="*80)
    
    if not results:
        print("❌ Aucun document pertinent trouvé.")
        return

    for i, data in enumerate(results):
        score = data['distance']
        text = data['text']
        meta = data['meta']
        
        doc_type = meta.get('document_type', 'N/A')
        source_file = meta.get('source_file', 'Fichier inconnu')
        
        print(f"\n🔹 RANG [{i+1}] - DISTANCE L2 : {score:.4f}")
        print(f"   ID       : {data['id']}")
        print(f"   MÉTHODE  : {meta.get('chunking_method', 'inconnu').upper()}")
        print(f"   SOURCE   : {source_file} ({doc_type})")
        print("-" * 80)
        print(text[:300] + "..." if len(text) > 300 else text)
        print("-" * 80)


def main():
    print(f"🔄 Connexion à ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return
    
    print("🤖 Chargement du modèle d'embedding BGE-M3 (Device: CUDA)...")
    # Chargement direct sans fp16 ici car on ne traite qu'une phrase à la fois, 
    # la consommation VRAM est négligeable en inférence simple.
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print(f"✅ Système Prêt ! Seuil Fallback L2 = {FALLBACK_DISTANCE_THRESHOLD}")

    while True:
        query = input("\n❓ Entrez votre requête juridique (ou 'q' pour quitter) : ").strip()
        if query.lower() == 'q': 
            break
        
        # 1. Encodage de la question
        q_embed = model.encode([query]).tolist()
        
        # 2. PLAN A : Recherche dans les chunks REGEX
        print("\n🔍 Lancement de la recherche structurée (Regex)...")
        final_results, top_dist = retrieve_vector_chunks(q_embed, "regex", collection)
        
        print(f"   > Meilleure distance sémantique L2 (Regex) : {top_dist:.4f}")
        
        # 3. ROUTAGE SÉQUENTIEL (Le Fallback)
        if top_dist > FALLBACK_DISTANCE_THRESHOLD or not final_results:
            print(f"⚠️ Distance L2 ({top_dist:.4f}) > Seuil ({FALLBACK_DISTANCE_THRESHOLD}).")
            print("🔀 Basculement sur le filet de sécurité (Chunks Récursifs)...")
            
            final_results, top_dist_rec = retrieve_vector_chunks(q_embed, "recursive", collection)
            print(f"   > Meilleure distance sémantique L2 (Récursif) : {top_dist_rec:.4f}")
            
            display_results(final_results, search_type="RÉCURSIF (Fallback)")
            
        else:
            print(f"🎯 Document structuré valide trouvé.")
            display_results(final_results, search_type="REGEX (Principal)")

if __name__ == "__main__":
    main()
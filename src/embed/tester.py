import chromadb
from sentence_transformers import SentenceTransformer
import os

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_DIR = os.path.join(BASE_DIR, "../../data/chroma_db")

# --- INIT ---
print("🔄 Chargement de la base de données et du modèle...")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("legal_algeria")

# Toujours le fix de sécurité pour BGE-M3
model = SentenceTransformer("BAAI/bge-m3", model_kwargs={"use_safetensors": True})

def search_legal_interactive():
    while True:
        print("\n" + "="*50)
        user_query = input("❓ Posez votre question (ou 'q' pour quitter) : ")
        
        if user_query.lower() in ['q', 'quit', 'exit']:
            print("Au revoir !")
            break
            
        # 1. Extraction automatique de mots-clés (mots > 3 lettres) pour le debug
        debug_keywords = [w.lower() for w in user_query.split() if len(w) > 3]
        
        print(f"🔎 Recherche en cours pour : '{user_query}'")
        
        # 2. Embedding de la question avec l'instruction BGE-M3
        instruction = "Represent this sentence for searching relevant passages: "
        query_vec = model.encode([instruction + user_query], normalize_embeddings=True).tolist()

        # 3. Requête Chroma (Top 15 + Filtre Décret)
        results = collection.query(
            query_embeddings=query_vec,
            n_results=15, 
            where={"type": "Decret"} 
        )

        # 4. Affichage des résultats
        count = len(results['ids'][0])
        print(f"   > {count} décrets trouvés.")
        
        if count == 0:
            print("❌ Aucun résultat.")
            continue

        for i in range(count):
            meta = results['metadatas'][0][i]
            text = results['documents'][0][i]
            score = results['distances'][0][i] # Distance (plus petit = plus proche)

            # Vérification visuelle : les mots de la question sont-ils dans le texte ?
            found_kw = [k for k in debug_keywords if k in text.lower()]
            # On crée un petit badge visuel
            if len(found_kw) > 0:
                badge = f"✅ Contient : {', '.join(found_kw)}"
            else:
                badge = "⚠️ Aucun mot-clé direct trouvé (Attention !)"

            title = meta.get('official_id', 'Sans Titre')
            date = meta.get('journal_date', '?')

            print(f"\n🔹 RANG {i+1} | Score: {score:.4f}")
            print(f"   {badge}")
            print(f"   📜 {title} | {date} | Source: {meta['source']}")
            # Affiche un extrait un peu plus long (300 caractères) et nettoie les retours à la ligne
            snippet = text[0:300].replace('\n', ' ')
            print(f"   📝 \"{snippet}...\"")

if __name__ == "__main__":
    search_legal_interactive()
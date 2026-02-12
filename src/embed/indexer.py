import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
JSON_DIR = "../../data/json_llm_extracted"  # Là où safe_chunker a mis les fichiers
CHROMA_PATH = "../../data/chroma_db"        # Le dossier qu'on va recréer
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

def main():
    # 1. Initialiser ChromaDB
    print(f"🔄 Initialisation de ChromaDB dans {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # On supprime la collection si elle existe pour repartir de zéro (double sécurité)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Ancienne collection supprimée.")
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    # 2. Charger le modèle d'embedding
    print(f"🤖 Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda", model_kwargs={"use_safetensors": True})

    # 3. Lister les fichiers JSON
    if not os.path.exists(JSON_DIR):
        print(f"❌ Erreur : Le dossier {JSON_DIR} n'existe pas. Lance safe_chunker.py d'abord.")
        return

    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"📦 {len(files)} fichiers trouvés à indexer.")

    # 4. Boucle d'indexation
    total_docs = 0
    
    for filename in files:
        file_path = os.path.join(JSON_DIR, filename)
        print(f"   📄 Traitement de {filename}...", end="")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # On vérifie que la structure est bonne (celle de safe_chunker)
            if "documents" not in data:
                print(" ⚠️ Pas de clé 'documents', fichier ignoré.")
                continue

            documents = []
            ids = []
            metadatas = []
            embeddings = []

            # Extraction des données
            texts_to_embed = []
            
            for item in data["documents"]:
                doc_content = item.get("content", "")
                
                if not doc_content.strip():
                    continue

                # On prépare les listes pour Chroma
                documents.append(doc_content)
                ids.append(item.get("id", f"{filename}_{total_docs}")) # Fallback ID
                metadatas.append({
                    "source": filename,
                    "title": item.get("title", filename)
                })
                texts_to_embed.append(doc_content)
                total_docs += 1

            # Calcul des embeddings (Vectorisation)
            if texts_to_embed:
                embeddings = model.encode(texts_to_embed).tolist()
                
                # Ajout dans la base
                collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f" ✅ {len(documents)} chunks indexés.")
            else:
                print(" ⚠️ Aucun texte valide trouvé.")

        except Exception as e:
            print(f" ❌ Erreur : {e}")

    print("\n" + "="*50)
    print(f"🎉 INDEXATION TERMINÉE ! Total : {total_docs} fragments de texte.")
    print("="*50)

if __name__ == "__main__":
    main()
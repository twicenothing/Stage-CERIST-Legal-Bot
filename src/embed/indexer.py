import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
JSON_DIR = "../../data/json"
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

def main():
    # 1. Initialize ChromaDB
    print(f"🔄 Initializing ChromaDB at: {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # Reset collection (Delete if exists)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Old collection deleted (Starting fresh).")
    except:
        pass
    
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. Load Embedding Model
    print(f"🤖 Loading Model: {MODEL_NAME} for Multi-GPU...")
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})
    
    # Start the multi-process pool
    gpu_pool = model.start_multi_process_pool()

    # 3. List JSON Files
    if not os.path.exists(JSON_DIR):
        print(f"❌ Error: JSON directory not found: {JSON_DIR}")
        return

    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"📦 Found {len(files)} JSON files to index.")

    total_chunks = 0
    
    # 4. Processing Loop
    for filename in files:
        file_path = os.path.join(JSON_DIR, filename)
        print(f"   📄 Processing {filename}...", end=" ")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "documents" not in data:
                print("⚠️ Skipped (No 'documents' key found).")
                continue

            ids = []
            documents = []  
            metadatas = []

            # Iterate through Decrees (Parents)
            for doc_idx, doc in enumerate(data["documents"]):
                parent_title = doc.get("title", "Sans titre")
                # 🔥 CHANGEMENT : On récupère la nouvelle clé "context"
                context_text = doc.get("context", "") 
                
                # --- A. INDEX THE PARENT (The Decree Itself) ---
                parent_id = f"{filename}_doc_{doc_idx}"
                # 🔥 CHANGEMENT : Plus besoin de summarize_text, on indexe le titre + le préambule propre
                parent_text_for_embedding = f"Titre: {parent_title}\nPréambule: {context_text}"
                
                ids.append(parent_id)
                documents.append(parent_text_for_embedding)
                metadatas.append({
                    "source": filename,
                    "type": "parent",
                    "title": parent_title,
                    "context": context_text  # 🔥 CHANGEMENT dans la métadonnée
                })
                total_chunks += 1

                # --- B. INDEX THE CHILDREN (The Articles) ---
                articles = doc.get("articles", [])

                for art_idx, article_text in enumerate(articles):
                    child_id = f"{parent_id}_art_{art_idx}"
                    
                    # 🔥 ON GARDE CECI : Titre + Article (C'est le mix parfait de contexte)
                    contextualized_text = f"Source: {parent_title}\nContenu: {article_text}"
                    
                    ids.append(child_id)
                    documents.append(contextualized_text) 
                    
                    metadatas.append({
                        "source": filename,
                        "type": "child",
                        "parent_title": parent_title,
                        "parent_id": parent_id,
                        "original_article_text": article_text 
                    })
                    total_chunks += 1

            # --- C. BATCH EMBEDDING & ADDING (MULTI-GPU) ---
            if documents:
                embeddings_array = model.encode_multi_process(documents, gpu_pool)
                embeddings = embeddings_array.tolist()
                
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                print(f"✅ Indexed {len(documents)} items.")
            else:
                print("⚠️ No valid text found.")

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n" + "="*60)
    print(f"🎉 INDEXING COMPLETE! Total Vectors: {total_chunks}")
    print("="*60)
    
    # Close the pool at the end
    model.stop_multi_process_pool(gpu_pool)

if __name__ == "__main__":
    main()
import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
# We go up two levels: embed -> src -> Stage-CERIST-Legal-Bot -> data
JSON_DIR = "../../data/json"
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

def summarize_text(text, max_length=1000):
    """
    Simple truncation for embedding parents (titles/preambles).
    Keeps the context manageable for the vector model.
    """
    return text[:max_length] + "..." if len(text) > max_length else text

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
    # Removed device="cpu" so it defaults to CUDA and can span across GPUs
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})
    
    # 🔥 NEW: Start the multi-process pool to use all available GPUs
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

            # Check for the new structure key
            if "documents" not in data:
                print("⚠️ Skipped (No 'documents' key found).")
                continue

            # Lists to hold batch data
            ids = []
            documents = []  # The text content to embed
            metadatas = []

            # Iterate through Decrees (Parents)
            for doc_idx, doc in enumerate(data["documents"]):
                parent_title = doc.get("title", "Sans titre")
                full_context = doc.get("full_context", "")
                
                # --- A. INDEX THE PARENT (The Decree Itself) ---
                parent_id = f"{filename}_doc_{doc_idx}"
                parent_text_for_embedding = f"{parent_title}\n{summarize_text(full_context)}"
                
                ids.append(parent_id)
                documents.append(parent_text_for_embedding)
                metadatas.append({
                    "source": filename,
                    "type": "parent",
                    "title": parent_title,
                    "full_text": full_context  # Store full text for retrieval display
                })
                total_chunks += 1

                # --- B. INDEX THE CHILDREN (The Articles) ---
                articles = doc.get("articles", [])

                for art_idx, article_text in enumerate(articles):
                    child_id = f"{parent_id}_art_{art_idx}"
                    
                    # Prepend the parent title to the article text
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
                # 🔥 NEW: Encode using all GPUs in parallel
                embeddings_array = model.encode_multi_process(documents, gpu_pool)
                embeddings = embeddings_array.tolist()
                
                # Add to Chroma
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
    
    # 🔥 NEW: Always close the pool at the end to free up GPU memory
    model.stop_multi_process_pool(gpu_pool)

if __name__ == "__main__":
    main()
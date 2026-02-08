import os
import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data"))
JSON_WHOLE_DIR = os.path.join(DATA_DIR, "json_whole")
JSON_ARRETES_DIR = os.path.join(DATA_DIR, "json_arretes")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

# Model Name (HuggingFace)
MODEL_NAME = "BAAI/bge-m3"

# --- 1. SETUP EMBEDDING MODEL ---
print(f"🔄 Loading Model: {MODEL_NAME}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   > Running on: {device.upper()}")

# On force l'utilisation de "safetensors" pour éviter l'erreur de sécurité PyTorch
model = SentenceTransformer(
    MODEL_NAME, 
    device=device,
    model_kwargs={"use_safetensors": True}
)
# BGE-M3 supports up to 8192 tokens. We set a safe limit.
model.max_seq_length = 8192 

# --- 2. SETUP CHROMADB ---
print(f"🔄 Initializing ChromaDB at: {CHROMA_DB_DIR}")
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

# Create or get the collection
# We use Cosine Similarity (metadata={"hnsw:space": "cosine"})
collection = client.get_or_create_collection(
    name="legal_algeria",
    metadata={"hnsw:space": "cosine"}
)

def process_folder(folder_path, doc_type_tag):
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return

    print(f"📂 Scanning folder: {folder_path}")
    
    files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        print(f"   > Processing: {filename}")
        
        # Buffers for batch processing (faster than one by one)
        ids = []
        documents = []
        metadatas = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # Create a unique ID for the database
                    # format: filename_lineindex (e.g., F2025001_0)
                    doc_id = f"{filename}_{line_num}"
                    
                    # Prepare Metadata
                    meta = {
                        "source": data.get("source", filename),
                        "journal_date": data.get("journal_date", "Unknown"),
                        "page": data.get("page_start", 1),
                        "type": doc_type_tag, # "Decree" or "Arrete"
                        "official_id": data.get("decree_id") or data.get("title_extract", "Unknown")
                    }
                    
                    # Prepare Text
                    text_content = data.get("text", "")
                    
                    if len(text_content) > 50:
                        ids.append(doc_id)
                        documents.append(text_content)
                        metadatas.append(meta)
                        
                except json.JSONDecodeError:
                    continue

        # --- BATCH EMBEDDING & UPSERT ---
        if documents:
            # Generate Embeddings
            # BGE-M3 instructions: Pass raw text.
            embeddings = model.encode(documents, normalize_embeddings=True)
            
            # Add to Chroma
            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(), # Convert numpy array to list
                metadatas=metadatas,
                documents=documents
            )
            print(f"     ✅ Indexed {len(documents)} documents.")

def main():
    # Process Decrees
    process_folder(JSON_WHOLE_DIR, "Decret")
    
    # Process Arretes
    process_folder(JSON_ARRETES_DIR, "Arrete")
    
    print("\n🎉 Indexing Complete! Database saved in 'data/chroma_db'")
    print(f"Total documents in collection: {collection.count()}")

if __name__ == "__main__":
    main()
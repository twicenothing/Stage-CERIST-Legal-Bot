import os
import json
import chromadb
from sentence_transformers import SentenceTransformer
from collections import Counter
import torch

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JSON_DIR = os.path.join(BASE_DIR, "data", "json")
CHROMA_PATH = os.path.join(BASE_DIR, "data", "chroma_db")
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE_PER_GPU = 32 

def summarize_text(text, max_length=1500):
    return text[:max_length] + "..." if len(text) > max_length else text

def uniquify_ids(ids):
    count = Counter(ids)
    unique_ids = []
    seen = {}
    for id_val in ids:
        if count[id_val] > 1:
            if id_val not in seen: seen[id_val] = 0
            seen[id_val] += 1
            unique_ids.append(f"{id_val}_{seen[id_val]}")
        else:
            unique_ids.append(id_val)
    return unique_ids

def main():
    print(f"🔄 Initialisation de ChromaDB dans {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Ancienne collection supprimée.")
    except:
        pass
    
    collection = client.create_collection(name=COLLECTION_NAME)

    # --- 🚀 GPU SETUP ---
    target_devices = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"⚡ {gpu_count} GPUs détectés.")
        target_devices = list(range(gpu_count))

    print(f"🤖 Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda" if target_devices else "cpu", model_kwargs={"use_safetensors": True})

    if target_devices:
        pool = model.start_multi_process_pool(target_devices=target_devices)
    
    if not os.path.exists(JSON_DIR):
        print(f"❌ Erreur : Dossier {JSON_DIR} introuvable.")
        return

    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    print(f"📦 {len(files)} fichiers à indexer.")

    total_docs = 0
    
    for filename in files:
        file_path = os.path.join(JSON_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "hierarchical_documents" not in data:
                continue

            documents = []
            ids = []
            metadatas = []
            texts_to_embed = []
            
            for hierarchy in data["hierarchical_documents"]:
                parent = hierarchy.get("parent", {})
                children = hierarchy.get("children", [])
                
                if not parent: continue

                # --- TRAITEMENT DU PARENT ---
                parent_id = parent.get("id", f"{filename}_parent_{len(ids)}")
                parent_text = parent.get("text", "")
                
                # Extraction du titre pour le contexte
                parent_title = parent.get("metadata", {}).get("title", "")
                if not parent_title:
                    parent_title = parent_text[:250].replace("\n", " ")

                parent_summary = summarize_text(parent_text)
                parent_metadata = parent.get("metadata", {})
                parent_metadata.update({"source": filename, "type": "parent", "full_text": parent_text})
                
                documents.append(parent_summary)
                ids.append(parent_id)
                metadatas.append(parent_metadata)
                texts_to_embed.append(parent_summary)
                total_docs += 1

                # --- TRAITEMENT DES ENFANTS (ARTICLES) ---
                for child in children:
                    child_id = child.get("id", f"{parent_id}_child_{len(ids)}")
                    child_text = child.get("text", "")
                    
                    if not child_text.strip(): continue

                    child_metadata = child.get("metadata", {})
                    child_metadata.update({
                        "source": filename, 
                        "type": "child", 
                        "parent_id": parent_id,
                        "full_context": f"{parent_title}\n---\n{child_text}"
                    })
                    
                    # 💡 MODIFICATION CRITIQUE ICI 💡
                    # On crée un texte qui contient explicitement le titre ET l'article
                    contextualized_text = f"Source: {parent_title}\n---\nContenu Article: {child_text}"

                    # AVANT : documents.append(child_text)  <-- C'était l'erreur (Mistral ne voyait pas le titre)
                    # APRÈS : On donne le texte enrichi à Mistral
                    documents.append(contextualized_text) 
                    
                    ids.append(child_id)
                    metadatas.append(child_metadata)
                    texts_to_embed.append(contextualized_text) # On embed aussi le texte enrichi
                    total_docs += 1

            ids = uniquify_ids(ids)

            if texts_to_embed:
                if target_devices:
                    embeddings = model.encode_multi_process(texts_to_embed, pool, batch_size=BATCH_SIZE_PER_GPU)
                    if hasattr(embeddings, "tolist"): embeddings = embeddings.tolist()
                else:
                    embeddings = model.encode(texts_to_embed, batch_size=BATCH_SIZE_PER_GPU).tolist()
                
                collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
                print(f" ✅ {filename} : {len(documents)} docs indexés.")

        except Exception as e:
            print(f" ❌ Erreur sur {filename}: {e}")

    if target_devices:
        model.stop_multi_process_pool(pool)

    print(f"\n🎉 Indexation terminée ! ({total_docs} fragments)")

if __name__ == "__main__":
    main()
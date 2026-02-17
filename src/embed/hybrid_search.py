import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import ollama  # Requires: pip install ollama

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# --- 1. GENERATION WITH CONTEXT AWARENESS ---
def generate_answer(query, ranked_results):
    """
    Constructs a prompt using Parent/Child metadata to give Mistral full context.
    """
    if not ranked_results:
        return "Désolé, je n'ai trouvé aucun document pertinent."

    # Build Context
    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:5]): # Top 5 results
        meta = data['meta']
        text = data['text']
        
        # --- NEW LOGIC: Use Metadata to label sources correctly ---
        if meta.get('type') == 'child':
            # It's an Article -> Show which Decree it belongs to
            parent_title = meta.get('parent_title', 'Décret Inconnu')
            formatted_text = f"SOURCE : {parent_title}\nEXTRAIT : {text}"
        else:
            # It's a Parent (Summary) -> Show Title + Summary
            title = meta.get('title', 'Document')
            formatted_text = f"SOURCE : {title}\nTYPE : Résumé/Préambule\nCONTENU : {text}"

        context_pieces.append(f"[Document {rank+1}]\n{formatted_text}")

    full_context = "\n\n---\n\n".join(context_pieces)

    # Prompt
    prompt = f"""
    Tu es un assistant juridique expert en droit algérien.
    Réponds à la question en utilisant EXCLUSIVEMENT le contexte ci-dessous.
    
    RÈGLES :
    - Cite toujours le décret ou la loi (ex: "Selon le décret exécutif 25-54...").
    - Si l'information n'est pas dans le contexte, dis-le clairement.
    - Réponds en français de manière professionnelle.

    CONTEXTE :
    {full_context}

    QUESTION :
    {query}
    """

    print("🤖 Mistral rédige...", end="", flush=True)
    
    try:
        response = ollama.chat(
            model='mistral', 
            messages=[{'role': 'user', 'content': prompt}]
        )
        print(" ✅")
        return response['message']['content']
        
    except Exception as e:
        return f"\n❌ Erreur connexion Ollama : {e}. Vérifie que 'ollama serve' tourne."

# --- 2. UTILS ---
def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def reciprocal_rank_fusion(results_dict, k=60):
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0, "text": doc_text, "meta": metadata}
            fused_scores[doc_id]["score"] += 1 / (k + rank + 1)
    
    results_list = [(doc_id, data) for doc_id, data in fused_scores.items()]
    return sorted(results_list, key=lambda x: x[1]["score"], reverse=True)

# --- 3. MAIN LOOP ---
def main():
    print(f"🔄 Connecting to ChromaDB at {CHROMA_PATH}...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error loading ChromaDB: {e}")
        return
    
    print(f"🤖 Loading Embedding Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cpu", model_kwargs={"use_safetensors": True})

    # BM25 Setup (In-Memory)
    print("📚 Building Keyword Index (BM25)...")
    all_docs = collection.get() # Fetch all data
    documents = all_docs['documents']
    ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    if not documents:
        print("⚠️  Database is empty. Run 'indexer.py' first.")
        return

    tokenized_corpus = [normalize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ System Ready.")

    while True:
        query = input("\n❓ Question (q to quit): ").strip()
        if query.lower() == 'q': break
        
        # A. Vector Search
        print(f"   🔎 Searching Vectors...", end="")
        q_embed = model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        
        vec_list = []
        if vec_res['ids']:
            for i in range(len(vec_res['ids'][0])):
                vec_list.append((
                    vec_res['ids'][0][i], 
                    vec_res['documents'][0][i], 
                    vec_res['metadatas'][0][i]
                ))
        print(" Done.")

        # B. Keyword Search
        print(f"   🔡 Searching Keywords...", end="")
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10]
        
        kw_list = []
        for idx in top_n:
            if doc_scores[idx] > 0:
                kw_list.append((ids[idx], documents[idx], metadatas[idx]))
        print(" Done.")

        # C. Fusion (Hybrid Search)
        final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list})
        
        # D. Generate
        answer = generate_answer(query, final_results)
        
        print("\n" + "-"*50)
        print("💡 ANSWER:")
        print(answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
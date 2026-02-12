import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import ollama  # 🔥 NOUVEAU : Import pour parler à Mistral

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# --- 🔥 NOUVEAU : LA FONCTION DE GÉNÉRATION ---
def generate_answer(query, ranked_results):
    """
    Prend la question et les résultats triés, construit un prompt
    et demande à Mistral de rédiger la réponse.
    """
    if not ranked_results:
        return "Désolé, je n'ai trouvé aucun document pertinent pour répondre."

    # 1. Construction du Contexte
    # On prend les 5 meilleurs résultats (Top 5)
    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:5]):
        source = data['meta'].get('source', 'Inconnu')
        text = data['text']
        # On formate clairement pour que le LLM sache d'où vient l'info
        context_pieces.append(f"[Document {rank+1} - Source: {source}]\n{text}")

    full_context = "\n\n---\n\n".join(context_pieces)

    # 2. Le Prompt (Instructions strictes)
    prompt = f"""
    Tu es un assistant juridique expert spécialisé dans le droit algérien.
    Ta mission est de répondre à la question de l'utilisateur en te basant EXCLUSIVEMENT sur le contexte fourni ci-dessous.
    
    RÈGLES IMPORTANTES :
    - Si la réponse n'est pas dans le contexte, dis : "Désolé, cette information ne figure pas dans les documents consultés."
    - Ne cite pas tes connaissances générales si elles ne sont pas dans le texte.
    - Sois précis, cite tes sources (ex: "Selon le décret...").
    - Réponds en français.

    CONTEXTE FOURNI :
    {full_context}

    QUESTION DE L'UTILISATEUR :
    {query}
    """

    print("🤖 Mistral est en train de rédiger la réponse...", end="", flush=True)
    
    try:
        # ... (dans generate_answer) ...
        
        # AJOUTE CECI POUR LE DEBUG :
        print("\n🧐 --- DEBUG : CE QUE MISTRAL LIT ---")
        print(full_context)
        print("--------------------------------------\n")

        print("🤖 Mistral est en train de rédiger la réponse...", end="", flush=True)
        # ...
        # 3. Appel à l'API Ollama
        response = ollama.chat(
            model='mistral', 
            messages=[{'role': 'user', 'content': prompt}]
        )
        print(" Fait !")
        return response['message']['content']
        
    except Exception as e:
        return f"\n❌ Erreur Mistral : {e}"

# --- FONCTIONS EXISTANTES (Ton code) ---

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
    sorted_results = sorted(results_list, key=lambda x: x[1]["score"], reverse=True)
    return sorted_results

def main():
    # 1. SETUP
    print("🔄 Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    print(f"🔄 Loading Model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device="cuda", model_kwargs={"use_safetensors": True})

    # 2. BM25 INDEX
    print("📚 Building BM25 Index...")
    all_docs = collection.get()
    documents = all_docs['documents']
    ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    if not documents:
        print("❌ Database empty.")
        return

    tokenized_corpus = [normalize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ System Ready! Pose ta question.")

    # 3. SEARCH LOOP
    while True:
        query = input("\n❓ Pose ta question (ou 'q' pour quitter) : ").strip()
        if query.lower() == 'q': break
        
        # --- A. VECTOR SEARCH ---
        print(f"   🔎 Recherche Vectorielle...")
        vector_results = collection.query(query_embeddings=model.encode([query]).tolist(), n_results=10)
        vec_list = []
        if vector_results['ids']:
            r_ids, r_docs, r_metas = vector_results['ids'][0], vector_results['documents'][0], vector_results['metadatas'][0]
            for i in range(len(r_ids)): vec_list.append((r_ids[i], r_docs[i], r_metas[i]))

        # --- B. KEYWORD SEARCH ---
        print(f"   🔡 Recherche Mots-clés...")
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:10]
        kw_list = []
        for idx in top_n_indices:
            if doc_scores[idx] > 0:
                kw_list.append((ids[idx], documents[idx], metadatas[idx]))

        # --- C. FUSION ---
        print(f"   ⚗️  Fusion des résultats...")
        final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list})

        # --- D. DISPLAY SOURCES (Optionnel, pour vérifier) ---
        print(f"\n📄 {len(final_results)} documents trouvés. Top 3 Sources :")
        for rank, (doc_id, data) in enumerate(final_results[:3]):
            print(f"   {rank+1}. {data['meta'].get('source', 'Inconnu')} (Score: {data['score']:.4f})")

        # --- 🔥 E. GÉNÉRATION MISTRAL (Le grand final) ---
        answer = generate_answer(query, final_results)

        print("\n" + "="*60)
        print("💡 RÉPONSE DE MISTRAL :")
        print("="*60)
        print(answer)
        print("="*60)

if __name__ == "__main__":
    main()
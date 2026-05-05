import os
import sys
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
import chromadb
from dotenv import load_dotenv, find_dotenv
# 1. Calculate the absolute path to the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 2. Add the parent directory to Python's module search path
sys.path.append(parent_dir) 
# Import the retrieval function we built previously
from retrieve.retrieve import get_retrieved_documents, COLLECTION_NAME, CHROMA_PATH, MODEL_NAME

load_dotenv()

# --- CONFIGURATION ---
# We dedicate GPU 1 for the Embeddings and GPU 2 for the Reranker 
# to keep memory clean and maximize throughput.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" 

RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

def rerank_documents(query, retrieved_docs, reranker_model, top_k=3):
    """
    Takes the loosely retrieved documents from ChromaDB and scores them 
    using a Cross-Encoder for pinpoint accuracy.
    """
    if not retrieved_docs:
        return []

    # 1. Prepare the input pairs for the Cross-Encoder: [[query, doc1], [query, doc2], ...]
    cross_inp = [[query, doc["text"]] for doc in retrieved_docs]

    # 2. Predict the relevance scores
    # The Cross-Encoder outputs a single score for each pair. 
    # HIGHER score = BETTER match (unlike ChromaDB's L2 distance)
    scores = reranker_model.predict(cross_inp)

    # 3. Attach the new scores to our documents
    for i in range(len(retrieved_docs)):
        retrieved_docs[i]["rerank_score"] = float(scores[i])

    # 4. Sort the documents based on the new Cross-Encoder score (Descending)
    reranked_docs = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)

    # 5. Return only the absolute best documents for the LLM
    return reranked_docs[:top_k]

def main():
    
    print("🔄 Initializing Vector Database...")
    
    # 1. Trouve automatiquement l'emplacement physique du fichier .env
    dotenv_path = find_dotenv()
    
    # 2. Récupère le dossier où se trouve le .env
    env_dir = os.path.dirname(dotenv_path)
    
    # 3. Récupère la valeur brute écrite dans le .env ("../data/chroma_db")
    env_chroma = os.getenv("CHROMA_PATH", "../data/chroma_db")
    
    # 4. Fusionne les deux pour obtenir le vrai chemin absolu absolu 
    REAL_CHROMA_PATH = os.path.abspath(os.path.join(env_dir, env_chroma))
    
    print(f"📁 Vrai chemin ChromaDB calculé : {REAL_CHROMA_PATH}")
    
    # 5. Connexion avec le bon chemin !
    chroma_client = chromadb.PersistentClient(path=REAL_CHROMA_PATH)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    
    print("🤖 Loading Bi-Encoder (BGE-M3) on GPU 0...")
    # ... (le reste du code reste strictement identique) ...
    print("🤖 Loading Bi-Encoder (BGE-M3) on GPU 0...")
    # This uses the first available GPU in our CUDA_VISIBLE_DEVICES list (which is physical GPU 1)
    bi_encoder = SentenceTransformer(MODEL_NAME, device="cuda:0")

    print("🧠 Loading Cross-Encoder Reranker on GPU 1...")
    # This uses the second available GPU (physical GPU 2)
    # Using float16 saves VRAM and speeds up inference without losing accuracy
    reranker = CrossEncoder(RERANKER_MODEL_NAME, device="cuda:1", model_kwargs={"torch_dtype": torch.float16})
    
    print("✅ RAG Pipeline Ready!\n" + "="*50)

    while True:
        query = input("\n❓ Entrez votre requête juridique (ou 'q' pour quitter) : ").strip()
        if query.lower() == 'q': 
            break
            
        print("\n🔍 STAGE 1: Fast Retrieval (Bi-Encoder)...")
        # Get the top 20 broad matches from our optimized retriever
        initial_docs, strategy = get_retrieved_documents(query, bi_encoder, collection, top_k=20)
        
        if not initial_docs:
            print("❌ Aucun document trouvé.")
            continue
            
        print(f"   > Found {len(initial_docs)} candidates using {strategy.upper()} strategy.")
        
        print("🎯 STAGE 2: Deep Reranking (Cross-Encoder)...")
        # Pass the 20 candidates through the Cross-Encoder and keep the Top 3
        final_docs = rerank_documents(query, initial_docs, reranker, top_k=3)

        print("\n" + "="*80)
        print("🏆 FINAL TOP 3 DOCUMENTS FOR THE LLM")
        print("="*80)
        
        for i, doc in enumerate(final_docs):
            meta = doc['meta']
            source_file = meta.get('source_file', 'Inconnu')
            
            print(f"\n🥇 RANK [{i+1}] - Cross-Encoder Score: {doc['rerank_score']:.4f}")
            print(f"   SOURCE : {source_file}")
            print(f"   METHOD : {meta.get('chunking_method', 'N/A').upper()}")
            print("-" * 80)
            print(doc['text'][:400] + "..." if len(doc['text']) > 400 else doc['text'])
            print("-" * 80)

if __name__ == "__main__":
    main()
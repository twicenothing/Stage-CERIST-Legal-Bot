import os
import string
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from ollama import Client  # 🔥 Imported Client for custom port configuration

# 🔥 Hide blocked GPU 0, use empty GPUs for the embedding model
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# 🔥 Custom Ollama Port setup
OLLAMA_HOST = "http://127.0.0.1:11435"
ollama_client = Client(host=OLLAMA_HOST)

# 🔥 Fallback Threshold: L2 distance ranges from 0 (perfect match) to ~2 (no match). 
# If the top regex chunk's distance is higher than this, it triggers the recursive search.
FALLBACK_DISTANCE_THRESHOLD = 1.0 

# --- 1. GENERATION WITH CONTEXT AWARENESS ---
def generate_answer(question, ranked_results):
    if not ranked_results:
        return "Désolé, je n'ai trouvé aucun document pertinent dans la base de données."

    # --- 🛠️ DEBUG LOGS ---
    print(f"\n" + "="*60)
    print(f"🛠️ DEBUG : TOP 3 DOCUMENTS EXTRAITS PAR LE RAG")
    print("="*60)
    
    for i, (doc_id, data) in enumerate(ranked_results[:3]):
        score = data['score']
        text = data['text']
        
        print(f"\n🔹 DOCUMENT [{i+1}] - ID: {doc_id} (RRF Score: {score:.4f})")
        print("-" * 40)
        print(text)
        print("-" * 40)
    print("\n" + "="*60 + "\n")

    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:3]):
        meta = data['meta']
        text = data['text']
        source_title = meta.get('parent_title', meta.get('title', 'Document sans titre'))
        context_pieces.append(f"DOCUMENT {rank+1} (Source: {source_title})\nCONTENU: {text}")

    full_context = "\n\n---\n\n".join(context_pieces)

    # 🔥 Updated System Prompt
    prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.

⚠️ RÈGLES STRICTES DE RÉDACTION :
1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement, puis justifie en citant la base légale.
3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant.
4. PRÉCISION CHIRURGICALE : Reproduis fidèlement les noms propres, montants, et utilise des listes si nécessaire.
5. NETTOYAGE VISUEL : Ignore les pointillés ("....") dans le texte brut.

CONTEXTE FOURNI :
{full_context}

QUESTION DE L'UTILISATEUR :
{question}

RÉPONSE :
"""

    print("🤖 Modele rédige...", end="", flush=True)
    try:
        # 🔥 Using the custom client and mixtral model
        response = ollama_client.chat(
            model='mistral-nemo', 
            messages=[{'role': 'user', 'content': prompt}]
        )
        print(" ✅")
        return response['message']['content']
    except Exception as e:
        return f"\n❌ Erreur Ollama : {e}\n(Avez-vous bien lancé 'OLLAMA_HOST=127.0.0.1:11435 ollama serve' ?)"

# --- 2. UTILS ---
def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def reciprocal_rank_fusion(results_dict, weights=None, k=60):
    if weights is None:
        weights = {system: 1.0 for system in results_dict.keys()}
        
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        weight = weights.get(system_name, 1.0) 
        
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "text": doc_text, "meta": metadata}
            fused_scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
            
    return sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)

# --- 3. RETRIEVAL LOGIC ---
def retrieve_chunks(query, q_embed, chunk_type, collection, bm25_index, data_dict):
    """Effectue la recherche Vectorielle + BM25 sur un type de chunk spécifique (regex ou recursive)."""
    
    # A. Vector Search (Filtered by chunking_method)
    vec_res = collection.query(
        query_embeddings=q_embed, 
        n_results=10,
        where={"chunking_method": chunk_type} # 🔥 Isolate by type
    )
    
    vec_list = []
    top_distance = 999.0 # Valeur par défaut si aucun résultat
    
    if vec_res['ids'] and len(vec_res['ids'][0]) > 0:
        top_distance = vec_res['distances'][0][0] # Chroma uses L2 distance (Lower is better)
        for i in range(len(vec_res['ids'][0])):
            vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

    # B. Keyword Search (BM25)
    kw_list = []
    if bm25_index:
        tokenized_query = normalize_text(query)
        doc_scores = bm25_index.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10]
        
        for idx in top_n:
            if doc_scores[idx] > 0:
                kw_list.append((data_dict['ids'][idx], data_dict['docs'][idx], data_dict['metas'][idx]))

    # C. Fusion RRF
    best_weights = {"keyword": 0.7, "vector": 0.3}
    final_results = reciprocal_rank_fusion(
        {"vector": vec_list, "keyword": kw_list}, 
        weights=best_weights
    )
    
    return final_results, top_distance

# --- 4. MAIN LOOP ---
def main():
    print(f"🔄 Connexion à ChromaDB...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return
    
    print("🤖 Chargement du modèle d'embedding BGE-M3...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print("📚 Séparation des documents pour l'index BM25...")
    all_docs = collection.get()
    
    if not all_docs['documents']:
        print("⚠️ Base vide.")
        return

    # 🔥 SPLIT THE DATA FOR DUAL BM25 INDEXING
    regex_data = {'docs': [], 'ids': [], 'metas': []}
    recursive_data = {'docs': [], 'ids': [], 'metas': []}

    for doc, doc_id, meta in zip(all_docs['documents'], all_docs['ids'], all_docs['metadatas']):
        if meta.get('chunking_method') == 'regex':
            regex_data['docs'].append(doc)
            regex_data['ids'].append(doc_id)
            regex_data['metas'].append(meta)
        else:
            recursive_data['docs'].append(doc)
            recursive_data['ids'].append(doc_id)
            recursive_data['metas'].append(meta)

    # Initialisation des deux moteurs BM25
    bm25_regex = BM25Okapi([normalize_text(d) for d in regex_data['docs']]) if regex_data['docs'] else None
    bm25_recursive = BM25Okapi([normalize_text(d) for d in recursive_data['docs']]) if recursive_data['docs'] else None
    
    print(f"✅ Système Prêt. ({len(regex_data['docs'])} Regex chunks, {len(recursive_data['docs'])} Recursive chunks)")

    while True:
        query = input("\n❓ Question (q pour quitter) : ").strip()
        if query.lower() == 'q': break
        
        q_embed = model.encode([query]).tolist()
        
        # 🔥 DOUBLE CHUNKING LOGIC : ETAPE 1 (REGEX)
        print("🔍 Recherche dans les chunks Regex structurés...")
        final_results, top_dist = retrieve_chunks(query, q_embed, "regex", collection, bm25_regex, regex_data)
        
        # 🔥 DOUBLE CHUNKING LOGIC : ETAPE 2 (RECURSIVE FALLBACK)
        # ChromaDB retourne une distance L2. Si elle est > 1.0, la similarité sémantique est faible.
        if top_dist > FALLBACK_DISTANCE_THRESHOLD or not final_results:
            print(f"⚠️ Score de pertinence trop faible (Distance: {top_dist:.2f}). Basculement sur les chunks Récursifs...")
            final_results, _ = retrieve_chunks(query, q_embed, "recursive", collection, bm25_recursive, recursive_data)
        else:
            print(f"🎯 Documents structurés trouvés (Distance: {top_dist:.2f}).")

        # Génération
        answer = generate_answer(query, final_results)
        
        print("\n" + "-"*50)
        print(f"💡 RÉPONSE :\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
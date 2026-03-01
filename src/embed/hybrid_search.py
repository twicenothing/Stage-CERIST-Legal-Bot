import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import ollama

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# --- 1. GENERATION WITH CONTEXT AWARENESS ---
def generate_answer(query, ranked_results):
    if not ranked_results:
        return "Désolé, je n'ai trouvé aucun document pertinent dans la base de données."

    # --- 🛠️ DEBUG LOGS : VISUALISER LE TEXTE EXACT ENVOYÉ À MISTRAL ---
    print(f"\n" + "="*60)
    print(f"🛠️ DEBUG : TOP 3 DOCUMENTS EXTRAITS PAR LE RAG")
    print("="*60)
    
    for i, (doc_id, data) in enumerate(ranked_results[:3]):
        score = data['score']
        text = data['text']
        
        print(f"\n🔹 DOCUMENT [{i+1}] - ID: {doc_id} (Score: {score:.4f})")
        print("-" * 40)
        print(text)
        print("-" * 40)
    print("\n" + "="*60 + "\n")

    # Préparation du contexte pour Mistral
    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:3]):
        meta = data['meta']
        text = data['text']
        
        source_title = meta.get('parent_title', meta.get('title', 'Document sans titre'))
        context_pieces.append(f"DOCUMENT {rank+1} (Source: {source_title})\nCONTENU: {text}")

    full_context = "\n\n---\n\n".join(context_pieces)

    # Prompt renforcé
    prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
        Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.

        ⚠️ RÈGLES STRICTES DE RÉDACTION :
        1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
        2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement (ex: "Oui.", "C'est M. X.", "Il s'agit du décret..."), puis justifie en citant la base légale.
        3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant tels qu'ils apparaissent dans la Source (ex: "Selon l'article 3 du décret n° 25-74..."). Ne dis JAMAIS "D'après le document 1" ou "Selon la source fournie".
        4. PRÉCISION CHIRURGICALE : 
        - Noms et Fonctions : Reproduis fidèlement les noms propres, institutions et intitulés officiels.
        - Nombres et Budgets : Écris les montants, durées ou quantités exactes. S'il s'agit d'un budget, fais bien la distinction entre les crédits "annulés" et "ouverts/appliqués".
        - Énumérations : Utilise une liste numérotée si la réponse contient plusieurs éléments distincts (comme des sous-directions ou des délégations).
        5. NETTOYAGE VISUEL : Ignore les pointillés ("....") ou les mentions d'édition type "(sans changement)" présents dans le texte brut.

        CONTEXTE FOURNI :
        {full_context}

        QUESTION DE L'UTILISATEUR :
        {query}

        RÉPONSE :
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
        return f"\n❌ Erreur Ollama : {e}"

# --- 2. UTILS ---
def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

# 🛑 MISE À JOUR : La fonction accepte maintenant les poids (weights)
def reciprocal_rank_fusion(results_dict, weights=None, k=60):
    if weights is None:
        weights = {system: 1.0 for system in results_dict.keys()}
        
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        weight = weights.get(system_name, 1.0) # On récupère le poids
        
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "text": doc_text, "meta": metadata}
            # RRF Formula avec le multiplicateur de poids
            fused_scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
    
    return sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)

# --- 3. MAIN LOOP ---
def main():
    print(f"🔄 Connexion à ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return
    
    print("🤖 Chargement du modèle d'embedding...")
    model = SentenceTransformer(
        MODEL_NAME, 
        device="cpu", 
        model_kwargs={"use_safetensors": True}
    )

    print("📚 Chargement de l'index BM25...")
    all_docs = collection.get()
    documents = all_docs['documents']
    ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    if not documents:
        print("⚠️ Base vide.")
        return

    tokenized_corpus = [normalize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    print("✅ Système Prêt.")

    while True:
        query = input("\n❓ Question (q pour quitter) : ").strip()
        if query.lower() == 'q': break
        
        # A. Vector Search
        q_embed = model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        
        vec_list = []
        if vec_res['ids']:
            for i in range(len(vec_res['ids'][0])):
                vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

        # B. Keyword Search (BM25)
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10]
        
        kw_list = []
        for idx in top_n:
            if doc_scores[idx] > 0:
                kw_list.append((ids[idx], documents[idx], metadatas[idx]))

        # C. Fusion
        # 🛑 MISE À JOUR : Application des poids validés par l'évaluation
        best_weights = {"keyword": 0.7, "vector": 0.3}
        final_results = reciprocal_rank_fusion(
            {"vector": vec_list, "keyword": kw_list}, 
            weights=best_weights
        )
        
        # D. Generate
        answer = generate_answer(query, final_results)
        
        print("\n" + "-"*50)
        print(f"💡 RÉPONSE :\n{answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()
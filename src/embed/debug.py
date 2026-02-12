import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import numpy as np

# CONFIG
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
QUERY = "Décret présidentiel n° 24-440"  # La cible exacte

def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def main():
    print(f"🕵️‍♂️ ENQUÊTE SUR : '{QUERY}'")
    
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    all_docs = collection.get()
    
    # 1. EST-CE QUE LE TEXTE EXISTE ?
    found_count = 0
    print("\n1. RECHERCHE BRUTE (Ctrl+F)...")
    for i, doc in enumerate(all_docs['documents']):
        # On cherche juste "24-440" pour être large
        if "24-440" in doc:
            found_count += 1
            print(f"   ✅ TROUVÉ dans le document ID: {all_docs['ids'][i]}")
            print(f"      Extrait : {doc[:150]}...") # Affiche le début du chunk
            print("-" * 20)
            
    if found_count == 0:
        print("   ❌ FATAL : La chaîne '24-440' n'existe nulle part dans la base ChromaDB !")
        print("   -> Vérifie ton OCR ou ton script d'indexation.")
        return

    # 2. POURQUOI IL NE SORT PAS ? (Test BM25)
    print("\n2. TEST DU SCORE BM25...")
    tokenized_corpus = [normalize_text(d) for d in all_docs['documents']]
    bm25 = BM25Okapi(tokenized_corpus)
    
    tokenized_query = normalize_text(QUERY)
    scores = bm25.get_scores(tokenized_query)
    
    # On regarde les meilleurs scores BM25
    top_indices = np.argsort(scores)[::-1][:5]
    
    print("   Les gagnants du BM25 sont :")
    for idx in top_indices:
        doc_content = all_docs['documents'][idx]
        has_target = "24-440" in doc_content
        print(f"   - Score: {scores[idx]:.4f} | Contient '24-440' ? {'✅ OUI' if has_target else '❌ NON'}")
        if has_target:
            print("     -> C'est celui qu'on veut ! Pourquoi n'est-il pas sorti dans le Hybrid Search ?")

if __name__ == "__main__":
    main()
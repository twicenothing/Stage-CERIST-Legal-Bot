import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3

# On cible uniquement les questions qui ont échoué
TARGET_IDS = [1, 5, 7, 15, 23, 25]

# Ton dataset (j'ai gardé que les cibles pour alléger le code)
dataset = {
    "samples" : [
        {"id": 1, "question" : "Pourquoi la juridiction suprême a-t-elle refusé d'accéder à la requête des parlementaires concernant l'article 158 ?", "answer" : "La Cour a rejeté la demande sur le fond car elle estime que les dispositions de l'article 158 sont parfaitement claires."},
        # J'ai corrigé la référence de la Q5 pour le test ! Mets la vraie réponse si tu l'as.
        {"id": 5, "question" : "Quelle est la nouvelle échéance accordée aux groupements d'agriculteurs pour se mettre en règle avec la législation de 1996 qui les régit ?", "answer" : "Le délai de mise en conformité pour les coopératives agricoles et leurs unions (vis-à-vis du décret de 1996) a été repoussé au 31 décembre 2025."},
        {"id": 7, "question" : "La liste des membres du Conseil national économique, social et environnemental (CNESE) nommés en janvier 2025 est-elle complète, et quelle est la durée de leur mandat ?", "answer" : "Non, la liste des membres nommés en janvier 2025 n'est pas exhaustive ; les membres restants seront désignés ultérieurement. Pour ceux déjà nommés, la durée de leur mandat est de quatre (4) ans."},
        {"id": 15, "question": "D'après l'annexe de l'accord de coopération culturelle et scientifique entre l'Algérie et l'Allemagne, à quelles conditions strictes un expert allemand détaché en Algérie peut-il importer son véhicule personnel sans payer de droits de douane, et quand aura-t-il le droit de le revendre sur place ?", "answer": "L'expert peut importer son véhicule en franchise de droits de douane à deux conditions : le véhicule doit avoir été utilisé pendant au moins 6 mois avant le transfert, et il doit être dédouané dans les 12 mois suivant son installation. Pour le revendre (ou le céder gratuitement) en Algérie, il doit obligatoirement attendre un délai de 12 mois, sauf s'il décide de payer les droits de douane au préalable (Annexe, Paragraphe 3)."},
        {"id": 23, "question": "D'après l'arrêté du 19 décembre 2024 portant nomination au conseil d'administration du musée national du moudjahid, qui a été désigné comme président de ce conseil et quel ministère représente-t-il ? De plus, quels sont les noms exacts des représentants nommés au titre de l'organisation nationale des enfants de chouhada ?", "answer": "Selon cet arrêté, le président du conseil d'administration est Alallou Abdelhamid, qui siège en tant que représentant du ministre des moudjahidine et des ayants droit. Par ailleurs, l'organisation nationale des enfants de chouhada est exceptionnellement représentée par deux membres : Abidli Mohamed Amine et Bakhouche Mokhtar."},
        {"id": 25, "question": "Selon le décret présidentiel n° 25-03 du 6 janvier 2025, quelle est la durée du mandat d'un membre du Conseil (et est-ce renouvelable ?), comparée à celle d'un membre du bureau ou d'un président de commission ? De plus, à combien de commissions permanentes un membre peut-il appartenir au maximum ?", "answer": "D'après l'article 8 modifié, le mandat d'un membre du Conseil est de quatre (4) ans, renouvelable une seule fois. En revanche, les membres du bureau (article 41) et les présidents des commissions permanentes (article 45) sont élus pour un mandat de deux (2) ans, non renouvelable. Enfin, l'article 45 précise qu'un membre du Conseil ne peut faire partie de plus de deux (2) commissions permanentes."}
    ]
}

def call_mistral(prompt_text):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "mistral",
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": False,
        "options": {"temperature": 0.0}
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Exception as e:
        return f"Erreur API: {e}"

def debug_pipeline():
    print("🔄 Chargement de ChromaDB et du modèle BGE-M3...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    for sample in dataset["samples"]:
        q_id = sample["id"]
        if q_id not in TARGET_IDS:
            continue
            
        question = sample["question"]
        print("\n" + "="*80)
        print(f"🎯 INVESTIGATION - QUESTION {q_id}")
        print(f"❓ Question : {question}")
        print("="*80)
        
        # 1. ÉTAPE RETRIEVAL (On fouille la base de données)
        query_emb = model.encode(question).tolist()
        results = collection.query(query_embeddings=[query_emb], n_results=TOP_K)
        
        retrieved_docs = results['documents'][0]
        full_context = "\n\n---\n\n".join(retrieved_docs)
        
        print("\n📄 --- DOCUMENTS RÉCUPÉRÉS PAR CHROMADB ---")
        for i, doc in enumerate(retrieved_docs):
            # On n'affiche que les 300 premiers caractères pour ne pas inonder le terminal
            print(f"[{i+1}] {doc[:300]}...") 
        print("--------------------------------------------\n")
        
        # 2. PROMPT ENGINEERING (La nouvelle recette pour Mistral)
        # On utilise une structure plus directive pour forcer le modèle à bien lire.
        prompt = f"""Tu es un expert juridique algérien très précis. 
Réponds à la question en te basant UNIQUEMENT sur le contexte fourni.

CONTEXTE :
{full_context}

QUESTION :
{question}

INSTRUCTIONS STRICTES :
1. Cherche la réponse exacte dans le contexte.
2. Si la réponse s'y trouve, donne-la de façon directe et détaillée, en citant les articles.
3. Si la réponse est totalement absente du contexte, écris EXACTEMENT : "Information absente des documents."
4. Ne confonds pas les durées ou les rôles s'il y en a plusieurs.

RÉPONSE :"""

        # 3. GÉNÉRATION
        answer = call_mistral(prompt)
        print(f"🤖 NOUVELLE RÉPONSE MISTRAL :\n{answer}\n")
        print(f"✅ RÉFÉRENCE (Ce qu'on attendait) :\n{sample['answer']}")

if __name__ == "__main__":
    debug_pipeline()
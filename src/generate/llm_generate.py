import os
import sys
import time
import math  # Ajouté pour le calcul de la sigmoïde
from ollama import Client
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# ==============================================================================
# 🔐 SÉCURITÉ DES CHEMINS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

sys.path.append(src_dir) 

try:
    from rerank.rerank import get_best_documents_for_llm
except ImportError:
    from rerank import get_best_documents_for_llm

try:
    from query_parse import rewrite_query
except ImportError:
    from query_parse import rewrite_query

# --- CONFIGURATION DEPUIS .ENV ---
ENV_CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
if not os.path.isabs(ENV_CHROMA_PATH):
    clean_path = ENV_CHROMA_PATH[2:] if ENV_CHROMA_PATH.startswith("./") else ENV_CHROMA_PATH
    ABSOLUTE_CHROMA_PATH = os.path.join(project_root, clean_path)
else:
    ABSOLUTE_CHROMA_PATH = ENV_CHROMA_PATH

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

ollama_client = Client(host=OLLAMA_HOST)

# ==============================================================================
# 🛠️ INITIALISATION DU PIPELINE RAG
# ==============================================================================
def init_rag_pipeline():
    print(f"⏳ Connexion à la base ChromaDB existante : {ABSOLUTE_CHROMA_PATH}")
    
    if not os.path.exists(ABSOLUTE_CHROMA_PATH):
        print(f"❌ ERREUR CRITIQUE : Le dossier de la base est introuvable à {ABSOLUTE_CHROMA_PATH}.")
        sys.exit(1)
        
    db_client = chromadb.PersistentClient(path=ABSOLUTE_CHROMA_PATH)
    
    try:
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' trouvée.")
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : Impossible de trouver la collection '{COLLECTION_NAME}'.\n{e}")
        sys.exit(1)
    
    print(f"⏳ Chargement du Bi-Encoder ({EMBEDDING_MODEL})...")
    bi_encoder = SentenceTransformer(EMBEDDING_MODEL)
    
    print(f"⏳ Chargement du Cross-Encoder ({RERANKER_MODEL})...")
    reranker = CrossEncoder(RERANKER_MODEL)
    
    return collection, bi_encoder, reranker


# ==============================================================================
# 🧠 GÉNÉRATION LLM
# ==============================================================================
def generate_legal_response(question, retrieved_docs, model_name=LLM_MODEL):
    formatted_context = ""
    for i, doc in enumerate(retrieved_docs):
        # 🔥 RÉCUPÉRATION DU VRAI TITRE JURIDIQUE AU LIEU DU PDF
        titre_juridique = doc['meta'].get('parent_title', 'Texte de loi inconnu')
        doc_type = doc['meta'].get('document_type', 'Extrait')
        page = doc['meta'].get('page', 'Inconnu')
        
        # 👈 Le LLM lit maintenant un texte 100% naturel
        formatted_context += f"--- SOURCE : {titre_juridique} | PAGE : {page} ({doc_type}) ---\n"
        formatted_context += f"{doc['text']}\n\n"

    # 🔥 Prompt strict pour empêcher le LLM de discuter ou d'inventer
    system_prompt = """Tu es un assistant juridique strict. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

RÈGLES DE FORMATAGE STRICTES (À RESPECTER ABSOLUMENT) :
1. INTERDICTION FORMELLE d'utiliser des phrases d'introduction ou de conclusion. Ne dis JAMAIS "En vertu des instructions", "Après examen", "Je vais analyser", etc.
2. INTERDICTION d'expliquer ton raisonnement. Ne décris pas ce que tu as trouvé avant de répondre.
3. Commence DIRECTEMENT ta réponse.

RÈGLE CRITIQUE DE REJET :
Si l'information exacte ne se trouve pas dans les documents, tu NE DOIS RIEN ÉCRIRE D'AUTRE que cette phrase exacte :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
N'ajoute AUCUN préfixe. Juste cette phrase unique.

FORMAT SI LA RÉPONSE EST TROUVÉE :
- Réponds de manière directe, factuelle et concise.
- Utilise des listes à puces si nécessaire.
- Cite obligatoirement tes sources de manière naturelle (Type de texte, Numéro, Page, Article).

=== EXEMPLES DE COMPORTEMENT ATTENDU ===

Exemple 1 (Information présente) :
<documents>
--- SOURCE : Décret exécutif n° 23-64 du 14 Rajab 1444 correspondant au 5 février 2023 | PAGE : 3 (Décret) ---
Contenu : Art. 2. — La réalisation et l'exploitation d'un aérodrome destiné à l'usage privé, sont soumises à l'autorisation de l'autorité chargée de l'aviation civile.
</documents>
<question>Qui autorise la création d'un aérodrome privé ?</question>
Réponse directe :
La réalisation et l'exploitation d'un aérodrome à usage privé nécessitent l'autorisation de l'autorité chargée de l'aviation civile.
- [Source : Décret exécutif n° 23-64, Page 3, Art. 2]

Exemple 2 (Information absente) :
<documents>
--- SOURCE : Arrêté interministériel du 5 Rajab 1429 | PAGE : 5 (Arrêté) ---
Contenu : Art. 1. — Le présent arrêté fixe le tarif des redevances.
</documents>
<question>Quelle est la durée du congé maternité ?</question>
Réponse directe :
Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information.
"""




    user_prompt = f"""<documents>
{formatted_context}
</documents>

<question>
{question}
</question>

Réponse directe :"""

    # ==================================================================
    # 🕵️ DEBUG : VOIR EXACTEMENT CE QUE LE LLM REÇOIT
    # ==================================================================
    print("\n" + "👁️"*40)
    print("👁️  DEBUG : PROMPT COMPLET ENVOYÉ AU LLM")
    print("👁️"*40)
    print("\n[--- SYSTEM PROMPT ---]")
    print(system_prompt)
    print("\n[--- USER PROMPT ---]")
    print(user_prompt)
    print("👁️"*40 + "\n")
    # ==================================================================

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                "temperature": 0.0,
                "num_ctx": 8192  # 👈 Fenêtre de contexte augmentée pour gérer 8+ documents
            }
        )
        return response['message']['content']
        
    except Exception as e:
        return f"❌ Erreur de connexion à Ollama ({OLLAMA_HOST}) : {e}\nAssurez-vous qu'Ollama est lancé."


def main():
    print("🚀 Démarrage du Legal Bot CERIST...")
    print(f"⚙️  Modèle LLM configuré : {LLM_MODEL}")
    print(f"⚙️  Hôte Ollama : {OLLAMA_HOST}")
    
    collection, bi_encoder, reranker = init_rag_pipeline()
    
    print("\n✅ Système prêt ! Posez vos questions.")
    print("="*60)

    while True:
        original_question = input("\n❓ Question juridique (ou 'q' pour quitter) : ").strip()
        if original_question.lower() == 'q':
            break

        start_time = time.time()

        print("🪄  Optimisation de la requête pour la base de données...")
        optimized_question = rewrite_query(original_question)
        print("optimized_question", optimized_question)

        print("🔍 Recherche et reclassement des documents en cours...")
        best_docs = get_best_documents_for_llm(
            optimized_question, 
            collection, 
            bi_encoder, 
            reranker, 
            top_k_retrieve=8, 
            top_k_rerank=3
        )

        if not best_docs:
            print("\n🤖 Réponse : Les documents fournis ne contiennent pas cette information.")
            continue

        print(f"🤖 Analyse juridique en cours par {LLM_MODEL}...")
        
        # C'est ici que l'affichage du Prompt Complet va se déclencher
        reponse_llm = generate_legal_response(original_question, best_docs)

        end_time = time.time()

        print("\n" + "="*80)
        print("⚖️ RÉPONSE DU BOT JURIDIQUE :")
        print("="*80)
        print(reponse_llm)
        print("\n" + "-"*80)
        print(f"⏱️ Temps total de traitement : {end_time - start_time:.2f} secondes")
        print("-"*80)

if __name__ == "__main__":
    main()
import os
import sys
import time
from ollama import Client
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# ==============================================================================
# 🔐 SÉCURITÉ DES CHEMINS (Adapté à ./src/generate/llm_generate.py)
# ==============================================================================
# current_dir  = src/generate/
# src_dir      = src/
# project_root = / (racine du projet)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

# On ajoute src/ au PYTHONPATH pour pouvoir importer rerank.rerank
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
# On convertit le chemin relatif de l'environnement en chemin absolu strict
ENV_CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
if not os.path.isabs(ENV_CHROMA_PATH):
    # Enlève le "./" si présent pour concaténer proprement
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
# 🛠️ INITIALISATION DU PIPELINE RAG (MODE LECTURE SEULE STRICTE)
# ==============================================================================
def init_rag_pipeline():
    print(f"⏳ Connexion à la base ChromaDB existante : {ABSOLUTE_CHROMA_PATH}")
    
    if not os.path.exists(ABSOLUTE_CHROMA_PATH):
        print(f"❌ ERREUR CRITIQUE : Le dossier de la base de données est introuvable à l'adresse {ABSOLUTE_CHROMA_PATH}.")
        sys.exit(1)
        
    db_client = chromadb.PersistentClient(path=ABSOLUTE_CHROMA_PATH)
    
    # 🔥 STRICTEMENT 'get_collection' : On ne crée RIEN de nouveau.
    try:
        collection = db_client.get_collection(name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' trouvée avec succès.")
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : Impossible de trouver la collection '{COLLECTION_NAME}' dans la base.")
        print(f"Détail de l'erreur : {e}")
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
        source = doc['meta'].get('source_file', f'Document inconnu {i+1}')
        article = doc['meta'].get('document_type', 'Extrait')
        
        formatted_context += f"--- SOURCE : {source} ({article}) ---\n"
        formatted_context += f"{doc['text']}\n\n"

    system_prompt = """Tu es un assistant juridique strict et expert en droit administratif algérien. Ta mission exclusive est de répondre aux questions en te basant UNIQUEMENT sur les documents fournis dans la balise <documents>.

DIRECTIVES ABSOLUES (TOLÉRANCE ZÉRO POUR L'INVENTION) :
1. Tu ne possèdes aucune connaissance externe. Si l'information n'est pas explicitement écrite dans les <documents>, tu l'ignores.
2. Ne fais aucune déduction logique au-delà de ce qui est strictement écrit.
3. Si les documents ne contiennent pas la réponse complète, fournis uniquement ce qui est disponible et précise ce qui manque.

MÉTHODOLOGIE DE RÉPONSE :
Étape 1 : Analyse silencieusement la question et cherche les correspondances exactes dans le texte fourni.
Étape 2 : Si aucune correspondance n'est trouvée, arrête-toi immédiatement et applique la RÈGLE CRITIQUE ci-dessous.
Étape 3 : Si l'information est présente, rédige ta réponse de manière formelle, concise et objective.

FORMAT DE SORTIE EXIGÉ :
- Utilise des listes à puces pour énumérer les conditions ou les articles si nécessaire.
- Tu DOIS citer tes sources à la fin de chaque affirmation importante (ex: [Source : Loi n° 90-11, Art. 4]).

RÈGLE CRITIQUE DE REJET :
Si la réponse ne se trouve pas de manière évidente dans les documents, ta réponse finale DOIT être EXACTEMENT et UNIQUEMENT cette phrase :
"Je suis désolé, je n'ai pas la réponse à cette question car la base de données ne contient pas cette information."
Ne rajoute aucune explication à cette phrase."""

user_prompt = f"""Veuillez analyser les documents de référence suivants pour répondre à la question.

<documents>
{formatted_context}
</documents>

<question>
{question}
</question>

Réponse (Rappel : cite tes sources et n'invente rien) :"""

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                "temperature": 0.0 
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

        optimized_question = rewrite_query(original_question)

        print("🔍 Recherche et reclassement des documents en cours...")
        best_docs = get_best_documents_for_llm(optimized_question, collection, bi_encoder, reranker)

        if not best_docs:
            print("\n🤖 Réponse : Les documents fournis ne contiennent pas cette information.")
            continue

        # --- DEBUG DES DOCUMENTS ---
        print("\n" + "="*80)
        print("🛠️ DEBUG : TOP 3 DOCUMENTS ENVOYÉS AU LLM")
        print("="*80)
        for i, doc in enumerate(best_docs):
            meta = doc.get('meta', {})
            source = meta.get('source_file', 'Inconnu')
            method = meta.get('chunking_method', 'N/A')
            score = doc.get('rerank_score', 'N/A')
            
            if isinstance(score, float):
                print(f"🥇 DOC [{i+1}] | Score Cross-Encoder: {score:.4f} | Source: {source} ({method})")
            else:
                print(f"🥇 DOC [{i+1}] | Source: {source} ({method})")
                
            print("-" * 80)
            texte = doc.get('text', '')
            print(texte[:400] + "..." if len(texte) > 400 else texte)
            print("\n")
        print("="*80 + "\n")
        # ---------------------------

        print(f"🤖 Analyse juridique en cours par {LLM_MODEL}...")
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
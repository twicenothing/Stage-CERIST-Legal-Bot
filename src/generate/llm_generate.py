import os
import sys
import time
import ollama
from dotenv import load_dotenv
load_dotenv()

# --- SÉCURITÉ DES CHEMINS ---
# Permet d'importer le dossier rerank, peu importe où tu places ce fichier
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 

# Import de ton pipeline RAG (assure-toi que le dossier rerank est accessible)
try:
    from rerank.rerank import init_rag_pipeline, get_best_documents_for_llm
except ImportError:
    # Si le fichier est directement dans src/, on importe juste 'rerank'
    from rerank import init_rag_pipeline, get_best_documents_for_llm


LLM_MODEL = os.getenv("LLM_MODEL", "mistral:latest")

def generate_legal_response(question, retrieved_docs, model_name="mistral:latest"):
    """
    Prend les documents issus du Reranker, les formate, et génère une réponse
    en utilisant Mistral via Ollama.
    """
    
    # 1. Formatage du contexte (On concatène les textes des documents)
    formatted_context = ""
    for i, doc in enumerate(retrieved_docs):
        source = doc['meta'].get('source_file', f'Document inconnu {i+1}')
        article = doc['meta'].get('document_type', 'Extrait')
        
        formatted_context += f"--- SOURCE : {source} ({article}) ---\n"
        formatted_context += f"{doc['text']}\n\n"

    # 2. Ton System Prompt exact
    system_prompt = """Tu es un assistant juridique expert en droit administratif algérien. Ta seule et unique mission est de répondre aux questions de l'utilisateur en te basant STRICTEMENT et EXCLUSIVEMENT sur les documents de référence fournis.

MÉTHODOLOGIE OBLIGATOIRE :
Tu es un modèle de raisonnement. Tu vas automatiquement utiliser ta balise <think> pour réfléchir avant de répondre. 
Dans cette réflexion :
1. Identifie les concepts clés de la question.
2. Cherche attentivement ces concepts dans les documents fournis.
3. Détermine si l'information s'y trouve réellement.

RÈGLES POUR LA RÉPONSE FINALE (Après la balise <think>) :
1. RÈGLE CRITIQUE : Si ton analyse conclut que la réponse ne figure pas dans les documents, ta réponse finale DOIT être EXACTEMENT : "Les documents fournis ne contiennent pas cette information."
2. Si la réponse s'y trouve, réponds de manière directe, factuelle et précise.
3. Cite toujours le numéro de l'Article ou l'intitulé du texte de loi justifiant ta réponse."""

    # 3. Ton User Prompt exact
    user_prompt = f"""Voici les documents de référence :

{formatted_context}

Question de l'utilisateur : {question}

Réponse :"""

    # 4. Appel à Ollama (API synchrone)
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                "temperature": 0.0 # Zéro hallucination pour le domaine juridique
            }
        )
        
        return response['message']['content']
        
    except Exception as e:
        return f"❌ Erreur de connexion à Ollama : {e}\nAssurez-vous qu'Ollama est lancé en arrière-plan."


def main():
    print("🚀 Démarrage du Legal Bot CERIST...")
    print("⏳ Chargement des bases et des modèles sur les GPUs (Patientez...)")
    
    # ÉTAPE 1 : Initialisation de ChromaDB, BGE-M3 et du Reranker
    collection, bi_encoder, reranker = init_rag_pipeline()
    
    print("\n✅ Système prêt ! Posez vos questions.")
    print("="*60)

    while True:
        question = input("\n❓ Question juridique (ou 'q' pour quitter) : ").strip()
        if question.lower() == 'q':
            break

        start_time = time.time()

        # ÉTAPE 2 : Retrieval & Reranking 
        print("🔍 Recherche et reclassement des documents en cours...")
        best_docs = get_best_documents_for_llm(question, collection, bi_encoder, reranker)

        if not best_docs:
            print("\n🤖 Réponse : Les documents fournis ne contiennent pas cette information.")
            continue

        # =====================================================================
        # 🛠️ AJOUT DU DEBUG ICI : Affichage des 3 documents finaux
        # =====================================================================
        print("\n" + "="*80)
        print("🛠️ DEBUG : TOP 3 DOCUMENTS ENVOYÉS AU LLM")
        print("="*80)
        for i, doc in enumerate(best_docs):
            meta = doc.get('meta', {})
            source = meta.get('source_file', 'Inconnu')
            method = meta.get('chunking_method', 'N/A')
            score = doc.get('rerank_score', 'N/A') # Si le score est dispo
            
            # On formate l'affichage pour le terminal
            if isinstance(score, float):
                print(f"🥇 DOC [{i+1}] | Score Cross-Encoder: {score:.4f} | Source: {source} ({method})")
            else:
                print(f"🥇 DOC [{i+1}] | Source: {source} ({method})")
                
            print("-" * 80)
            # On affiche les 400 premiers caractères pour ne pas inonder le terminal
            texte = doc.get('text', '')
            print(texte)
            print("\n")
        print("="*80 + "\n")
        # =====================================================================

        # ÉTAPE 3 : Génération de la réponse
        print("🤖 Analyse juridique en cours par Mistral...")
        reponse_llm = generate_legal_response(question, best_docs)

        end_time = time.time()

        # ÉTAPE 4 : Affichage du résultat final
        print("\n" + "="*80)
        print("⚖️ RÉPONSE DU BOT JURIDIQUE :")
        print("="*80)
        print(reponse_llm)
        print("\n" + "-"*80)
        print(f"⏱️ Temps total de traitement : {end_time - start_time:.2f} secondes")
        print("-"*80)


if __name__ == "__main__":
    main()
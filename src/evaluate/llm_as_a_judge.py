import os
import sys
import json
import time
from ollama import Client
import torch
import gc

# --- CONFIGURATION ---
OLLAMA_HOST = "http://127.0.0.1:11434"
JUDGE_MODEL = "mistral:latest"
DATASET_PATH = "../../data/golden_dataset/golden_dataset.json"

client = Client(host=OLLAMA_HOST)

# ==============================================================================
# 🔐 CONFIGURATION DES CHEMINS & IMPORTS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

# 🔥 LA SOLUTION EST ICI : On ajoute 'src', mais aussi 'generate' et 'rerank' 
# au chemin Python pour que les imports locaux à l'intérieur de ces dossiers fonctionnent.
sys.path.append(src_dir) 
sys.path.append(os.path.join(src_dir, "generate"))
sys.path.append(os.path.join(src_dir, "rerank"))

# 1. Imports de vos modules RAG
try:
    from rerank.rerank import get_best_documents_for_llm
except ImportError:
    from rerank import get_best_documents_for_llm

try:
    from generate.query_parse import rewrite_query
except ImportError:
    from query_parse import rewrite_query

# 2. Import des fonctions de génération depuis llm_generate.py
try:
    from generate.llm_generate import init_rag_pipeline, generate_legal_response
except ImportError:
    from llm_generate import init_rag_pipeline, generate_legal_response


# ==============================================================================
# 🧠 PROMPT DU LLM JUGE
# ==============================================================================
JUDGE_SYSTEM_PROMPT = """Tu es un juge impartial et expert en évaluation de modèles d'Intelligence Artificielle.
Ton rôle est d'évaluer la prédiction d'un modèle IA face à une question, en la comparant à une "Réponse de Référence" (Ground Truth).

Tu dois évaluer la prédiction selon deux critères stricts :

CRITÈRE 1 : EXACTITUDE FACTUELLE (Score : 0, 1, ou 2)
- 2 (Correct) : La prédiction contient toutes les informations factuelles de la réponse de référence. Les montants, chiffres et faits sont exacts.
- 1 (Partiel) : La prédiction contient une partie des informations, mais manque de précision ou oublie un détail important.
- 0 (Faux) : La prédiction est fausse, contredit la référence, ou est hors sujet.

CRITÈRE 2 : RESPECT DU FORMATAGE (Score : 0 ou 1)
- 1 (Succès) : La prédiction est DIRECTE. Elle ne contient AUCUNE phrase d'introduction (ex: "Voici la réponse", "Selon le document", "Il est indiqué que").
- 0 (Échec) : La prédiction contient du blabla conversationnel, une introduction ou une conclusion inutile.

INSTRUCTIONS DE SORTIE :
Tu dois obligatoirement répondre au format JSON exact suivant, sans aucun autre texte :
{
  "exactitude": <int>,
  "formatage": <int>,
  "raisonnement": "<explication très courte de tes notes>"
}"""

def evaluate_response(question, ground_truth, prediction):
    """Envoie les données au LLM Juge et récupère son score JSON."""
    
    user_prompt = f"""
[Question] : {question}
[Réponse de Référence] : {ground_truth}
[Prédiction à évaluer] : {prediction}

Évalue la prédiction en utilisant le format JSON demandé.
"""
    try:
        response = client.chat(
            model=JUDGE_MODEL,
            messages=[
                {'role': 'system', 'content': JUDGE_SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json',
            options={"temperature": 0.0} 
        )
        return json.loads(response['message']['content'])
    except Exception as e:
        print(f"Erreur d'évaluation : {e}")
        return {"exactitude": 0, "formatage": 0, "raisonnement": "Erreur API"}

# ==============================================================================
# 🚀 BOUCLE D'ÉVALUATION PRINCIPALE
# ==============================================================================
def main():
    print("⚙️ Initialisation du pipeline RAG pour l'évaluation...")
    collection, bi_encoder, reranker = init_rag_pipeline()
    
    # 1. Charger le Golden Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Fichier dataset introuvable : {DATASET_PATH}")
        sys.exit(1)
        
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # 🔥 LIMITATION DU DATASET À 100 ENTRÉES
    dataset = dataset[:100]
        
    print(f"\n🎯 Lancement de l'évaluation sur {len(dataset)} questions avec le juge '{JUDGE_MODEL}'...\n")

    total_exactitude = 0
    total_formatage = 0
    results = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        ground_truth = item["reponse"]
        
        # ---------------------------------------------------------
        # 🚧 ÉTAPE DE GÉNÉRATION 
        # ---------------------------------------------------------
        print(f"[{idx+1}/{len(dataset)}] Évaluation de la question : {question[:50]}...")
        
        # Optimisation de la question
        optimized_question = rewrite_query(question)
        
        # Récupération stricte 
        retrieved_docs = get_best_documents_for_llm(
            optimized_question, 
            collection, 
            bi_encoder, 
            reranker, 
            top_k_retrieve=8, 
            top_k_rerank=3
        )
        
        # Génération via votre script existant
        mistral_prediction = generate_legal_response(question, retrieved_docs) 
        
        # ---------------------------------------------------------
        # ⚖️ ÉTAPE DU JUGEMENT
        # ---------------------------------------------------------
        evaluation = evaluate_response(question, ground_truth, mistral_prediction)
        
        total_exactitude += evaluation.get("exactitude", 0)
        total_formatage += evaluation.get("formatage", 0)
        
        print(f"   ↳ Exactitude : {evaluation.get('exactitude')}/2 | Formatage : {evaluation.get('formatage')}/1")
        print(f"   ↳ Raisonnement : {evaluation.get('raisonnement')}\n")
        
        results.append({
            "question": question,
            "prediction": mistral_prediction,
            "evaluation": evaluation
        })
        
        time.sleep(0.5)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # ==============================================================================
    # 📊 CALCUL DES MÉTRIQUES FINALES
    # ==============================================================================
    max_exactitude = len(dataset) * 2
    max_formatage = len(dataset) * 1
    
    score_exactitude = (total_exactitude / max_exactitude) * 100 if max_exactitude > 0 else 0
    score_formatage = (total_formatage / max_formatage) * 100 if max_formatage > 0 else 0
    
    print("="*50)
    print("📈 RÉSULTATS GLOBAUX DE L'ÉVALUATION")
    print("="*50)
    print(f"Exactitude Factuelle : {score_exactitude:.1f}%")
    print(f"Respect du Formatage : {score_formatage:.1f}%")
    print("="*50)
    
    # Sauvegarde des résultats
    output_file = "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n💾 Résultats détaillés sauvegardés dans '{output_file}'")

if __name__ == "__main__":
    main()
import os
import sys
import time
from ollama import Client
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")

ollama_client = Client(host=OLLAMA_HOST)
def rewrite_query(user_query, model_name=LLM_MODEL):
    """
    Prend l'entrée brute de l'utilisateur et utilise le LLM pour la traduire en une 
    requête de recherche hautement optimisée, adaptée à la base de données vectorielle juridique.
    """
    system_prompt = """Tu es un expert en optimisation de recherche juridique pour le Journal Officiel algérien.
Ta SEULE tâche est de reformuler la requête de l'utilisateur pour l'optimiser pour la recherche dans une base de données vectorielle.

RÈGLES STRICTES :
1. Clarifie les phrases ambiguës et utilise la terminologie juridique algérienne exacte.
2. Ajoute des synonymes pertinents dans une formulation fluide.
3. FORMAT EXIGÉ : Tu dois générer UNE SEULE ET UNIQUE PHRASE. Aucun saut de ligne, aucune liste, aucune puce.

PORTE DE SORTIE (RÈGLE ABSOLUE) : 
Si l'entrée est une salutation (ex: "bonjour"), un test (ex: "test"), ou du charabia (ex: "blabla", "azerty"), renvoie UNIQUEMENT : SKIP_OPTIMIZATION

EXEMPLES DE COMPORTEMENT ATTENDU :
Entrée : "congé mat"
Sortie : Durée légale et conditions du congé de maternité pour les employées en Algérie

Entrée : "test"
Sortie : SKIP_OPTIMIZATION

Entrée : "loi sur le commerce"
Sortie : Législation et réglementation applicables aux sociétés commerciales et au droit des affaires en Algérie
"""

    user_prompt = f"""Reformule cette requête utilisateur pour la recherche en base de données :
{user_query}"""

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={
                "temperature": 0.0 # 0.0 empêche le modèle d'être "créatif" et le force à respecter les règles
            }
        )
        print(f"✅ Requête reformulée par le LLM : {response['message']['content']}")
        # .strip() supprime les sauts de ligne ou espaces accidentels ajoutés par le LLM
        return response['message']['content'].strip() 
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la reformulation : {e}. Utilisation de la requête originale.")
        return user_query # Solution de repli de sécurité : si Ollama échoue, on utilise le texte original

# def main():
#     print("  Outil de test : Optimisation de requêtes juridiques (Query Rewriting)")
#     print(f"  Modèle configuré : {LLM_MODEL}")
#     print(f"  Hôte Ollama : {OLLAMA_HOST}")
#     print("=" * 80)
#     print("  Astuce : Testez avec des phrases vagues, des fautes de frappe ou du langage familier.")
#     print("Tapez 'q' ou 'quitter' pour arrêter le test.\n")

#     while True:
#         # 1. Saisie utilisateur
#         user_input = input("  Entrez une question brute : ").strip()
        
#         # 2. Condition de sortie
#         if user_input.lower() in ['q', 'quitter']:
#             print("  Fin du test.")
#             break
            
#         if not user_input:
#             continue

#         print(" Reformulation en cours...")
        
#         # 3. Appel de la fonction de réécriture
#         optimized_query = rewrite_query(user_input)
        
#         # 4. Affichage du comparatif
#         print("-" * 80)
#         print(f" Originale : {user_input}")
#         print(f" Optimisée : {optimized_query}")
#         print("=" * 80 + "\n")

# if __name__ == "__main__":
#     main()
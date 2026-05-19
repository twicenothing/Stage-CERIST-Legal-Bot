import json
import os

# --- CONFIGURATION ---
GOLDEN_DATASET_PATH = "../../data/golden_dataset/golden_dataset.json" # Ajustez si besoin
FAILURES_PATH = "diagnostics_echecs.json"
OUTPUT_PATH = "analyse_detaillee_echecs.json"

def main():
    # 1. Vérification des fichiers
    if not os.path.exists(GOLDEN_DATASET_PATH):
        print(f"❌ Fichier Golden Dataset introuvable : {GOLDEN_DATASET_PATH}")
        return
    if not os.path.exists(FAILURES_PATH):
        print(f"❌ Fichier des échecs introuvable : {FAILURES_PATH}")
        return

    # 2. Charger le Golden Dataset et créer un dictionnaire de correspondance rapide
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
    
    # Création d'un dictionnaire { "question": "reponse attendue" }
    ground_truth_map = {item["question"]: item["reponse"] for item in golden_data}

    # 3. Charger les échecs
    with open(FAILURES_PATH, "r", encoding="utf-8") as f:
        failures_data = json.load(f)

    merged_results = {}

    # 4. Parcourir les erreurs (exactitude et formatage) et fusionner les données
    categories_erreurs = ["erreurs_exactitude", "erreurs_formatage"]
    
    for categorie in categories_erreurs:
        for item in failures_data.get(categorie, []):
            question = item["question"]
            
            # On utilise un dictionnaire pour éviter les doublons 
            # (si une question a raté l'exactitude ET le formatage)
            if question not in merged_results:
                merged_results[question] = {
                    "question": question,
                    "reponse_de_reference": ground_truth_map.get(question, "Référence non trouvée dans le Golden Dataset"),
                    "prediction_du_modele": item["prediction"],
                    "notes": {
                        "exactitude": item["evaluation"].get("exactitude"),
                        "formatage": item["evaluation"].get("formatage")
                    },
                    "raisonnement_du_juge": item["evaluation"].get("raisonnement")
                }

    # 5. Convertir le dictionnaire en liste pour le format JSON
    final_list = list(merged_results.values())

    # 6. Sauvegarder le résultat
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_list, f, indent=4, ensure_ascii=False)

    print("="*60)
    print("✅ FUSION TERMINÉE AVEC SUCCÈS")
    print("="*60)
    print(f"Total des questions en échec traitées : {len(final_list)}")
    print(f"💾 Fichier généré : '{OUTPUT_PATH}'")
    print("\nOuvrez ce fichier : vous aurez la prédiction, la vraie réponse et l'avis du juge au même endroit !")

if __name__ == "__main__":
    main()
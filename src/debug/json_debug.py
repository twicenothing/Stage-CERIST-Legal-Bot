import os
import json
import re

# --- CONFIGURATION ---
# Chemin relatif vers ton dossier JSON (depuis le dossier src/debug)
# Si tu lances le script depuis la racine du projet, adapte le chemin.
JSON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/json_llm_extracted"))

SEARCH_TERM = "24-440"  # Le numéro du décret maudit

def search_in_json(data, filename, path="root"):
    """
    Fonction récursive pour chercher dans des structures JSON imbriquées (Dict ou List).
    """
    found = False
    
    if isinstance(data, dict):
        for key, value in data.items():
            # Vérifie si le terme est dans la valeur (si c'est une string)
            if isinstance(value, str) and SEARCH_TERM in value:
                print(f"✅ TROUVÉ dans {filename}")
                print(f"   📂 Chemin : {path} -> {key}")
                print(f"   📄 Extrait : {value[:200]}...") # Affiche le début
                print("-" * 40)
                found = True
            # Appel récursif
            elif isinstance(value, (dict, list)):
                if search_in_json(value, filename, path=f"{path} -> {key}"):
                    found = True
                    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            # Appel récursif pour chaque item de la liste
            if search_in_json(item, filename, path=f"{path}[{i}]"):
                found = True
                
    return found

def main():
    print(f"🕵️‍♂️ Recherche de '{SEARCH_TERM}' dans : {JSON_DIR}\n")
    
    if not os.path.exists(JSON_DIR):
        print(f"❌ ERREUR : Le dossier {JSON_DIR} n'existe pas !")
        return

    files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]
    total_found = 0

    for filename in files:
        file_path = os.path.join(JSON_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            # On lance la recherche récursive
            if search_in_json(content, filename):
                total_found += 1
                
        except json.JSONDecodeError:
            print(f"⚠️  Erreur de lecture JSON sur {filename}")
        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

    print("\n" + "="*40)
    if total_found > 0:
        print(f"🎉 VICTOIRE : Le terme a été trouvé dans {total_found} fichier(s).")
        print("-> Le problème vient donc de l'indexation (Embeddings/Chroma).")
    else:
        print("😱 ÉCHEC : Le terme est introuvable dans les JSONs.")
        print("-> Le problème vient de l'extraction LLM (llm_structure.py) qui a 'oublié' ce passage.")
    print("="*40)

if __name__ == "__main__":
    main()
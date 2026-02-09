import os
import re

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Entrée : Le dossier où sont tes TXT extraits avec pypdf
INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/txt"))
# Sortie : Le dossier propre
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/cleaned"))

def remove_sommaire_pages(text):
    """
    Découpe le texte par pages (basé sur tes marqueurs '--- Page X ---')
    et supprime les blocs contenant le mot 'SOMMAIRE'.
    """
    # 1. On découpe le fichier en gardant le séparateur (le marqueur de page)
    # Le regex capture le marqueur : (--- Page \d+ ---)
    parts = re.split(r'(--- Page \d+ ---)', text)
    
    cleaned_parts = []
    
    # parts[0] est souvent vide (avant la première page), on le garde si besoin
    if parts[0].strip():
        cleaned_parts.append(parts[0])

    # 2. On boucle par pas de 2 car le split donne : [Marqueur, Contenu, Marqueur, Contenu...]
    # On commence à 1 car l'index 0 est le début du fichier (souvent vide)
    for i in range(1, len(parts), 2):
        marker = parts[i]       # Ex: "--- Page 2 ---"
        content = parts[i+1]    # Ex: "\n\nTexte de la page..."
        
        # 3. La condition unique : Si "SOMMAIRE" est dans le contenu, on zappe tout le bloc
        if "SOMMAIRE" in content:
            print(f"   🗑️  Sommaire détecté et supprimé : {marker}")
            continue # On passe à la page suivante sans rien ajouter
        
        # Sinon, on garde le marqueur et le contenu
        cleaned_parts.append(marker)
        cleaned_parts.append(content)
        
    return "".join(cleaned_parts)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(INPUT_DIR):
        print(f"❌ Erreur : Le dossier d'entrée n'existe pas : {INPUT_DIR}")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"🧹 Nettoyage 'Sommaire' sur {len(files)} fichiers...")
    print(f"📂 Entrée : {INPUT_DIR}")
    print(f"📂 Sortie : {OUTPUT_DIR}\n")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            
            # Action unique : Supprimer les pages Sommaire
            final_content = remove_sommaire_pages(raw_content)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_content)
                
            print(f"✅ Nettoyé : {filename}")

        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

if __name__ == "__main__":
    main()
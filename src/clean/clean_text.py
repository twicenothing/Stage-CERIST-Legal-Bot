import re
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "data", "txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cleaned")

def remove_arabic(text):
    """Supprime les caractères arabes."""
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+'
    return re.sub(arabic_pattern, '', text)

def is_header_noise(line):
    """
    Détecte les en-têtes à supprimer avec des Regex génériques.
    """
    line = line.strip()
    
    # 1. Liste des mois français pour la date Grégorienne
    mois_gregoriens = r"(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    
    noise_patterns = [
        # Titre du journal (classique)
        r"JOURNAL OFFICIEL DE LA REPUBLIQUE",
        r"Imprimerie Officielle",
        
        # DATE HÉGIRIENNE (Générique)
        # Ex: "12 Chaâbane 1446" ou "1 Rabie Ethani 1445"
        # Logique: Un chiffre + du texte (le mois) + une année 14xx
        r"^\d{1,2}\s+[a-zA-Z\s\u00C0-\u017F]+\s+14\d{2}$",

        # DATE GRÉGORIENNE (Générique)
        # Ex: "11 février 2025"
        # Logique: Un chiffre + un des 12 mois + une année 20xx
        rf"^\d{{1,2}}\s+{mois_gregoriens}\s+20\d{{2}}$",
        
        # Numéro de page seul (ex: "5" ou "33")
        r"^\d+$"
    ]
    
    for p in noise_patterns:
        if re.search(p, line, re.IGNORECASE):
            return True
            
    return False

def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()

    # 1. Découpage par page (grâce aux marqueurs de l'étape précédente)
    page_chunks = raw_content.split('==== PAGE')

    cleaned_content = []

    # On ignore la page 1 (index 0=vide, index 1=Page 1)
    if len(page_chunks) < 2:
        print(f"   ⚠️  Fichier {filename} trop court.")
        return

    # On commence à la page 2 (index 2)
    for chunk in page_chunks[2:]: 
        try:
            # On récupère le numéro de page pour référence
            # chunk ressemble à : " 2 ====\nTexte..."
            header_part, text_part = chunk.split('====', 1)
            page_num = header_part.strip()
            
            # On insère un marqueur propre
            cleaned_content.append(f"\n\n[[PAGE_REF:{page_num}]]\n")
            
        except ValueError:
            text_part = chunk

        # 2. Nettoyage ligne par ligne
        lines = text_part.split('\n')
        clean_lines = []

        for line in lines:
            line = line.strip()
            if not line: continue

            # Suppression Arabe
            line = remove_arabic(line)
            if not line.strip(): continue

            # Suppression En-têtes (Noise)
            if is_header_noise(line):
                continue

            clean_lines.append(line)

        # 3. Recollage (Stitching) pour gérer les colonnes
        page_text = " ".join(clean_lines)
        cleaned_content.append(page_text)

    # Sauvegarde
    final_text = "".join(cleaned_content)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    
    if not files:
        print("❌ Aucun fichier .txt trouvé.")
        return

    print(f"🧹 Nettoyage de {len(files)} fichiers...")
    
    for file in files:
        process_file(file)
        print(f"   ✅ Nettoyé : {file}")

if __name__ == "__main__":
    main()
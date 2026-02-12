import fitz  # PyMuPDF
import os
from tqdm import tqdm

# --- CONFIGURATION ---
PDF_DIR = "../../data/pdfs"            # Tes PDFs originaux
TXT_OUTPUT_DIR = "../../data/raw_text" # Les TXT nettoyés

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = []
    
    for i, page in enumerate(doc):
        # 1. Extraction avec tri intelligent (Colonnes)
        text = page.get_text("text", sort=True)
        
        # 2. FILTRE ANTI-SOMMAIRE 🚫
        # Dans le JO Algérien, le mot "SOMMAIRE" est souvent en haut de page.
        # On vérifie s'il est présent.
        if "SOMMAIRE" in text:
            # On affiche un petit message pour confirmer qu'on a bien sauté la page
            # (Utilise print conditionnel pour ne pas spammer si tu veux)
            # print(f"   -> Page {i+1} ignorée (Contient 'SOMMAIRE')")
            continue

        # 3. Nettoyage basique des en-têtes/pieds de page répétitifs
        # (Optionnel : enlève "JOURNAL OFFICIEL" si ça se répète trop)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # On ignore les lignes trop courtes ou purement décoratives
            if len(line.strip()) > 3: 
                cleaned_lines.append(line)
        
        full_text.append("\n".join(cleaned_lines))
        
    return "\n\n".join(full_text)

def main():
    # Vérifications des dossiers
    if not os.path.exists(PDF_DIR):
        print(f"❌ Erreur : Dossier {PDF_DIR} introuvable.")
        return
    if not os.path.exists(TXT_OUTPUT_DIR):
        os.makedirs(TXT_OUTPUT_DIR)

    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"🧹 Démarrage du nettoyage sur {len(files)} fichiers (Pages 'SOMMAIRE' exclues)...")

    count_success = 0
    for filename in tqdm(files):
        pdf_path = os.path.join(PDF_DIR, filename)
        txt_filename = filename.replace(".pdf", ".txt").replace(".PDF", ".txt")
        txt_path = os.path.join(TXT_OUTPUT_DIR, txt_filename)
        
        try:
            clean_content = extract_text_from_pdf(pdf_path)
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(clean_content)
            count_success += 1
                
        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

    print(f"\n✨ Terminé ! {count_success} fichiers traités.")
    print("👉 IMPORTANT : N'oublie pas de relancer 'safe_chunker.py' puis 'indexer.py' !")

if __name__ == "__main__":
    main()
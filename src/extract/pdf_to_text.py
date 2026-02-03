import fitz  # PyMuPDF
import os

# --- CONFIGURATION ---
# On remonte d'un niveau (../) pour sortir de 'src/extract' et aller dans 'data'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
TXT_DIR = os.path.join(BASE_DIR, "data", "txt")

def extract_text_preserving_columns(pdf_path, output_path):
    """
    Extrait le texte page par page en respectant l'ordre des colonnes.
    """
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        print(f"📄 Traitement de : {os.path.basename(pdf_path)}")

        for i, page in enumerate(doc):
            # sort=True est CRUCIAL pour les documents à double colonne (JORADP)
            # Il force la lecture de haut en bas, colonne gauche puis colonne droite.
            text = page.get_text(sort=True)
            
            # On ajoute un marqueur de page (utile pour le débogage visuel, sera retiré au nettoyage)
            full_text.append(f"\n{'='*20} PAGE {i+1} {'='*20}\n")
            full_text.append(text)

        # Sauvegarde
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("".join(full_text))
            
        print(f"   ✅ Sauvegardé sous : {os.path.basename(output_path)}")

    except Exception as e:
        print(f"   ❌ Erreur sur {os.path.basename(pdf_path)} : {e}")

def main():
    # 1. Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)
        print(f"📁 Dossier créé : {TXT_DIR}")

    # 2. Lister les PDFs
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print(f"⚠️  Aucun fichier PDF trouvé dans {PDF_DIR}")
        return

    print(f"🚀 Démarrage de l'extraction pour {len(pdf_files)} fichier(s)...")

    # 3. Boucle de traitement
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        
        # On garde le même nom mais en .txt
        txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
        txt_path = os.path.join(TXT_DIR, txt_filename)
        
        extract_text_preserving_columns(pdf_path, txt_path)

    print("\n✨ Extraction terminée !")

if __name__ == "__main__":
    main()
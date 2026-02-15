import fitz  # PyMuPDF
import re
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
OUTPUT_TXT_DIR = os.path.join(BASE_DIR, "data", "txt")

def remove_arabic(text):
    """Supprime les caractères arabes via Regex"""
    return re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', '', text)

def is_sommaire_page(text_blocks):
    """Vérifie si la page est une page de sommaire"""
    if not text_blocks: return False
    # On regarde les premiers blocs de texte (5 premiers)
    header_text = " ".join([b[4] for b in text_blocks[:5]]).lower()
    return "sommaire" in header_text

def get_sorted_text_from_page(page):
    """
    Extrait le texte en respectant les colonnes (En-tête -> Gauche -> Droite).
    """
    blocks = page.get_text("blocks")
    
    page_width = page.rect.width
    mid_point = page_width / 2

    full_width_blocks = []
    left_col_blocks = []  
    right_col_blocks = []  

    for b in blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        
        text = text.strip()
        if not text:
            continue

        block_width = x1 - x0

        # Si le bloc fait plus de 75% de la page, c'est un en-tête (titre traversant)
        if block_width > (page_width * 0.75):
            full_width_blocks.append(b)
        # Sinon, tri gauche/droite
        elif x0 < mid_point:
            left_col_blocks.append(b)
        else:
            right_col_blocks.append(b)

    # --- TRI PAR POSITION VERTICALE (Haut vers Bas pour chaque colonne) ---
    full_width_blocks.sort(key=lambda b: b[1])
    left_col_blocks.sort(key=lambda b: b[1])
    right_col_blocks.sort(key=lambda b: b[1])

    # --- ASSEMBLAGE : En-tête -> Gauche -> Droite ---
    sorted_blocks = full_width_blocks + left_col_blocks + right_col_blocks
    
    final_text = ""
    for b in sorted_blocks:
        # b[4] est le contenu texte du bloc
        final_text += b[4] + "\n"
        
    return final_text

def main():
    if not os.path.exists(OUTPUT_TXT_DIR):
        os.makedirs(OUTPUT_TXT_DIR)

    files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    print(f"📦 Démarrage : Traitement de {len(files)} fichiers PDF...")

    for filename in files:
        pdf_path = os.path.join(PDF_DIR, filename)
        txt_filename = filename.replace(".pdf", ".txt").replace(".PDF", ".txt")
        txt_path = os.path.join(OUTPUT_TXT_DIR, txt_filename)
        
        full_doc_text = ""
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc):
                
                # 🛑 1. SKIP PREMIÈRE PAGE (Page de garde)
                # C'est la première instruction, elle bloque tout le reste pour la page 0
                if page_num == 0:
                    continue

                # 2. Récupérer les blocs pour vérifier le sommaire
                blocks = page.get_text("blocks")
                
                # 🛑 3. SKIP SOMMAIRE
                if is_sommaire_page(blocks):
                    print(f"   🚫 {filename} - Page {page_num+1} ignorée (Sommaire)")
                    continue

                # ✅ 4. Extraction Intelligente (Tri des colonnes)
                # On utilise ta fonction de tri ici
                page_text = get_sorted_text_from_page(page)
                
                # 5. Nettoyage Arabe
                page_text = remove_arabic(page_text)
                
                # Ajout au texte global
                full_doc_text += page_text + "\n\n"

            # Sauvegarde
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_doc_text)
                
            print(f"✅ Extrait : {filename}")

        except Exception as e:
            print(f"❌ Erreur sur {filename} : {e}")

    print("\n🚀 Extraction terminée !")

if __name__ == "__main__":
    main()
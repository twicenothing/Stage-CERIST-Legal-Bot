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

def is_ignored_title(text):
    """
    Vérifie si le texte (seul sur sa ligne/bloc) fait partie des titres à ignorer.
    """
    # Nettoyage : on met en minuscules et on normalise les espaces
    clean_text = " ".join(text.strip().lower().split())
    
    # Liste des titres stricts à supprimer (avec et sans accents)
    exact_matches = [
        "decisions et avis", "décisions et avis",
        "arretes", "arrêtés", "arrêtes",
        "arretes, decisions et avis", "arrêtés, décisions et avis",
        "conventions et accords internationaux",
        "decrets", "décrets",
        "decisions individuelles", "décisions individuelles",
        "annonces et communications"
    ]
    
    return clean_text in exact_matches

def get_sorted_text_from_page(page):
    """
    Extrait le texte en gérant les ruptures de colonnes (titres centrés).
    Dès qu'un titre centré est détecté, on traite les colonnes du dessus, 
    puis on recommence pour les colonnes du dessous.
    """
    blocks = page.get_text("blocks")
    page_width = page.rect.width
    mid_point = page_width / 2

    # 1. On garde uniquement le texte et on trie TOUT de haut en bas (axe Y)
    valid_blocks = []
    for b in blocks:
        if b[6] == 0 and b[4].strip(): # b[6] == 0 signifie que c'est du texte, pas une image
            valid_blocks.append(b)
            
    valid_blocks.sort(key=lambda b: b[1]) # b[1] est y0 (la position verticale)

    final_text = ""
    current_band_blocks = [] # Les blocs de la "section" en cours

    # Fonction interne pour traiter une section (lire sa gauche puis sa droite)
    def process_band(band_blocks):
        if not band_blocks: return ""
        left_col = []
        right_col = []
        for b in band_blocks:
            if b[0] < mid_point:
                left_col.append(b)
            else:
                right_col.append(b)
        
        # On trie verticalement chaque colonne
        left_col.sort(key=lambda b: b[1])
        right_col.sort(key=lambda b: b[1])
        
        band_text = ""
        # On ajoute toute la gauche, PUIS toute la droite de cette section
        for b in left_col + right_col:
            band_text += b[4] + "\n"
        return band_text

    # 2. On parcourt la page de haut en bas
    for b in valid_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        block_width = x1 - x0

        # Est-ce que ce bloc est un séparateur (Mur) ?
        is_separator = False
        if is_ignored_title(text):
            is_separator = True
        elif block_width > (page_width * 0.75):
            is_separator = True

        if is_separator:
            # ON A TOUCHÉ UN MUR !
            # On traite tout ce qu'on a accumulé au-dessus (Gauche -> Droite)
            final_text += process_band(current_band_blocks)
            current_band_blocks = [] # On vide la liste pour la section d'en dessous
            
            # Si le séparateur N'EST PAS dans notre liste d'ignorés, on l'ajoute au texte
            if not is_ignored_title(text):
                final_text += text + "\n\n"
        else:
            # C'est un bloc de colonne normal, on le met de côté
            current_band_blocks.append(b)

    # 3. Fin de la page : on traite la toute dernière section (en dessous du dernier mur)
    final_text += process_band(current_band_blocks)

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
                if page_num == 0:
                    continue

                # 2. Récupérer les blocs pour vérifier le sommaire
                blocks = page.get_text("blocks")
                
                # 🛑 3. SKIP SOMMAIRE
                if is_sommaire_page(blocks):
                    print(f"   🚫 {filename} - Page {page_num+1} ignorée (Sommaire)")
                    continue

                # ✅ 4. Extraction Intelligente (Tri des colonnes + Suppression des titres)
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
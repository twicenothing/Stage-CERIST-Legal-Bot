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
    
    # 1. Keep only valid text blocks (type 0) that are not empty
    valid_blocks = [b for b in text_blocks if b[6] == 0 and b[4].strip()]
    
    # 2. 🚨 THE FIX: Sort blocks strictly from TOP to BOTTOM based on their y0 coordinate
    valid_blocks.sort(key=lambda b: b[1])
    
    # 3. Now we can safely take the top 15 physical blocks
    header_text = " ".join([b[4] for b in valid_blocks[:15]]).lower()
    
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
    Extrait le texte en gérant les ruptures de colonnes (titres centrés)
    ET intègre les tableaux proprement au format Markdown.
    """
    page_width = page.rect.width
    mid_point = page_width / 2

    # --- 1. DÉTECTION ET FORMATAGE DES TABLEAUX ---
    tables = page.find_tables()
    table_bboxes = []
    table_blocks = []

    if tables.tables:
        for tab in tables.tables:
            # Récupération des coordonnées du tableau
            bbox = tab.bbox
            table_bboxes.append(bbox)
            
            # Construction du tableau en format texte (Markdown)
            md_table = "\n"
            rows = tab.extract()
            for i, row in enumerate(rows):
                # Nettoyage des cellules (retirer les retours à la ligne dans une même cellule)
                clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                md_table += "| " + " | ".join(clean_row) + " |\n"
                
                # Ajout de la ligne de séparation Markdown après l'en-tête
                if i == 0:
                    md_table += "|" + "|".join(["---"] * len(row)) + "|\n"
            md_table += "\n"
            
            # On crée un faux "bloc" pour l'injecter dans notre tri
            # (x0, y0, x1, y1, texte, block_no, block_type)
            # On lui donne la largeur max pour qu'il agisse comme un séparateur
            table_blocks.append((bbox[0], bbox[1], bbox[2], bbox[3], md_table, -1, 0))

    # --- 2. EXTRACTION DU TEXTE (Avec masquage des tableaux) ---
    raw_blocks = page.get_text("blocks")
    valid_blocks = []

    for b in raw_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        
        # On ignore les images et les blocs vides
        if block_type != 0 or not text.strip():
            continue

        # Calcul du centre du bloc pour vérifier s'il est dans un tableau
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        
        in_table = False
        for t_bbox in table_bboxes:
            tx0, ty0, tx1, ty1 = t_bbox
            # Si le centre du texte est dans les limites du tableau, on l'ignore
            if tx0 <= center_x <= tx1 and ty0 <= center_y <= ty1:
                in_table = True
                break
                
        if not in_table:
            valid_blocks.append(b)

    # --- 3. FUSION ET TRI VERTICAL GLOBAL ---
    # On mélange le texte normal avec nos blocs de tableaux
    all_blocks = valid_blocks + table_blocks
    all_blocks.sort(key=lambda b: b[1]) # Tri de haut en bas

    # --- 4. LOGIQUE DE BANDES (Colonnes + Murs) ---
    final_text = ""
    current_band_blocks = []

    def process_band(band_blocks):
        if not band_blocks: return ""
        left_col = []
        right_col = []
        for b in band_blocks:
            if b[0] < mid_point:
                left_col.append(b)
            else:
                right_col.append(b)
        
        left_col.sort(key=lambda b: b[1])
        right_col.sort(key=lambda b: b[1])
        
        band_text = ""
        for b in left_col + right_col:
            band_text += b[4] + "\n"
        return band_text

    for b in all_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        block_width = x1 - x0

        is_separator = False
        
        # Un tableau ou un titre ignoré agit comme un mur
        if is_ignored_title(text):
            is_separator = True
        elif block_width > (page_width * 0.75):
            is_separator = True
        # Si c'est notre tableau généré (on le reconnaît par sa syntaxe Markdown)
        elif text.startswith("\n|"): 
            is_separator = True

        if is_separator:
            # On vide la bande actuelle (au-dessus du séparateur)
            final_text += process_band(current_band_blocks)
            current_band_blocks = []
            
            # Si le séparateur N'EST PAS un titre ignoré, on l'ajoute !
            # (Donc, si c'est un tableau ou un vrai titre large, on l'écrit)
            if not is_ignored_title(text):
                final_text += text + "\n\n"
        else:
            current_band_blocks.append(b)

    # Fin de la page
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
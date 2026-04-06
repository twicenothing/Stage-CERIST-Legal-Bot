import fitz  # PyMuPDF
import re
import os

# --- CONFIGURATION ---
# Mettez vos dossiers (ex: 2002, 2003, 2004) dans ce dossier principal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_PDF_DIR = os.path.join(BASE_DIR, "data", "pdf_old")
OUTPUT_TXT_DIR =  os.path.join(BASE_DIR, "data", "txt")


def remove_arabic(text):
    """Supprime les caractères arabes via Regex"""
    return re.sub(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+', '', text)

def is_sommaire_page(text_blocks):
    if not text_blocks: return False
    valid_blocks = [b for b in text_blocks if b[6] == 0 and b[4].strip()]
    valid_blocks.sort(key=lambda b: b[1])
    header_text = " ".join([b[4] for b in valid_blocks[:15]]).lower()
    header_text_clean = re.sub(r'\s+', '', header_text)
    return "sommaire" in header_text_clean

def is_ignored_title(text):
    """
    Vérifie si le texte est un titre à ignorer.
    Gère les lettres espacées comme 'D E C R E T S' en supprimant tous les espaces.
    """
    clean_text = text.strip().lower()
    # On supprime absolument TOUS les espaces
    clean_text_no_spaces = re.sub(r'\s+', '', clean_text)
    
    # Liste sans aucun espace
    exact_matches_no_spaces = [
        "decisionsetavis", "décisionsetavis",
        "arretes", "arrêtés", "arrêtes",
        "arretes,decisionsetavis", "arrêtés,décisionsetavis",
        "conventionsetaccordsinternationaux",
        "decrets", "décrets",
        "decisionsindividuelles", "décisionsindividuelles",
        "annoncesetcommunications",
        "reglements", "règlements","lois","proclamations",
        "reglementsinterieurs", "règlementsintérieurs",
        "arretesetproclamations", "arrêtésétproclamations",
        "proclamationsetdecisions", "proclamationsetdécisions", "avis","avisetlois",
        "ordonnances","instructionspresidentielles", "instructionsprésidentielles"
    ]
    
    return clean_text_no_spaces in exact_matches_no_spaces

def get_text_2003_2004(page):
    """
    Méthode spéciale pour 2003-2004 :
    Tri géométrique strict + Détection de tableaux avec Bouclier Anti-Hallucination.
    """
    page_width = page.rect.width
    page_height = page.rect.height
    mid_point = page_width / 2

    # --- 1. DÉTECTION INTELLIGENTE DES TABLEAUX ---
    tables = page.find_tables()
    table_bboxes = []
    table_blocks = []

    if tables.tables:
        for tab in tables.tables:
            bbox = tab.bbox
            table_height = bbox[3] - bbox[1]
            col_count = len(tab.header.names) if tab.header else 0
            
            # 🛡️ LE BOUCLIER ANTI-HALLUCINATION
            if table_height > (page_height * 0.5) and col_count <= 2:
                continue
                
            table_bboxes.append(bbox)
            md_table = "\n"
            rows = tab.extract()
            for i, row in enumerate(rows):
                clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                md_table += "| " + " | ".join(clean_row) + " |\n"
                if i == 0:
                    md_table += "|" + "|".join(["---"] * len(row)) + "|\n"
            md_table += "\n"
            
            table_blocks.append((bbox[0], bbox[1], bbox[2], bbox[3], md_table, -1, 0))

    # --- 2. EXTRACTION DU TEXTE ---
    raw_blocks = page.get_text("blocks")
    valid_blocks = []

    for b in raw_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        
        if block_type != 0 or not text.strip():
            continue
            
        # Ignorer l'en-tête du journal
        clean_t = text.lower()
        if y0 < 80 and ("journal officiel" in clean_t or "republique algerienne" in clean_t or "république algérienne" in clean_t):
            continue

        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        in_table = False
        for t_bbox in table_bboxes:
            tx0, ty0, tx1, ty1 = t_bbox
            if tx0 <= center_x <= tx1 and ty0 <= center_y <= ty1:
                in_table = True
                break
                
        if not in_table:
            valid_blocks.append(b)

    # --- 3. FUSION ET TRI VERTICAL ---
    all_blocks = valid_blocks + table_blocks
    all_blocks.sort(key=lambda b: b[1])

    # --- 4. LOGIQUE DE BANDES ---
    final_text = ""
    current_band_blocks = []

    def process_band(band_blocks):
        if not band_blocks: return ""
        left_col = []
        right_col = []
        for b in band_blocks:
            center_x = (b[0] + b[2]) / 2 
            if center_x < mid_point:
                left_col.append(b)
            else:
                right_col.append(b)
                
        left_col.sort(key=lambda b: b[1])
        right_col.sort(key=lambda b: b[1])
        
        band_text = ""
        for b in left_col: band_text += b[4].strip() + "\n\n"
        for b in right_col: band_text += b[4].strip() + "\n\n"
        return band_text

    for b in all_blocks:
        x0, y0, x1, y1, text, block_no, block_type = b
        block_width = x1 - x0
        is_separator = False
        
        # Un tableau Markdown est toujours un séparateur
        if block_type == -1:
            is_separator = True
        elif is_ignored_title(text):
            is_separator = True
        
        # 🚨 LA CORRECTION EST ICI 🚨
        elif block_width > (page_width * 0.40):
            # Si le bloc déborde sur les deux moitiés de la page (gauche et droite du centre),
            # c'est physiquement impossible que ce soit une colonne. C'est donc un mur !
            if x0 < (mid_point - 15) and x1 > (mid_point + 15):
                is_separator = True
                
        else:
            center_x = (x0 + x1) / 2
            if abs(center_x - mid_point) < (page_width * 0.1) and block_width < (page_width * 0.5):
                 if "——" in text or "ETAT ANNEXE" in text.upper():
                     is_separator = True

        if is_separator:
            final_text += process_band(current_band_blocks)
            current_band_blocks = []
            if not is_ignored_title(text):
                final_text += text.strip() + "\n\n"
        else:
            current_band_blocks.append(b)

    final_text += process_band(current_band_blocks)
    return final_text

def main():
    if not os.path.exists(OUTPUT_TXT_DIR):
        os.makedirs(OUTPUT_TXT_DIR)

    # Vérifie si le dossier de base existe
    if not os.path.exists(BASE_PDF_DIR):
        print(f"❌ Erreur: Le dossier '{BASE_PDF_DIR}' n'existe pas.")
        print(f"Veuillez créer un dossier '{BASE_PDF_DIR}' et y placer vos dossiers d'années (ex: 2002, 2003, 2004...).")
        return

    # Récupère tous les sous-dossiers dans BASE_PDF_DIR
    folders = [f for f in os.listdir(BASE_PDF_DIR) if os.path.isdir(os.path.join(BASE_PDF_DIR, f))]
    
    if not folders:
        print(f"⚠️ Aucun sous-dossier trouvé dans '{BASE_PDF_DIR}'.")
        return

    print(f"📦 Démarrage : Traitement par lots sur {len(folders)} dossiers d'années...\n")

    for folder_name in sorted(folders):
        folder_path = os.path.join(BASE_PDF_DIR, folder_name)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        
        print(f"📂 Ouverture du dossier: {folder_name} | Fichiers: {len(files)}")

        for filename in files:
            pdf_path = os.path.join(folder_path, filename)
            txt_filename = filename.replace(".pdf", ".txt").replace(".PDF", ".txt")
            txt_path = os.path.join(OUTPUT_TXT_DIR, txt_filename)
            
            full_doc_text = ""
            
            try:
                doc = fitz.open(pdf_path)
                
                for page_num, page in enumerate(doc):
                    
                    # 🚨 CAS SPÉCIAL : F2004004.pdf
                    if filename == "F2004004.pdf":
                        # On saute manuellement les 2 premières pages (Page de garde + Sommaire)
                        if page_num < 2:
                            continue
                        # On ne passe PAS par is_sommaire_page pour ce fichier,
                        # ce qui protège le sommaire interne du cahier des charges.
                        
                    # 🛡️ RÈGLE GÉNÉRALE : Pour tous les autres fichiers
                    else:
                        # 1. Ignorer la première page
                        if page_num == 0:
                            continue

                        # 2. Ignorer les pages de sommaire
                        blocks = page.get_text("blocks")
                        if is_sommaire_page(blocks):
                            print(f"   🚫 {filename} - Page {page_num+1} ignorée (Sommaire)")
                            continue

                    # ✅ 3. Extraction Spéciale 2003-2004
                    page_text = get_text_2003_2004(page)
                    
                    # ✅ 4. Nettoyage Arabe
                    page_text = remove_arabic(page_text)
                    
                    full_doc_text += page_text + "\n"

                # Sauvegarde
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_doc_text)
                    
                print(f"  ✅ Extrait : {filename}")

            except Exception as e:
                print(f"  ❌ Erreur sur {filename} : {e}")

    print("\n🚀 Extraction de tous les dossiers terminée !")

if __name__ == "__main__":
    main()
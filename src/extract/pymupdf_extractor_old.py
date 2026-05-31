import fitz  # PyMuPDF
import re
import os
import json

# --- CONFIGURATION ---
# Mettez vos dossiers (ex: 2002, 2003, 2004) dans ce dossier principal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_PDF_DIR = os.path.join(BASE_DIR, "data", "pdf_old")
OUTPUT_TXT_DIR =  os.path.join(BASE_DIR, "data", "txt")
OUTPUT_TABLES_DIR = os.path.join(BASE_DIR, "data", "tables")
OUTPUT_TABLE_CHUNKS_DIR = os.path.join(BASE_DIR, "data", "table_chunks")


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


# --- TABLE SIDE-CAR EXTRACTION HELPERS ---
# These helpers do not change the existing text extraction logic.
# They only create extra JSON files for tables and table-row chunks.

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def one_line(text):
    return re.sub(r'\s+', ' ', str(text or '')).strip()


def clean_table_cell(cell):
    if cell is None:
        return ""
    value = str(cell).replace('\r', '\n')
    value = remove_arabic(value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def make_unique_headers(header_row, max_cols):
    """Create stable, non-empty, unique headers for row-level JSON chunks."""
    headers = []
    seen = {}
    for i in range(max_cols):
        raw = header_row[i] if i < len(header_row) else ""
        h = one_line(raw)
        if not h:
            h = f"COL_{i+1}"
        # Keep headers readable but short enough for metadata / chunks.
        h = h[:120]
        key = h.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            h = f"{h}_{seen[key]}"
        headers.append(h)
    return headers


def table_to_markdown(rows):
    """Same Markdown style as the original script: keep table inside TXT flow."""
    if not rows:
        return "\n"
    max_cols = max(len(r) for r in rows)
    normalized_rows = [list(r) + [""] * (max_cols - len(r)) for r in rows]

    md_table = "\n"
    for i, row in enumerate(normalized_rows):
        md_table += "| " + " | ".join(one_line(c) for c in row) + " |\n"
        if i == 0:
            md_table += "|" + "|".join(["---"] * max_cols) + "|\n"
    md_table += "\n"
    return md_table


def find_table_caption(page, bbox):
    """Light caption extraction: nearby text immediately above the table."""
    x0, y0, x1, y1 = bbox
    candidates = []
    try:
        for b in page.get_text("blocks"):
            bx0, by0, bx1, by1, text, block_no, block_type = b
            if block_type != 0 or not str(text).strip():
                continue
            # Only take text slightly above the table, not the whole page header.
            if by1 < y0 and by1 > max(0, y0 - 110):
                t = one_line(remove_arabic(text))
                if not t:
                    continue
                low = t.lower()
                if "journal officiel" in low or "republique algerienne" in low or "république algérienne" in low:
                    continue
                candidates.append((by0, t))
    except Exception:
        return ""
    candidates.sort(key=lambda x: x[0])
    return one_line(" ".join(t for _, t in candidates))


def detect_table_kind(headers, rows, caption=""):
    joined = one_line(" ".join(headers) + " " + caption).upper()
    if any(k in joined for k in ["POSITION", "SOUS-POSITION", "TARIFAIRE", "TAUX", "D'ORDRE", "D’ORDRE"]):
        return "tariff_table"
    if any(k in joined for k in ["NOMS", "PRENOMS", "PRÉNOMS", "ORGANISME", "WILAYAS"]):
        return "names_list_table"
    if any(k in joined for k in ["EFFECTIF", "ETABLISSEMENTS", "ÉTABLISSEMENTS", "CATEGORIE", "CATÉGORIE", "POINT INDICIAIRE"]):
        return "staffing_annex_table"
    return "generic_table"


def row_to_text_generic(source_file, page_num, table_id, caption, headers, row, row_index):
    bits = [
        f"Source: {source_file}",
        f"Page: {page_num}",
        f"Tableau: {caption or table_id}",
        f"Ligne: {row_index}",
    ]
    for h, v in zip(headers, row):
        h = one_line(h)
        v = one_line(v)
        if v:
            bits.append(f"{h}: {v}")
    return "\n".join(bits)


def build_table_sidecar(page, tab, source_file, page_num, table_index):
    """
    Extract one table into:
      1) Markdown for the original TXT flow
      2) structured table JSON
      3) full-table + row-level chunks for embedding
    """
    stem = os.path.splitext(os.path.basename(source_file))[0]
    table_id = f"{stem}_p{page_num}_t{table_index}"
    bbox = tuple(float(v) for v in tab.bbox)

    raw_rows = tab.extract()
    clean_rows = []
    for row in raw_rows:
        clean_row = [clean_table_cell(cell) for cell in row]
        if any(clean_row):
            clean_rows.append(clean_row)

    md_table = table_to_markdown(clean_rows)
    if not clean_rows:
        return md_table, None, []

    max_cols = max(len(r) for r in clean_rows)
    clean_rows = [list(r) + [""] * (max_cols - len(r)) for r in clean_rows]
    headers = make_unique_headers(clean_rows[0], max_cols)
    caption = find_table_caption(page, bbox)
    table_kind = detect_table_kind(headers, clean_rows, caption)

    row_records = []
    for idx, row in enumerate(clean_rows[1:], start=1):
        if not any(one_line(c) for c in row):
            continue
        row_records.append({
            "row_index": idx,
            "cells": {headers[i]: one_line(row[i]) for i in range(max_cols)},
        })

    table_obj = {
        "table_id": table_id,
        "source_file": os.path.basename(source_file),
        "page": page_num,
        "bbox": list(bbox),
        "caption": caption,
        "headers": headers,
        "rows": row_records,
        "raw_rows": clean_rows,
        "markdown": md_table,
        "table_kind": table_kind,
        "extractor": "pymupdf.find_tables",
    }

    chunks = []
    # Full-table chunk: useful for broad questions like "what does this annex contain?"
    chunks.append({
        "id": f"{table_id}_full",
        "text": f"Source: {os.path.basename(source_file)}\nPage: {page_num}\nTableau: {caption or table_id}\n{md_table.strip()}",
        "metadata": {
            "source_file": os.path.basename(source_file),
            "page": page_num,
            "table_id": table_id,
            "table_kind": table_kind,
            "chunking_method": "table_full",
            "chunk_format": "full_table_markdown",
        }
    })

    # Row chunks: useful for exact lookup inside tables.
    for rec in row_records:
        row_index = rec["row_index"]
        row = [rec["cells"].get(h, "") for h in headers]
        chunks.append({
            "id": f"{table_id}_row_{row_index}",
            "text": row_to_text_generic(os.path.basename(source_file), page_num, table_id, caption, headers, row, row_index),
            "metadata": {
                "source_file": os.path.basename(source_file),
                "page": page_num,
                "table_id": table_id,
                "table_kind": table_kind,
                "row_index": row_index,
                "chunking_method": "table_row",
                "chunk_format": "table_row",
            }
        })

    return md_table, table_obj, chunks


def save_table_outputs(source_file, table_records, table_chunks):
    stem = os.path.splitext(os.path.basename(source_file))[0]
    tables_path = os.path.join(OUTPUT_TABLES_DIR, stem + "_tables.json")
    chunks_path = os.path.join(OUTPUT_TABLE_CHUNKS_DIR, stem + "_table_chunks.json")

    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": os.path.basename(source_file),
            "tables_total": len(table_records),
            "tables": table_records,
        }, f, ensure_ascii=False, indent=2)

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": os.path.basename(source_file),
            "chunks_total": len(table_chunks),
            "chunks": table_chunks,
        }, f, ensure_ascii=False, indent=2)

    return tables_path, chunks_path


def get_text_2003_2004(page, source_file="", page_num=None, table_records=None, table_chunks=None):
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

            # Même comportement qu'avant pour le TXT: on injecte un tableau Markdown.
            # Nouveau comportement en plus: on sauvegarde une version structurée
            # et des chunks ligne-par-ligne pour l'embedder.
            actual_page_num = page_num if page_num is not None else 0
            table_index = len(table_bboxes)
            md_table, table_obj, chunks = build_table_sidecar(page, tab, source_file, actual_page_num, table_index)

            if table_records is not None and table_obj is not None:
                table_records.append(table_obj)
            if table_chunks is not None and chunks:
                table_chunks.extend(chunks)
            
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
    ensure_dir(OUTPUT_TABLES_DIR)
    ensure_dir(OUTPUT_TABLE_CHUNKS_DIR)

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
            pdf_table_records = []
            pdf_table_chunks = []
            
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
                    page_text = get_text_2003_2004(
                        page,
                        source_file=filename,
                        page_num=page_num + 1,
                        table_records=pdf_table_records,
                        table_chunks=pdf_table_chunks,
                    )
                    
                    # ✅ 4. Nettoyage Arabe
                    page_text = remove_arabic(page_text)
                    
                    full_doc_text += f"\n\n<<<PAGE_{page_num+1}>>>\n{page_text}\n\n"

                # Sauvegarde
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_doc_text)

                # Sauvegarde side-car des tableaux détectés
                save_table_outputs(filename, pdf_table_records, pdf_table_chunks)
                    
                print(f"  ✅ Extrait : {filename} | Tables: {len(pdf_table_records)} | Table chunks: {len(pdf_table_chunks)}")

            except Exception as e:
                print(f"  ❌ Erreur sur {filename} : {e}")

    print("\n🚀 Extraction de tous les dossiers terminée !")

if __name__ == "__main__":
    main()
import os
import re
import json

# --- CHEMINS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/cleaned"))
# On sauvegarde dans un dossier spécifique
JSON_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/json_arretes"))

# --- REGEX ---

# 1. Regex GLOBALE pour repérer TOUS les débuts de textes (Bornes de découpage)
# Capture : Décret, Arrêté ou Décision + Type optionnel + N° optionnel + "du" obligatoire
# Ex: "Arrêté du...", "Arrêté interministériel du...", "Décision n°... du..."
REGEX_ALL_STARTS = re.compile(
    r"((?:Décret|Arrêté|Décision)\s+(?:exécutif|présidentiel|interministériel|ministériel)?\s*(?:n°\s*[\d/-]+)?\s*du\s+)", 
    re.IGNORECASE
)

# 2. Regex spécifique pour identifier si c'est un Arrêté (pour le filtrage final)
REGEX_IS_ARRETE = re.compile(r"^Arrêté", re.IGNORECASE)

# 3. Metadata
REGEX_JOURNAL_DATE = re.compile(r"Correspondant\s+au\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)
REGEX_PAGE_MARKER = re.compile(r"--- Page (\d+) ---", re.IGNORECASE)

def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def get_journal_metadata(content):
    date_match = REGEX_JOURNAL_DATE.search(content[:5000])
    return date_match.group(1) if date_match else "Date Inconnue"

def find_page_number(content, start_index):
    preceding = content[:start_index]
    matches = list(REGEX_PAGE_MARKER.finditer(preceding))
    return int(matches[-1].group(1)) if matches else 1

def get_all_anchors(content):
    """
    Repère TOUS les débuts de textes législatifs pour définir les frontières.
    """
    anchors = []
    for match in REGEX_ALL_STARTS.finditer(content):
        start = match.start()
        # On capture un bout du titre pour l'ID (ex: "Arrêté interministériel du")
        # On étend la capture un peu après le "du" pour avoir la date si possible
        text_id_raw = content[start:match.end()+20].split('\n')[0] 
        text_id = normalize_text(text_id_raw)
        
        # --- FILTRE ANTI-CITATION ---
        context_before = content[max(0, start-50):start].lower()
        # Si précédé de "Vu l'", "Vu le", "par", "modifiant" -> Ignorer
        if any(x in context_before for x in ["vu le", "vu l’", "vu l'", "modifiant l", "complétant l"]):
            continue

        anchors.append({
            "start": start,
            "type_id": text_id,
            "is_arrete": bool(REGEX_IS_ARRETE.match(text_id))
        })
    return anchors

def process_file(filename, content):
    journal_date = get_journal_metadata(content)
    anchors = get_all_anchors(content)
    
    arretes_extracted = []
    
    for i in range(len(anchors)):
        current = anchors[i]
        
        # On ne traite que si c'est un ARRÊTÉ
        if not current['is_arrete']:
            continue
            
        start = current['start']
        
        # La fin est le début du PROCHAIN texte (peu importe si c'est un décret ou un arrêté)
        if i < len(anchors) - 1:
            end = anchors[i+1]['start']
        else:
            end = len(content)
            
        raw_block = content[start:end]
        
        # Nettoyage
        clean_block = re.sub(r"--- Page \d+ ---", "", raw_block)
        clean_block = normalize_text(clean_block)
        
        # Si le bloc est trop petit (erreur d'OCR ou titre seul), on jette
        if len(clean_block) < 50:
            continue
            
        page_num = find_page_number(content, start)
        
        # Extraction intelligente du Titre Complet (jusqu'au premier "Vu")
        # Les arrêtés commencent souvent par "Arrêté du [Date] fixant [Objet]." puis "Le Ministre..."
        # Ou directement "Le Ministre..."
        
        doc_object = {
            "source": filename,
            "journal_date": journal_date,
            "page_start": page_num,
            "doc_type": "Arrêté",
            "title_extract": current['type_id'], # Ex: "Arrêté du 12 janvier 2024"
            "text": clean_block
        }
        
        arretes_extracted.append(doc_object)

    return arretes_extracted

def main():
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)
        
    print(f"Extraction des Arrêtés depuis : {TXT_DIR}")
    
    count = 0
    for filename in os.listdir(TXT_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(TXT_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            results = process_file(filename, content)
            
            if results:
                out_name = filename.replace(".txt", ".jsonl")
                out_path = os.path.join(JSON_DIR, out_name)
                with open(out_path, "w", encoding="utf-8") as f_out:
                    for r in results:
                        f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"   > [OK] {len(results)} arrêtés dans {out_name}")
                count += len(results)
    
    print(f"\nTerminé. Total arrêtés extraits : {count}")

if __name__ == "__main__":
    main()
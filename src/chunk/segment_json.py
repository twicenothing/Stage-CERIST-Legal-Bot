import os
import re
import json

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# On prend les fichiers nettoyés et "recousus" (cleaner_stitch)
TXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/cleaned"))
JSON_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/json_whole"))

# --- REGEX ---

# 1. Capture le début d'un décret (Type + Numéro)
# Ex: Décret exécutif n° 25-59
REGEX_DECREE_START = re.compile(r"(Décret\s+(?:exécutif|présidentiel)\s+n°\s+[\d/-]+)", re.IGNORECASE)

# 2. Capture la date du journal (souvent en haut de la page 2 ou 3)
REGEX_JOURNAL_DATE = re.compile(r"Correspondant\s+au\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)

# 3. Capture les marqueurs de page (pour savoir où on est)
REGEX_PAGE_MARKER = re.compile(r"--- Page (\d+) ---", re.IGNORECASE)

def normalize_text(text):
    """Nettoie les espaces multiples."""
    return re.sub(r'\s+', ' ', text).strip()

def get_journal_metadata(content):
    """Extrait la date globale du journal."""
    date_match = REGEX_JOURNAL_DATE.search(content[:5000])
    return date_match.group(1) if date_match else "Date Inconnue"

def find_decree_anchors(content):
    """
    Trouve tous les index de début de décret.
    Filtre les citations (Vu le...) pour ne garder que les vrais titres.
    """
    anchors = []
    
    for match in REGEX_DECREE_START.finditer(content):
        start_index = match.start()
        text_id = normalize_text(match.group(1))
        
        # --- FILTRE ANTI-CITATION ---
        # On regarde les 50 caractères AVANT le match
        context_before = content[max(0, start_index-50):start_index].lower()
        
        # Si ça contient "vu le", "vu l'", "modifiant le"... ce n'est pas un début de décret
        if any(x in context_before for x in ["vu le", "vu l’", "vu l'", "modifiant le", "complétant le"]):
            continue

        anchors.append({
            "start": start_index,
            "id": text_id
        })
    
    return anchors

def find_page_number(content, decree_start_index):
    """
    Retrouve le numéro de page en cherchant le dernier marqueur '--- Page X ---'
    avant le début du décret.
    """
    # On cherche tous les marqueurs avant l'index du décret
    preceding_text = content[:decree_start_index]
    matches = list(REGEX_PAGE_MARKER.finditer(preceding_text))
    
    if matches:
        # Le dernier match est la page courante
        return int(matches[-1].group(1))
    return 1 # Par défaut si pas trouvé (ex: début du fichier)

def process_file(filename, content):
    journal_date = get_journal_metadata(content)
    anchors = find_decree_anchors(content)
    
    results = []
    
    for i in range(len(anchors)):
        current = anchors[i]
        start = current['start']
        
        # La fin est soit le début du prochain, soit la fin du fichier
        if i < len(anchors) - 1:
            end = anchors[i+1]['start']
        else:
            end = len(content)
            
        # Extraction du bloc brut
        raw_block = content[start:end].strip()
        
        # Nettoyage final du bloc (on enlève les marqueurs de page qui traînent au milieu)
        clean_block = re.sub(r"--- Page \d+ ---", "", raw_block)
        clean_block = normalize_text(clean_block)
        
        # Récupération de la page de début
        page_num = find_page_number(content, start)
        
        # Construction du JSON final
        # C'est ici qu'on définit la structure pour le RAG
        doc_object = {
            "source": filename,
            "journal_date": journal_date,
            "page_start": page_num,
            "decree_id": current['id'], # Ex: "Décret exécutif n° 25-59"
            "text": clean_block # TOUT le texte (Titre + Vus + Articles)
        }
        
        # Petit filtre de sécurité : si le texte est trop court (< 100 chars), c'est probablement un déchet
        if len(clean_block) > 100:
            results.append(doc_object)

    return results

def main():
    if not os.path.exists(JSON_DIR):
        os.makedirs(JSON_DIR)

    print(f"Chunking (Mode Décret Entier) depuis : {TXT_DIR}")
    
    for filename in os.listdir(TXT_DIR):
        if filename.endswith(".txt"):
            print(f"Traitement : {filename}")
            path = os.path.join(TXT_DIR, filename)
            
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            decrees = process_file(filename, content)
            
            if decrees:
                out_name = filename.replace(".txt", ".jsonl")
                out_path = os.path.join(JSON_DIR, out_name)
                
                with open(out_path, "w", encoding="utf-8") as f_out:
                    for d in decrees:
                        f_out.write(json.dumps(d, ensure_ascii=False) + "\n")
                
                print(f"   > [OK] {len(decrees)} décrets extraits -> {out_name}")
            else:
                print("   > [VIDE] Aucun décret trouvé.")

if __name__ == "__main__":
    main()
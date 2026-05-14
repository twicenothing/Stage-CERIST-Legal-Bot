import os
import re
import json

# --- CONFIGURATION ---
INPUT_FOLDER = "../../data/txt"
OUTPUT_FOLDER = "../../data/json"

def clean_text(text):
    # 1. Normalize newlines (Crucial for regex anchors)
    text = re.sub(r'\r\n', '\n', text) 
    text = re.sub(r'\n+', '\n', text)
    # 2. Clean strange characters
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF\n]+', ' ', text)
    return text.strip()

def extract_page_mapping(text):
    """Construit la carte des pages et nettoie les balises injectées par le script PDF."""
    page_map = []
    clean_text_no_markers = ""
    last_idx = 0
    
    for match in re.finditer(r'<<<PAGE_(\d+)>>>\s*', text):
        chunk = text[last_idx:match.start()]
        clean_text_no_markers += chunk
        page_map.append((len(clean_text_no_markers), int(match.group(1))))
        last_idx = match.end()
        
    clean_text_no_markers += text[last_idx:]
    return clean_text_no_markers, page_map

def get_page_for_index(char_index, page_map):
    """Trouve la page correspondante à un index de caractère."""
    current_page = "Inconnu"
    for marker_idx, page_num in page_map:
        if char_index >= marker_idx:
            current_page = page_num
        else:
            break
    return current_page

def extract_articles_simple(decree_body: str, doc_type: str = "type1"):
    """
    Splits the decree body into a list of full article strings.
    doc_type détermine la souplesse de l'extraction.
    (TA LOGIQUE ORIGINALE INTACTE)
    """
    if doc_type == "type2":
        article_header_pattern = re.compile(
            r'(?:^|\n)\s*Art(?:icle)?\.?\s*(\d+(?:er|ER)?|unique|ler|Ier)(?:\.?\s*[-—–]+|\s*(?=\n|$))', 
            re.IGNORECASE
        )
    else:
        article_header_pattern = re.compile(
            r'(?:^|\n)\s*Art(?:icle)?\.?\s*(\d+(?:er|ER)?|unique|ler|Ier)\.?\s*[-—–]+', 
            re.IGNORECASE
        )

    matches = list(article_header_pattern.finditer(decree_body))
    
    if not matches and doc_type == "type2":
        chap_pattern = re.compile(
            r'(?:^|\n)\s*Chapitre\s+(\d+|premier|unique)(?:\.?\s*[-—–]+|\s*(?=\n|$))', 
            re.IGNORECASE
        )
        matches = list(chap_pattern.finditer(decree_body))

    if not matches:
        return []

    articles_list = []

    for i in range(len(matches)):
        current_match = matches[i]
        start_pos = current_match.start()
        
        if i + 1 < len(matches):
            end_pos = matches[i+1].start()
        else:
            remaining_text = decree_body[start_pos:]
            stop_markers = ["Fait à Alger", "Fait à ", "Le Premier ministre", "Le Président"]
            cutoff = len(remaining_text)
            for marker in stop_markers:
                idx = remaining_text.find(marker)
                if idx != -1 and idx < cutoff:
                    cutoff = idx
            end_pos = start_pos + cutoff

        full_article_text = decree_body[start_pos:end_pos].strip()
        clean_article_text = re.sub(r'\s+', ' ', full_article_text)

        if clean_article_text:
            articles_list.append(clean_article_text)

    return articles_list

def extract_documents_and_articles(text: str, page_map: list):
    # --- DOUBLE DOCUMENT TITLE REGEX (TA REGEX ORIGINALE INTACTE) ---
    title_pattern = re.compile(
        r"""
        (?:^|\n)                                
        (?:
            # TYPE 1 : Décrets, Arrêtés, Décisions, Avis, Règlements, Lois, Proclamations, Délibérations, Instructions, Ordonnances
            (?P<type1>                                       
              (?:                                  
                (?:
                  (?:Décret|DÉCRET|Decret|DECRET)\s+(?:présidentiel|exécutif|PRÉSIDENTIEL|EXÉCUTIF)|       
                  (?:Arrêté|ARRÊTÉ|Arrete|ARRETE)(?:\s+interministériel|\s+INTERMINISTÉRIEL)?|           
                  (?:Décision|DÉCISION|Decision|DECISION)|
                  (?:Avis|AVIS)|
                  (?:Loi|LOI)|
                  (?:Proclamation|PROCLAMATION)|
                  (?:Délibération|DÉLIBÉRATION|Deliberation|DELIBERATION)|
                  (?:Instruction|INSTRUCTION)(?:\s+(?:interministérielle|INTERMINISTÉRIELLE|présidentielle|PRÉSIDENTIELLE|presidentielle|PRESIDENTIELLE))?|
                  (?:Ordonnance|ORDONNANCE)  
                )\s+(?:n[°o\.]?|du|N[°O\.]?|DU|\d+)
                |
                (?:Règlement|RÈGLEMENT|Reglement|REGLEMENT)\b  
              )                    
              (?:(?!\n\s*Art(?:icle)?\.?\s*(?:\d|[Uu]nique|[Uu]NIQUE)).)*?  
              \.?                                    
              \s* [-—–_H]{3,}
            )
            |
            # TYPE 2 : Accords, Conventions, Mémorandums (et Conventions Internationales)
            (?P<type2>
              (?:
                (?:Accord|ACCORD|Convention|CONVENTION|Mémorandum|MÉMORANDUM|Memorandum)\b
                (?:(?!\n\s*(?:Le Gouvernement|Les Gouvernements|Les Parties|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|Chapitre\s+(?:premier|\d))).)*?
                \b(?:entre|Entre)\b
                (?:(?!\n\s*(?:Le Gouvernement|Les Gouvernements|Les Parties|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|Chapitre\s+(?:premier|\d))).)*?
                \b(?:et|Et)\b
                (?:(?!\n\s*(?:Le Gouvernement|Les Gouvernements|Les Parties|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|Chapitre\s+(?:premier|\d))).)*?
                (?=\n\s*(?:Le Gouvernement|Les Gouvernements|Les Parties|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|Chapitre\s+(?:premier|\d)))
              )
              |
              # Conventions internationales sans "Entre/Et"
              (?:
                (?:Convention|CONVENTION)(?:\s+\d+)?\s+(?:concernant|sur|CONCERNANT|SUR)\b
                (?:(?!\n\s*(?:La conférence|La Conférence|LA CONFERENCE|L'assemblée|L'Assemblée|L'ASSEMBLEE|Le Conseil|Les Etats|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|PARTIE|Chapitre\s+(?:premier|\d))).)*?
                (?=\n\s*(?:La conférence|La Conférence|LA CONFERENCE|L'assemblée|L'Assemblée|L'ASSEMBLEE|Le Conseil|Les Etats|Désireux|Considérant|Préambule|PREAMBULE|Article\s+\d|Art\.|PARTIE|Chapitre\s+(?:premier|\d)))
              )
            )
        )                          
        """, 
        re.VERBOSE | re.DOTALL 
    )

    matches = list(title_pattern.finditer(text))

    if not matches:
        return []

    documents = []

    for i in range(len(matches)):
        match = matches[i]
        
        # 🎯 CALCUL DE LA PAGE (L'AJOUT EST ICI)
        doc_page = get_page_for_index(match.start(), page_map)
        
        if match.group('type1'):
            doc_type = "type1"
            raw_title = re.sub(r'\s*[-—–_H]{3,}$', '', match.group('type1')).strip()
        else:
            doc_type = "type2"
            raw_title = match.group('type2').strip()
        
        if "Article" in raw_title or "Art." in raw_title or "Chapitre" in raw_title:
            continue
        
        clean_title_str = re.sub(r'\s+', ' ', raw_title)

        start_body = match.end()
        if i + 1 < len(matches):
            end_body = matches[i+1].start()
        else:
            end_body = len(text)
            
        body_text = text[start_body:end_body].strip()
        
        simple_articles = extract_articles_simple(body_text, doc_type)
        
        if doc_type == "type1":
            preamble_end_pattern = re.compile(
                r'(?:^|\n)\s*(Décrète|Décrètent|Décide|Décident|Arrête|Arrêtent|.*?adopte.*?suit|.*?promulgue.*?suit|.*?adopte les dispositions suivantes.*?)\s*[:;]\s*(?:\n|$)', 
                re.IGNORECASE
            )
            preamble_match = preamble_end_pattern.search(body_text)
            
            if preamble_match:
                preamble = body_text[:preamble_match.end()].strip()
            else:
                first_art_match = re.search(r'(?:^|\n)\s*(?:Art(?:icle)?\.?\s*(?:1?(?:er|ER)?|unique|ler|Ier)|Chapitre\s+(?:premier|1|unique))\.?\s*[-—–]+', body_text, re.IGNORECASE)
                if first_art_match:
                    preamble = body_text[:first_art_match.start()].strip()
                else:
                    preamble = body_text.strip()
        else:
            preamble_end_pattern = re.compile(
                r'(?:^|\n)\s*(?:sont convenus|ont convenu|sont convenues|ont convenues)(?:\s+de)?\s+ce\s+qui\s+suit\s*:\s*(?:\n|$)', 
                re.IGNORECASE
            )
            preamble_match = preamble_end_pattern.search(body_text)
            
            if preamble_match:
                preamble = body_text[:preamble_match.end()].strip()
            else:
                first_art_match = re.search(r'(?:^|\n)\s*(?:Art(?:icle)?\.?\s*(?:1?(?:er|ER)?|unique|ler|Ier)|Chapitre\s+(?:premier|1|unique))(?:\.?\s*[-—–]+|\s*(?=\n|$))', body_text, re.IGNORECASE)
                if first_art_match:
                    preamble = body_text[:first_art_match.start()].strip()
                else:
                    preamble = body_text.strip()
                    
        context_text = f"{clean_title_str}\n\n{preamble}"
        
        # 🎯 INJECTION DE LA PAGE DANS LA STRUCTURE DE DONNÉES
        documents.append({
            "page": doc_page,
            "title": clean_title_str,
            "articles": simple_articles,
            "context": context_text
        })

    return documents
    
def process_all_files():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output folder: {OUTPUT_FOLDER}")

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.txt')]
    
    if not files:
        print(f"⚠️ No .txt files found in {INPUT_FOLDER}")
        return

    print(f"🚀 Starting batch processing for {len(files)} files...\n")

    for index, filename in enumerate(files):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        print(f"   [{index+1}/{len(files)}] Processing {filename}...", end=" ")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            cleaned_content = clean_text(content)
            
            # 🎯 APPEL DES NOUVELLES FONCTIONS DE MAPPING
            final_text, page_map = extract_page_mapping(cleaned_content)
            data = extract_documents_and_articles(final_text, page_map)
            
            final_output = {
                "source_file": filename,
                "chunking_method": "regex",
                "total_documents": len(data),
                "documents": data
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Done. ({len(data)} docs found)")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")

    print(f"\n🎉 Batch processing complete! Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    process_all_files()
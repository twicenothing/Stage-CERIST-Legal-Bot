import os
import json
import re

# --- CONFIGURATION ---
RAW_TEXT_DIR = "../../data/txt"
OUTPUT_JSON_DIR = "../../data/json"

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]+', ' ', text)
    return text.strip()

def extract_decrees(text: str):
    chunks = []
    
    # Regex de détection (Large)
    header_pattern = re.compile(
        r'(Décret\s+(?:présidentiel|exécutif)|Arrêté|Décision)\s+(?:n[°o\.]?)?\s*(\d+[-‐‑]\d+|\d{1,4}(?!\d))', 
        re.IGNORECASE
    )

    # 1. On repère TOUS les candidats
    raw_matches = list(header_pattern.finditer(text))
    valid_matches = []

    # 2. FILTRAGE INTELLIGENT (Anti-Visas)
    for m in raw_matches:
        start = m.start()
        # On regarde les 50 caractères AVANT le match
        context_before = text[max(0, start-50):start].lower()
        
        # Si ça commence par "vu le", "vu l'", "considérant le", c'est une citation, pas un titre !
        if "vu le" in context_before or "vu l'" in context_before or "application du" in context_before:
            continue
            
        valid_matches.append(m)

    # Si aucun décret trouvé, on prend tout le texte
    if not valid_matches:
        return [{
            "parent": {
                "id": "doc_entier", 
                "text": text, 
                "metadata": {"title": "Document complet"}
            },
            "children": []
        }]

    # 3. DÉCOUPAGE SUR LES MATCHS VALIDES
    for i, match in enumerate(valid_matches):
        doc_type = match.group(1).replace(" ", "_").lower()
        doc_num = match.group(2)
        decree_id = f"{doc_num}" # ex: 24-440
        
        start_pos = match.start()
        
        # La fin est le début du prochain match valide
        if i + 1 < len(valid_matches):
            end_pos = valid_matches[i+1].start()
        else:
            end_pos = len(text)
            
        full_decree_text = text[start_pos:end_pos].strip()
        
        # Titre approximatif (les 300 premiers caractères)
        title_extract = full_decree_text[:300].replace("\n", " ")
        
        parent = {
            "type": "parent",
            "id": decree_id,
            "text": full_decree_text,
            "metadata": {
                "decree_number": doc_num,
                "title": title_extract[:200] + "..."
            }
        }
        
        # Extraction des articles (Inchangée)
        children = []
        article_pattern = r'Art(?:icle)?\.?\s*(\d+(?:er)?)\.?\s*—'
        art_matches = list(re.finditer(article_pattern, full_decree_text, re.IGNORECASE))
        
        for j, art_match in enumerate(art_matches):
            art_num = art_match.group(1).replace("er", "1")
            art_start = art_match.start()
            
            if j + 1 < len(art_matches):
                art_end = art_matches[j+1].start()
            else:
                art_end = len(full_decree_text)
                
            art_text = full_decree_text[art_start:art_end].strip()
            
            children.append({
                "type": "child",
                "id": f"{decree_id}_art{art_num}",
                "text": art_text,
                "metadata": {
                    "parent_id": decree_id,
                    "article_number": art_num
                }
            })
        
        chunks.append({
            "parent": parent,
            "children": children
        })
    
    return chunks

def main():
    if not os.path.exists(OUTPUT_JSON_DIR):
        os.makedirs(OUTPUT_JSON_DIR)

    files = [f for f in os.listdir(RAW_TEXT_DIR) if f.endswith(".txt")]
    print(f"📦 Démarrage du découpage (Anti-Visas activé) sur {len(files)} fichiers...")

    for filename in files:
        file_path = os.path.join(RAW_TEXT_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            clean_content = clean_text(raw_text)
            hierarchical_chunks = extract_decrees(clean_content)
            
            # Petit log pour vérifier
            print(f"📄 {filename} : {len(hierarchical_chunks)} décrets trouvés.")
            for c in hierarchical_chunks:
                print(f"   - ID: {c['parent']['id']} ({len(c['children'])} articles)")

            json_output = {
                "source_file": filename,
                "total_decrees": len(hierarchical_chunks),
                "hierarchical_documents": hierarchical_chunks
            }

            output_filename = filename.replace(".txt", ".json")
            output_path = os.path.join(OUTPUT_JSON_DIR, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

    print("\n🎉 Terminé ! Relance l'indexation (indexer.py).")

if __name__ == "__main__":
    main()
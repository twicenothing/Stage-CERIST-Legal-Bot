
import os
import re
import json

# --- CONFIGURATION ---
# Update these paths to match your folder structure
INPUT_FOLDER = "../../data/txt"
OUTPUT_FOLDER = "../../data/json"

def clean_text(text):
    # 1. Normalize newlines (Crucial for regex anchors)
    text = re.sub(r'\r\n', '\n', text) 
    text = re.sub(r'\n+', '\n', text)
    # 2. Clean strange characters
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF\n]+', ' ', text)
    return text.strip()

def extract_articles_simple(decree_body: str):
    """
    Splits the decree body into a list of full article strings.
    """
    # 1. FIND ARTICLE HEADERS
    article_header_pattern = re.compile(
        r'(?:^|\n)\s*Art(?:icle)?\.?\s*(\d+(?:er|ER)?)\.?\s*[-—–]+', 
        re.IGNORECASE
    )

    matches = list(article_header_pattern.finditer(decree_body))
    
    if not matches:
        return []

    articles_list = []

    # 2. SLICING LOOP
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

def extract_documents_and_articles(text: str):
    # --- STRICT DOCUMENT TITLE REGEX ---
    title_pattern = re.compile(
        r"""
        (?:^|\n)                                
        (                                       
          (?:                                   
            Décret\s+(?:présidentiel|exécutif)|       
            Arrêté(?:\s+interministériel)?|           
            Décision                                  
          )
          \s+
          (?:n[°o\.]?|du)                       
          .*?                                   
          \.                                    
        )                                       
        \s* [-—–_]{3,}                          
        """, 
        re.IGNORECASE | re.VERBOSE | re.DOTALL 
    )

    matches = list(title_pattern.finditer(text))

    if not matches:
        return []

    documents = []

    for i in range(len(matches)):
        match = matches[i]
        raw_title = match.group(1).strip()
        
        # Security Filter
        if "Article" in raw_title or "Art." in raw_title or "Chapitre" in raw_title:
            continue
        
        clean_title_str = re.sub(r'\s+', ' ', raw_title)

        start_body = match.end()
        if i + 1 < len(matches):
            end_body = matches[i+1].start()
        else:
            end_body = len(text)
            
        body_text = text[start_body:end_body].strip()
        simple_articles = extract_articles_simple(body_text)
        
        documents.append({
            "title": clean_title_str,
            "articles": simple_articles,
            "full_context": body_text 
        })

    return documents

def process_all_files():
    # 1. Create Output Directory if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output folder: {OUTPUT_FOLDER}")

    # 2. List all .txt files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.txt')]
    
    if not files:
        print(f"⚠️ No .txt files found in {INPUT_FOLDER}")
        return

    print(f"🚀 Starting batch processing for {len(files)} files...\n")

    # 3. Process Loop
    for index, filename in enumerate(files):
        input_path = os.path.join(INPUT_FOLDER, filename)
        
        # Create output filename (replace .txt with .json)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        print(f"   [{index+1}/{len(files)}] Processing {filename}...", end=" ")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            cleaned_content = clean_text(content)
            data = extract_documents_and_articles(cleaned_content)
            
            final_output = {
                "source_file": filename,
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
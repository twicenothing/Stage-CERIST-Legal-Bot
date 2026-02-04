import os
import re

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "data", "txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "cleaned")

def remove_arabic(text):
    """Removes Arabic characters."""
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+'
    return re.sub(arabic_pattern, '', text)

def is_sommaire_page(text_chunk):
    """
    Returns True if the page is a Table of Contents (Sommaire).
    """
    # 1. Inspect only the top 20 lines
    lines = [line.strip().upper() for line in text_chunk.split('\n') if line.strip()]
    top_lines = lines[:20] 

    for line in top_lines:
        clean_line = line.replace(" ", "")
        # Matches: "SOMMAIRE", "SOMMAIRE (SUITE)"
        if "SOMMAIRE" in clean_line:
            if len(clean_line) < 20 or "SUITE" in clean_line:
                return True
                
    # 2. Backup: Dot Density Check
    dot_pattern = r"\.{4,}\s*\d+\s*$"
    matches = re.findall(dot_pattern, text_chunk, re.MULTILINE)
    if len(matches) >= 3:
        return True

    return False

def is_header_noise(line):
    """Detects standard page headers/footers/dates to remove."""
    line = line.strip()
    if not line: return True

    mois = r"(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)"
    
    noise_patterns = [
        r"^JOURNAL OFFICIEL",
        r"^DE LA REPUBLIQUE",
        r"^ALGERIENNE DEMOCRATIQUE",
        r"^\d{1,2}\s+[a-zA-Z\s]+\s+14\d{2}$", # Hijri Date
        rf"^\d{{1,2}}\s+{mois}\s+20\d{{2}}$",   # Gregorian Date
        r"^\d+$", # Standalone page number
        r"^\d+.*Dinar.*" # Price
    ]
    
    for p in noise_patterns:
        if re.search(p, line, re.IGNORECASE):
            return True
            
    return False

def clean_page_text(raw_text):
    """Clean a single page: remove noise lines and Arabic."""
    lines = raw_text.split('\n')
    clean_lines = []
    
    for line in lines:
        line = remove_arabic(line)
        if is_header_noise(line): continue
        
        clean_l = line.strip()
        if not clean_l: continue
            
        clean_lines.append(clean_l)
        
    return "\n".join(clean_lines)

def process_file(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    
    with open(input_path, "r", encoding="utf-8") as f:
        full_content = f.read()

    # Split by markers: ==== PAGE 1 ====
    page_splits = re.split(r"(==== PAGE \d+ ====)", full_content)
    
    final_content = []
    current_page_marker = ""
    
    for part in page_splits:
        part = part.strip()
        if not part: continue
        
        # Capture the marker
        if part.startswith("==== PAGE"):
            current_page_marker = part
            continue
            
        # Process the Text Content
        text_chunk = part
        
        # Get Page Number
        try:
            page_num = current_page_marker.split()[2]
        except:
            page_num = "?"

        # --- RULE 1: DROP PAGE 1 (Cover Page) ---
        if page_num == "1":
            # print(f"   🗑️  Dropped Page 1 (Cover)")
            continue

        # --- RULE 2: DROP SOMMAIRE ---
        if is_sommaire_page(text_chunk):
            # print(f"   🗑️  [Page {page_num}] Dropped Sommaire")
            continue
            
        # --- RULE 3: CLEAN CONTENT ---
        cleaned_text = clean_page_text(text_chunk)
        
        if cleaned_text:
            final_content.append(f"[[PAGE_REF:{page_num}]]")
            final_content.append(cleaned_text)
            final_content.append("\n" + ("-" * 20) + "\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_content))

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    
    if not files:
        print("❌ No text files found.")
        return

    print(f"🧹 Cleaning {len(files)} files (Dropping Pg1 & Sommaires)...")
    
    for f in files:
        process_file(f)
        print(f"   ✅ Cleaned: {f}")

if __name__ == "__main__":
    main()
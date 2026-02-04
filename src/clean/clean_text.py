import os
import re

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/txt"))
CLEANED_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/cleaned"))

def remove_sommaire_pages(text):
    """
    Removes any page block that contains the word 'SOMMAIRE'.
    """
    parts = re.split(r"(--- Page \d+ ---)", text)
    cleaned_parts = []
    
    # Handle start of file
    if parts[0].strip():
        cleaned_parts.append(parts[0])

    for i in range(1, len(parts), 2):
        marker = parts[i]
        content = parts[i+1]
        
        # Heuristic: If it has SOMMAIRE, it's trash.
        if "SOMMAIRE" in content:
            print(f"   [DELETE] Nuked Sommaire at {marker.strip()}")
            continue
        
        cleaned_parts.append(marker)
        cleaned_parts.append(content)
        
    return "".join(cleaned_parts)

def stitch_broken_lines(text):
    """
    Reconstructs sentences broken by newlines.
    Rules:
    1. 'word-\nnext' -> 'wordnext' (Fix hyphenation)
    2. 'word\nnext'  -> 'word next' (If 'word' doesn't end in punctuation)
    """
    
    # 1. Fix Hyphenation (Word split across lines)
    # Ex: "constitu-\ntion" -> "constitution"
    # We remove the hyphen and the newline.
    text = re.sub(r'-\n\s*', '', text)

    # 2. Fix Broken Lines (The Main Logic)
    # Regex Explanation:
    # (?<![.:;?!])   -> Lookbehind: If the char BEFORE \n is NOT punctuation (. : ; ? !)
    # \n             -> The target newline to remove
    # (?!\s*(?:---|===)) -> Lookahead: Ensure we don't merge into a Page Marker or Separator
    
    # We replace the newline with a SPACE.
    text = re.sub(r'(?<![.:;?!])\n(?!\s*(?:---|===))', ' ', text)
    
    # 3. Cleanup multiple spaces created by the merge
    text = re.sub(r' +', ' ', text)
    
    # 4. Restore paragraph breaks
    # Sometimes this logic kills valid empty lines. 
    # If you want to force space between "Article 1" and "Article 2", 
    # standard legal text usually has punctuation so it should be fine.
    
    return text

def main():
    if not os.path.exists(CLEANED_DIR):
        os.makedirs(CLEANED_DIR)

    print(f"Cleaning and Stitching files from: {TXT_DIR}")
    
    for filename in os.listdir(TXT_DIR):
        if filename.lower().endswith(".txt"):
            input_path = os.path.join(TXT_DIR, filename)
            output_path = os.path.join(CLEANED_DIR, filename)
            
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()

                # Step 1: Remove Sommaire
                content_no_sommaire = remove_sommaire_pages(raw_content)
                
                # Step 2: Stitch Sentences
                final_content = stitch_broken_lines(content_no_sommaire)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(final_content)
                    
                print(f"[FIXED] {filename} -> Sommaire removed & Lines stitched.")

            except Exception as e:
                print(f"[ERROR] {filename}: {e}")

if __name__ == "__main__":
    main()
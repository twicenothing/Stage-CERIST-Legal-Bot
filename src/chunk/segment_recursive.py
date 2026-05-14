import os
import re
import json

# --- CONFIGURATION ---
INPUT_FOLDER = "../../data/txt"
OUTPUT_FOLDER = "../../data/json_recursive" # Separate folder for parallel pipeline
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def clean_text(text):
    """Reusing your cleaning logic to ensure text quality before chunking."""
    text = re.sub(r'\r\n', '\n', text) 
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF\n]+', ' ', text)
    return text.strip()

# ==============================================================================
# 🎯 NOUVELLES FONCTIONS DE PAGINATION (Identiques au script Regex)
# ==============================================================================
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
    """Trouve la page correspondante à un index de caractère absolu."""
    current_page = "Inconnu"
    for marker_idx, page_num in page_map:
        if char_index >= marker_idx:
            current_page = page_num
        else:
            break
    return current_page
# ==============================================================================

def recursive_split(text: str, chunk_size: int, chunk_overlap: int, separators=None) -> list[str]:
    """Pure Python recursive character chunker."""
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    if len(text) <= chunk_size:
        return [text]

    active_separator = separators[-1]
    for sep in separators:
        if sep == "":
            active_separator = sep
            break
        if sep in text:
            active_separator = sep
            break

    if active_separator == "":
        splits = list(text) 
    else:
        splits = text.split(active_separator)

    chunks = []
    current_chunk_splits = []
    current_len = 0

    for split in splits:
        if len(split) > chunk_size:
            if current_chunk_splits:
                chunks.append(active_separator.join(current_chunk_splits))
                current_chunk_splits = []
                current_len = 0

            try:
                next_separators = separators[separators.index(active_separator) + 1:]
            except ValueError:
                next_separators = [""]

            recursed_chunks = recursive_split(split, chunk_size, chunk_overlap, next_separators)
            chunks.extend(recursed_chunks)
            continue

        sep_len = len(active_separator) if current_chunk_splits else 0

        if current_len + sep_len + len(split) > chunk_size:
            chunks.append(active_separator.join(current_chunk_splits))

            overlap_splits = []
            overlap_len = 0
            for s in reversed(current_chunk_splits):
                s_len = len(s) + (len(active_separator) if overlap_splits else 0)
                if overlap_len + s_len <= chunk_overlap:
                    overlap_splits.insert(0, s)
                    overlap_len += s_len
                else:
                    break

            current_chunk_splits = overlap_splits
            current_len = overlap_len

        current_chunk_splits.append(split)
        current_len += len(active_separator) + len(split) if len(current_chunk_splits) > 1 else len(split)

    if current_chunk_splits:
        chunks.append(active_separator.join(current_chunk_splits))

    return chunks

def process_all_txt_to_recursive_json():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output folder: {OUTPUT_FOLDER}")

    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.txt')]
    
    if not files:
        print(f"⚠️ No .txt files found in {INPUT_FOLDER}")
        return

    print(f"🚀 Starting Recursive Chunking for {len(files)} files...\n")

    for index, filename in enumerate(files):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_filename = os.path.splitext(filename)[0] + "_recursive.json"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        print(f"   [{index+1}/{len(files)}] Processing {filename}...", end=" ")

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 1. Clean the raw text (les marqueurs <<<PAGE_X>>> survivent à ça)
            cleaned_content = clean_text(content)
            
            # 2. Extract page mapping and get the pure text WITHOUT markers
            final_text, page_map = extract_page_mapping(cleaned_content)
            
            # 3. Run the recursive split on the pure text
            raw_chunks = recursive_split(final_text, CHUNK_SIZE, CHUNK_OVERLAP)
            
            # 4. Format into JSON with rich metadata + PAGE MAPPING
            formatted_chunks = []
            search_offset = 0  # Curseur de recherche glissant
            
            for i, chunk_text in enumerate(raw_chunks):
                # On cherche l'index exact où ce chunk a commencé dans le texte final
                chunk_start_idx = final_text.find(chunk_text, search_offset)
                
                # Sécurité (ne devrait jamais arriver si text.find réussit)
                if chunk_start_idx == -1:
                    chunk_start_idx = search_offset
                
                # On détermine la page avec l'index absolu
                chunk_page = get_page_for_index(chunk_start_idx, page_map)
                
                # On avance le curseur juste après le début de ce chunk 
                # (pour gérer correctement les overlaps au prochain tour)
                search_offset = chunk_start_idx + 1

                formatted_chunks.append({
                    "chunk_index": i + 1,
                    "text": chunk_text,
                    "metadata": {
                        "source_file": filename,
                        "chunking_method": "recursive",
                        "page": chunk_page # 🎯 PAGE INJECTÉE ICI
                    }
                })

            final_output = {
                "source_file": filename,
                "chunking_method": "recursive",
                "total_chunks": len(formatted_chunks),
                "chunks": formatted_chunks
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Done. ({len(formatted_chunks)} chunks generated)")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")

    print(f"\n🎉 Batch processing complete! Check the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    process_all_txt_to_recursive_json()
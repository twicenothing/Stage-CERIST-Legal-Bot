#!/usr/bin/env python3
"""
Robust Segmentation Script for Journal Officiel.
Fixes: Missing documents due to strict title matching, missing articles due to punctuation checks.
"""

import re
import json
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "data", "cleaned") # Read from cleaned!
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "json")

class DocumentSegmenter:
    """Segments Journal Officiel documents into structured chunks."""
    
    def __init__(self):
        self.current_page = "1"
    
    def extract_page_number(self, text_chunk, current_val):
        """
        Scans for [[PAGE_REF:X]] tags in the text chunk.
        Returns the last found page number, or the current one if none found.
        """
        matches = re.findall(r"\[\[PAGE_REF:(\d+)\]\]", text_chunk)
        if matches:
            return matches[-1]
        return current_val
    
    def clean_text(self, text):
        """Clean up text: remove page tags, fix spacing."""
        # Remove the page tags we added in cleaning (we only use them for logic, not content)
        text = re.sub(r"\[\[PAGE_REF:\d+\]\]", "", text)
        
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse excessive newlines (keep max 2)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def find_all_documents(self, full_text):
        """
        Robust Document Discovery:
        1. Find all '————' separators.
        2. Look BACKWARDS from each separator to find the 'Start Keyword' (Décret, Arrêté...).
        """
        documents = []
        
        # 1. Find all separators (The Anchor)
        # We use 'match' objects to get exact positions
        sep_iter = re.finditer(r'————+', full_text)
        
        for sep_match in sep_iter:
            sep_start = sep_match.start()
            sep_end = sep_match.end()
            
            # 2. Define a "Lookback Window" (e.g., 1500 chars before separator)
            # This covers even long titles.
            window_start = max(0, sep_start - 1500)
            window_text = full_text[window_start:sep_start]
            
            # 3. Find the LAST "Start Keyword" in this window
            # This is the title corresponding to this separator.
            # We add 'Loi', 'Ordonnance', 'Décision' to catch everything.
            keywords_pattern = r'(?:\n|^)\s*(Décret|Arrêté|Loi|Ordonnance|Décision)\b'
            
            # We want the LAST match in the window (closest to the separator)
            matches = list(re.finditer(keywords_pattern, window_text, re.IGNORECASE))
            
            if matches:
                last_match = matches[-1]
                
                # Calculate absolute position of the Title Start in full_text
                title_start_abs = window_start + last_match.start()
                
                # Extract the raw title (from Start Keyword to Separator)
                raw_title = full_text[title_start_abs:sep_start].strip()
                
                # Determine Type
                doc_type_word = last_match.group(1).capitalize() # e.g. "Décret"
                
                # Refine Type (e.g. "Décret exécutif")
                full_type_match = re.match(r'(Décret\s+\w+|Arrêté\s+\w+|Loi|Ordonnance|Décision)', raw_title, re.IGNORECASE)
                doc_type = full_type_match.group(1) if full_type_match else doc_type_word
                
                documents.append({
                    'type': doc_type,
                    'start_pos': title_start_abs,
                    'body_start_pos': sep_end, # Body starts AFTER the separator
                    'raw_title': raw_title
                })
        
        # Sort by position (just in case)
        documents.sort(key=lambda x: x['start_pos'])
        
        # Filter: Remove duplicates or contained matches (unlikely with this logic but good practice)
        return documents

    def split_preamble_and_articles(self, body_text):
        """
        Separates the Context (Vu...) from the Rules (Articles).
        """
        # 1. Look for the standard "Transition Trigger"
        # "Décrète :", "Arrête :", "Ordonne :"
        trigger_match = re.search(r'(Décrète\s*:|Arrête\s*:|Arrêtent\s*:|Ordonne\s*:|Décide\s*:)', body_text, re.IGNORECASE)
        
        if trigger_match:
            preamble = body_text[:trigger_match.start()]
            articles_block = body_text[trigger_match.end():]
            return preamble, articles_block
            
        # 2. Fallback: If no trigger, look for "Article 1"
        # We use a loose regex here just to find the split point
        first_art = re.search(r'(?:\n|^)\s*(Article\s+1er|Art\.?\s*1\b)', body_text, re.IGNORECASE)
        
        if first_art:
            preamble = body_text[:first_art.start()]
            articles_block = body_text[first_art.start():]
            return preamble, articles_block
            
        # 3. Last Resort: No split found -> It's all content (or all preamble)
        # Usually implies a short decision or formatting error.
        return body_text, ""

    def extract_articles(self, text):
        """
        Robust Article Extraction.
        Splits text by headers like 'Article 1er', 'Art. 2', 'Art 3'.
        """
        chunks = []
        
        # THE FIX: A more permissive regex for headers.
        # 1. Must be at start of line/string: (?:\n|^)\s*
        # 2. Keywords: Article or Art
        # 3. Number: 1er or digits
        # 4. Optional separator: dot, dash, space
        # We capture the whole header in group 1 to keep it.
        header_pattern = r'(?:^|\n)\s*((?:Article|Art\.?)\s+(?:1er|\d+(?:er)?)\b(?:[\.\-—–\s]*))'
        
        # re.split includes the capture group (the header) in the result list
        parts = re.split(header_pattern, text, flags=re.IGNORECASE)
        
        # parts[0] is text before the first article (usually empty or garbage)
        # parts[1] = Header (e.g. "Art. 1."), parts[2] = Body, parts[3] = Header...
        
        # If no articles found, return whole text as one chunk? 
        # No, better to return nothing and let the caller handle the "Global Context".
        if len(parts) < 3:
            return []
            
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip()
            
            # Cleanup Header (remove trailing punctuation for clean display)
            clean_header = re.sub(r'[\.\-—–\s]+$', '', header).strip()
            
            # Cleanup Content (remove signature garbage at the end of the last article)
            for stop_word in ['Fait à Alger', 'Fait à', 'Le Président', 'Le Premier ministre']:
                if stop_word in content:
                    content = content.split(stop_word)[0].strip()
            
            if content:
                chunks.append({
                    'header': clean_header,
                    'content': content
                })
                
        return chunks

    def segment_file(self, filename):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
        
        with open(input_path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # 1. Find Documents
        docs = self.find_all_documents(full_text)
        if not docs:
            print(f"   ⚠️  No documents found in {filename} (Check '————' separators)")
            return

        final_chunks = []
        
        # Global page tracker for the file
        current_page = "1"
        
        # Process Documents
        for i, doc in enumerate(docs):
            # Determine end of this document (start of next one or EOF)
            doc_end = docs[i+1]['start_pos'] if i + 1 < len(docs) else len(full_text)
            
            # Extract raw body
            raw_body = full_text[doc['body_start_pos'] : doc_end]
            
            # --- PAGE TRACKING ---
            # Get the page number that applies to the START of this document
            # by looking at the text before it.
            text_before = full_text[:doc['start_pos']]
            start_page = self.extract_page_number(text_before, current_page)
            current_page = start_page # Update tracker
            
            # Clean Title
            clean_title = self.clean_text(doc['raw_title'])
            
            # Split Body
            preamble, articles_block = self.split_preamble_and_articles(raw_body)
            clean_preamble = self.clean_text(preamble)
            
            # Extract Articles
            articles = self.extract_articles(articles_block)
            
            if articles:
                # Case A: We found specific articles
                for art in articles:
                    # Check if page changed inside the article content
                    # (This is rough estimation, assigning the page where the article *starts*)
                    # Ideally, we check for markers inside 'preamble' + previous articles
                    
                    chunk = {
                        "source_file": filename,
                        "document_type": doc['type'],
                        "document_title": clean_title,
                        "page_number": current_page,
                        "article_header": art['header'],
                        "article_content": self.clean_text(art['content']),
                        "full_context": f"{clean_title}\n\n{clean_preamble}\n\n{art['header']} : {self.clean_text(art['content'])}"
                    }
                    final_chunks.append(chunk)
            else:
                # Case B: No "Article X" structure found (e.g. short decrees)
                # Treat the whole body as one chunk
                clean_body = self.clean_text(raw_body)
                if clean_body:
                    chunk = {
                        "source_file": filename,
                        "document_type": doc['type'],
                        "document_title": clean_title,
                        "page_number": current_page,
                        "article_header": "Texte intégral",
                        "article_content": clean_body,
                        "full_context": f"{clean_title}\n\n{clean_body}"
                    }
                    final_chunks.append(chunk)

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            
        print(f"   ✅ Saved {len(final_chunks)} chunks from {filename}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"🧩 Segmenting {len(files)} files...")
    
    segmenter = DocumentSegmenter()
    for f in files:
        segmenter.segment_file(f)

if __name__ == "__main__":
    main()
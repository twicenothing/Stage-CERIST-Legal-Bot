#!/usr/bin/env python3
"""
Improved segmentation script for Journal Officiel documents.
Extracts Décrets and Arrêtés with their associated articles.
"""

import re
import json
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_DIR = os.path.join(BASE_DIR, "data", "txt")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "json")


class DocumentSegmenter:
    """Segments Journal Officiel documents into structured chunks."""
    
    def __init__(self):
        self.current_page = 1
    
    def extract_page_number(self, text_before_position):
        """Extract the most recent page number from text."""
        page_matches = re.findall(r'==== PAGE (\d+) ====', text_before_position)
        if page_matches:
            return int(page_matches[-1])
        return self.current_page
    
    def clean_text(self, text):
        """Clean up text by removing excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def find_all_documents(self, full_text):
        """
        Find all Décrets and Arrêtés in the text.
        
        Returns a list of tuples: (doc_type, start_pos, title_end_pos, separator_pos)
        """
        documents = []
        
        # More specific pattern: Look for document headers that end with the separator
        # The title is usually one or more lines, ending with ————
        
        # First, find all separator positions
        separators = [(m.start(), m.end()) for m in re.finditer(r'————+', full_text)]
        
        for sep_start, sep_end in separators:
            # Look backward from separator to find the document title
            # Title starts with Décret or Arrêté and can span multiple lines
            text_before_sep = full_text[max(0, sep_start-2000):sep_start]
            
            # Match the title (last occurrence of Décret/Arrêté before the separator)
            title_patterns = [
                r'(Décret\s+présidentiel\s+n°[^\n]+(?:\n[^\n]+)*)',
                r'(Décret\s+exécutif\s+n°[^\n]+(?:\n[^\n]+)*)',
                r'(Décret\s+législatif\s+n°[^\n]+(?:\n[^\n]+)*)',
                r'(Arrêté\s+interministériel\s+du[^\n]+(?:\n[^\n]+)*)',
                r'(Arrêté\s+du[^\n]+(?:\n[^\n]+)*)',
            ]
            
            title_match = None
            doc_type = None
            
            for pattern in title_patterns:
                matches = list(re.finditer(pattern, text_before_sep, re.IGNORECASE))
                if matches:
                    title_match = matches[-1]  # Get the last (closest to separator)
                    title_text = title_match.group(1).strip()
                    
                    # Determine type
                    if re.match(r'Décret', title_text, re.IGNORECASE):
                        doc_type = 'Décret'
                    elif re.match(r'Arrêté', title_text, re.IGNORECASE):
                        doc_type = 'Arrêté'
                    break
            
            if title_match and doc_type:
                # Calculate absolute positions
                abs_title_start = max(0, sep_start-2000) + title_match.start()
                abs_title_end = max(0, sep_start-2000) + title_match.end()
                
                # Clean the title - only take the first few lines (the actual title)
                # Stop at first "Vu" or other preamble markers
                raw_title = title_match.group(1).strip()
                title_lines = raw_title.split('\n')
                
                clean_title_lines = []
                for line in title_lines:
                    line = line.strip()
                    # Stop if we hit preamble markers
                    if any(marker in line for marker in ['Vu ', 'Le Président', 'Le Premier', 'Sur le rapport']):
                        break
                    if line:
                        clean_title_lines.append(line)
                
                clean_title = ' '.join(clean_title_lines).strip()
                
                documents.append({
                    'type': doc_type,
                    'start': abs_title_start,
                    'title_end': abs_title_end,
                    'separator_pos': sep_end,
                    'title': clean_title
                })
        
        # Sort by position
        documents.sort(key=lambda x: x['start'])
        
        # Remove duplicates (same separator found multiple times)
        unique_docs = []
        last_sep = -1
        for doc in documents:
            if doc['separator_pos'] != last_sep:
                unique_docs.append(doc)
                last_sep = doc['separator_pos']
        
        return unique_docs
    
    def extract_document_body(self, full_text, doc_info, next_doc_start=None):
        """
        Extract the body of a document (from after separator to next document or EOF).
        """
        body_start = doc_info['separator_pos']
        body_end = next_doc_start if next_doc_start else len(full_text)
        
        body_text = full_text[body_start:body_end]
        return body_text
    
    def split_preamble_and_articles(self, body_text, doc_type):
        """
        Split the body into preamble (Vu...) and articles section.
        
        The articles section starts after:
        - "Décrète :" for Décrets
        - "Arrête :" or "Arrêtent :" for Arrêtés
        """
        # Look for the trigger word
        if doc_type == 'Décret':
            trigger_pattern = r'Décrète\s*:'
        else:  # Arrêté
            trigger_pattern = r'Arrête(?:nt)?\s*:'
        
        trigger_match = re.search(trigger_pattern, body_text, re.IGNORECASE)
        
        if trigger_match:
            preamble = body_text[:trigger_match.start()].strip()
            articles_section = body_text[trigger_match.end():].strip()
        else:
            # Fallback: look for first Article
            first_article = re.search(r'Article\s+(?:1er|\d+)', body_text, re.IGNORECASE)
            if first_article:
                preamble = body_text[:first_article.start()].strip()
                articles_section = body_text[first_article.start():].strip()
            else:
                # No clear structure, treat all as preamble
                preamble = body_text
                articles_section = ""
        
        return preamble, articles_section
    
    def extract_articles(self, articles_text):
        """
        Extract individual articles from the articles section.
        
        Returns a list of dicts: {'header': 'Article 1er', 'content': '...'}
        """
        articles = []
        
        # Pattern to match ACTUAL article headers (must have period and dash)
        # Matches: "Article 1er. —" or "Art. 2. —" (the period and dash are crucial!)
        # This avoids matching inline references like "l'article 26" or "de l'article 177"
        article_pattern = r'((?:Article|Art\.?)\s+(?:1er|\d+(?:er)?)\.?\s*[.—–-])'
        
        # Split by article headers but keep the headers
        parts = re.split(f'({article_pattern})', articles_text, flags=re.IGNORECASE)
        
        # parts[0] is usually empty or garbage before first article
        # parts[1] = "Article 1er. —", parts[2] = content, parts[3] = "Art. 2. —", parts[4] = content...
        
        i = 1
        while i < len(parts):
            if i + 1 >= len(parts):
                break
                
            raw_header = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            
            # Extract clean article number from header
            # "Article 1er. —" -> "Article 1er"
            # "Art. 2. —" -> "Art. 2"
            header_clean = re.sub(r'[.—\-–\s]+$', '', raw_header).strip()
            
            # Clean up the content
            # Remove signature blocks and end markers
            for end_marker in [
                'Fait à Alger',
                'Fait à ',
                'Le Président de la République',
                'Le Premier ministre',
                'Abdelmadjid TEBBOUNE',
                'Mohamed Ennadir LARBAOUI',
                'Arrêtent :',  # Stop if we hit another document
                'Décrète :',
            ]:
                if end_marker in content:
                    content = content.split(end_marker)[0].strip()
            
            # Remove trailing separators and punctuation
            content = content.rstrip(' .—–-')
            
            # Skip empty or very short articles (probably extraction errors)
            if content and len(content) > 20:  # At least some meaningful content
                articles.append({
                    'header': header_clean,
                    'content': content
                })
            
            i += 2
        
        return articles
    
    def segment_file(self, filename):
        """Segment a single text file into structured JSON chunks."""
        
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".json"))
        
        print(f"\n📄 Processing: {filename}")
        
        # Read the full text
        with open(input_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        
        # Find all documents
        documents = self.find_all_documents(full_text)
        
        if not documents:
            print(f"   ⚠️  No Décrets or Arrêtés found in {filename}")
            return
        
        print(f"   Found {len(documents)} documents")
        
        all_chunks = []
        
        # Process each document
        for idx, doc_info in enumerate(documents):
            # Get the next document's start position (or None if this is the last)
            next_start = documents[idx + 1]['start'] if idx + 1 < len(documents) else None
            
            # Extract document body
            body_text = self.extract_document_body(full_text, doc_info, next_start)
            
            # Get page number
            text_before = full_text[:doc_info['start']]
            page_num = self.extract_page_number(text_before)
            
            # Clean title
            title = self.clean_text(doc_info['title'])
            
            # Split into preamble and articles
            preamble, articles_section = self.split_preamble_and_articles(
                body_text, doc_info['type']
            )
            
            # Extract individual articles
            articles = self.extract_articles(articles_section)
            
            # Create chunks for each article
            for article in articles:
                chunk = {
                    'source_file': filename,
                    'document_type': doc_info['type'],
                    'document_title': title,
                    'page_number': page_num,
                    'article_header': article['header'],
                    'article_content': self.clean_text(article['content']),
                    'full_context': f"{title}\n\n{article['header']}\n{article['content']}"
                }
                all_chunks.append(chunk)
        
        # Save to JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ Extracted {len(all_chunks)} article chunks from {len(documents)} documents")
        
        # Show summary
        decrets = sum(1 for d in documents if d['type'] == 'Décret')
        arretes = sum(1 for d in documents if d['type'] == 'Arrêté')
        print(f"      - Décrets: {decrets}")
        print(f"      - Arrêtés: {arretes}")


def main():
    """Main entry point."""
    
    # Create output directory if needed
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get all text files
    txt_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    
    if not txt_files:
        print(f"❌ No .txt files found in {INPUT_DIR}")
        return
    
    print("=" * 70)
    print("📚 Journal Officiel Document Segmentation")
    print("=" * 70)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Files to process: {len(txt_files)}")
    print("=" * 70)
    
    segmenter = DocumentSegmenter()
    
    for filename in txt_files:
        segmenter.segment_file(filename)
    
    print("\n" + "=" * 70)
    print("✅ Segmentation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
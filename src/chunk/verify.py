#!/usr/bin/env python3
"""
Verification script to check if all Décrets, Arrêtés, and Articles 
are properly extracted from the TXT files and present in the JSON outputs.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TXT_DIR = os.path.join(BASE_DIR, "data", "txt")
JSON_DIR = os.path.join(BASE_DIR, "data", "json")


class ExtractionVerifier:
    """Verifies completeness of PDF extraction and chunking."""
    
    def __init__(self):
        self.txt_dir = Path(TXT_DIR)
        self.json_dir = Path(JSON_DIR)
        self.results = {}
    
    def find_documents_in_txt(self, txt_file):
        """
        Find all Décrets and Arrêtés in a TXT file.
        Returns dict with document info.
        """
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        documents = []
        
        # Find all separators
        separators = [(m.start(), m.end()) for m in re.finditer(r'————+', content)]
        
        for sep_start, sep_end in separators:
            # Look backward to find title
            text_before = content[max(0, sep_start-2000):sep_start]
            
            # Try to match Décret or Arrêté titles
            patterns = [
                (r'(Décret\s+présidentiel\s+n°\s*[\d-]+)', 'Décret présidentiel'),
                (r'(Décret\s+exécutif\s+n°\s*[\d-]+)', 'Décret exécutif'),
                (r'(Décret\s+législatif\s+n°\s*[\d-]+)', 'Décret législatif'),
                (r'(Arrêté\s+interministériel\s+du\s+[\d\s\w]+)', 'Arrêté interministériel'),
                (r'(Arrêté\s+du\s+[\d\s\w]+)', 'Arrêté'),
            ]
            
            for pattern, doc_type in patterns:
                matches = list(re.finditer(pattern, text_before, re.IGNORECASE))
                if matches:
                    title_match = matches[-1]  # Get closest to separator
                    title = title_match.group(1).strip()
                    
                    # Look for articles after this separator
                    # Get text from separator to next separator or EOF
                    next_sep_idx = separators.index((sep_start, sep_end))
                    if next_sep_idx + 1 < len(separators):
                        body_end = separators[next_sep_idx + 1][0]
                    else:
                        body_end = len(content)
                    
                    body = content[sep_end:body_end]
                    
                    # Count articles
                    # Look for "Article 1er. —" or "Art. 2. —" patterns
                    article_pattern = r'(?:Article|Art\.?)\s+(?:1er|\d+(?:er)?)\.?\s*[.—–-]'
                    article_matches = list(re.finditer(article_pattern, body, re.IGNORECASE))
                    
                    articles = []
                    for art_match in article_matches:
                        # Extract article number
                        art_text = art_match.group(0)
                        # Clean it up
                        art_clean = re.sub(r'[.—\-–\s]+$', '', art_text).strip()
                        articles.append(art_clean)
                    
                    documents.append({
                        'type': doc_type,
                        'title': title,
                        'article_count': len(articles),
                        'articles': articles,
                        'position': sep_start
                    })
                    break
        
        return documents
    
    def load_json_data(self, json_file):
        """Load and organize JSON data by document."""
        with open(json_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        
        # Group by document title
        docs = defaultdict(lambda: {
            'type': None,
            'articles': []
        })
        
        for chunk in chunks:
            title = chunk['document_title']
            docs[title]['type'] = chunk['document_type']
            docs[title]['articles'].append(chunk['article_header'])
        
        return docs
    
    def verify_file(self, txt_file):
        """Verify a single TXT/JSON file pair."""
        filename = txt_file.name
        json_file = self.json_dir / filename.replace('.txt', '.json')
        
        print(f"\n{'='*70}")
        print(f"📄 Verifying: {filename}")
        print(f"{'='*70}")
        
        # Check if JSON exists
        if not json_file.exists():
            print(f"❌ ERROR: JSON file not found: {json_file}")
            return {
                'status': 'ERROR',
                'message': 'JSON file missing'
            }
        
        # Find documents in TXT
        txt_docs = self.find_documents_in_txt(txt_file)
        print(f"\n📖 Found in TXT: {len(txt_docs)} documents")
        
        # Load JSON data
        json_docs = self.load_json_data(json_file)
        print(f"📦 Found in JSON: {len(json_docs)} documents")
        
        # Detailed comparison
        print(f"\n{'─'*70}")
        print("DETAILED COMPARISON")
        print(f"{'─'*70}")
        
        total_txt_articles = 0
        total_json_articles = 0
        missing_docs = []
        missing_articles = []
        extra_articles = []
        
        for idx, txt_doc in enumerate(txt_docs, 1):
            total_txt_articles += txt_doc['article_count']
            
            print(f"\n{idx}. {txt_doc['type']}: {txt_doc['title'][:60]}...")
            print(f"   Articles in TXT: {txt_doc['article_count']}")
            
            # Try to find matching document in JSON
            # Match by partial title (first few words)
            title_key_words = ' '.join(txt_doc['title'].split()[:5])
            
            matching_json = None
            for json_title, json_data in json_docs.items():
                if title_key_words.lower() in json_title.lower():
                    matching_json = (json_title, json_data)
                    break
            
            if matching_json:
                json_title, json_data = matching_json
                json_article_count = len(json_data['articles'])
                total_json_articles += json_article_count
                
                print(f"   Articles in JSON: {json_article_count}")
                
                # Check if counts match
                if txt_doc['article_count'] == json_article_count:
                    print(f"   ✅ Article count matches!")
                else:
                    diff = txt_doc['article_count'] - json_article_count
                    if diff > 0:
                        print(f"   ⚠️  Missing {diff} article(s) in JSON")
                        missing_articles.append({
                            'document': txt_doc['title'],
                            'missing_count': diff
                        })
                    else:
                        print(f"   ⚠️  Extra {-diff} article(s) in JSON")
                        extra_articles.append({
                            'document': txt_doc['title'],
                            'extra_count': -diff
                        })
                
                # Show article details
                if txt_doc['article_count'] <= 10:  # Only for small docs
                    print(f"   TXT Articles: {', '.join(txt_doc['articles'])}")
                    print(f"   JSON Articles: {', '.join(json_data['articles'])}")
            else:
                print(f"   ❌ NOT FOUND in JSON!")
                missing_docs.append(txt_doc['title'])
        
        # Check for extra documents in JSON
        txt_titles_normalized = [' '.join(doc['title'].split()[:5]).lower() 
                                 for doc in txt_docs]
        
        extra_json_docs = []
        for json_title in json_docs.keys():
            json_normalized = ' '.join(json_title.split()[:5]).lower()
            if not any(json_normalized in txt_title or txt_title in json_normalized 
                      for txt_title in txt_titles_normalized):
                extra_json_docs.append(json_title)
        
        # Summary
        print(f"\n{'='*70}")
        print("📊 SUMMARY")
        print(f"{'='*70}")
        
        print(f"\nDocuments:")
        print(f"  TXT:  {len(txt_docs)}")
        print(f"  JSON: {len(json_docs)}")
        
        print(f"\nArticles:")
        print(f"  TXT:  {total_txt_articles}")
        print(f"  JSON: {total_json_articles}")
        
        issues = []
        
        if missing_docs:
            print(f"\n❌ Missing Documents in JSON: {len(missing_docs)}")
            for doc in missing_docs:
                print(f"   - {doc[:70]}...")
            issues.append(f"{len(missing_docs)} missing documents")
        
        if extra_json_docs:
            print(f"\n⚠️  Extra Documents in JSON: {len(extra_json_docs)}")
            for doc in extra_json_docs[:5]:  # Show first 5
                print(f"   - {doc[:70]}...")
            issues.append(f"{len(extra_json_docs)} extra documents")
        
        if missing_articles:
            print(f"\n⚠️  Documents with Missing Articles: {len(missing_articles)}")
            for item in missing_articles:
                print(f"   - {item['document'][:50]}... (missing {item['missing_count']})")
            issues.append(f"{len(missing_articles)} docs with missing articles")
        
        if extra_articles:
            print(f"\n⚠️  Documents with Extra Articles: {len(extra_articles)}")
            for item in extra_articles:
                print(f"   - {item['document'][:50]}... (extra {item['extra_count']})")
            issues.append(f"{len(extra_articles)} docs with extra articles")
        
        # Overall status
        if not issues:
            print(f"\n✅ VERIFICATION PASSED - All documents and articles accounted for!")
            status = 'PASS'
        else:
            print(f"\n⚠️  VERIFICATION WARNINGS - {len(issues)} issue(s) found")
            status = 'WARN'
        
        return {
            'status': status,
            'txt_docs': len(txt_docs),
            'json_docs': len(json_docs),
            'txt_articles': total_txt_articles,
            'json_articles': total_json_articles,
            'issues': issues
        }
    
    def verify_all(self):
        """Verify all TXT/JSON file pairs."""
        
        print("="*70)
        print("🔍 JOURNAL OFFICIEL EXTRACTION VERIFICATION")
        print("="*70)
        print(f"TXT Directory:  {self.txt_dir}")
        print(f"JSON Directory: {self.json_dir}")
        print("="*70)
        
        # Get all TXT files
        txt_files = list(self.txt_dir.glob("*.txt"))
        
        if not txt_files:
            print(f"\n❌ No TXT files found in {self.txt_dir}")
            return
        
        print(f"\nFound {len(txt_files)} TXT file(s) to verify\n")
        
        # Verify each file
        all_results = {}
        for txt_file in txt_files:
            result = self.verify_file(txt_file)
            all_results[txt_file.name] = result
        
        # Overall summary
        print(f"\n{'='*70}")
        print("🎯 OVERALL SUMMARY")
        print(f"{'='*70}")
        
        total_txt_docs = sum(r['txt_docs'] for r in all_results.values() if 'txt_docs' in r)
        total_json_docs = sum(r['json_docs'] for r in all_results.values() if 'json_docs' in r)
        total_txt_articles = sum(r['txt_articles'] for r in all_results.values() if 'txt_articles' in r)
        total_json_articles = sum(r['json_articles'] for r in all_results.values() if 'json_articles' in r)
        
        passed = sum(1 for r in all_results.values() if r.get('status') == 'PASS')
        warned = sum(1 for r in all_results.values() if r.get('status') == 'WARN')
        errors = sum(1 for r in all_results.values() if r.get('status') == 'ERROR')
        
        print(f"\nFiles Processed: {len(txt_files)}")
        print(f"  ✅ Passed:  {passed}")
        print(f"  ⚠️  Warnings: {warned}")
        print(f"  ❌ Errors:  {errors}")
        
        print(f"\nTotal Documents:")
        print(f"  TXT:  {total_txt_docs}")
        print(f"  JSON: {total_json_docs}")
        print(f"  Match: {'✅' if total_txt_docs == total_json_docs else '❌'}")
        
        print(f"\nTotal Articles:")
        print(f"  TXT:  {total_txt_articles}")
        print(f"  JSON: {total_json_articles}")
        print(f"  Match: {'✅' if total_txt_articles == total_json_articles else '⚠️ '}")
        
        if total_txt_articles != total_json_articles:
            diff = total_txt_articles - total_json_articles
            if diff > 0:
                print(f"  → {diff} article(s) missing in JSON")
            else:
                print(f"  → {-diff} extra article(s) in JSON")
        
        # Final verdict
        print(f"\n{'='*70}")
        if errors == 0 and warned == 0:
            print("✅ ALL VERIFICATIONS PASSED!")
            print("   Your extraction is complete and accurate.")
        elif errors == 0:
            print("⚠️  VERIFICATION COMPLETED WITH WARNINGS")
            print("   Some minor discrepancies found - review above.")
        else:
            print("❌ VERIFICATION FAILED")
            print("   Critical errors found - review above.")
        print(f"{'='*70}")


def main():
    """Main entry point."""
    verifier = ExtractionVerifier()
    verifier.verify_all()


if __name__ == "__main__":
    main()
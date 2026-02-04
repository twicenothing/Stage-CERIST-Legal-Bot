#!/usr/bin/env python3
"""
Smart PDF to TXT converter using PyMuPDF (Fitz).
Handles single/double columns, preserves reading order, and PROTECTS TABLES.
Compatible with JORADP Cleaning Pipeline.
"""

import os
import sys
from pathlib import Path
import numpy as np
from typing import List, Tuple

# Try importing dependencies
try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ Missing dependency: fitz (PyMuPDF). Install with: pip install pymupdf")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ Missing dependency: numpy. Install with: pip install numpy")
    sys.exit(1)


class SmartPDFExtractor:
    """Extract text from PDFs with intelligent column detection."""
    
    def __init__(self, pdf_dir: str = None, txt_dir: str = None):
        # Get the project root directory
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        
        if pdf_dir is None:
            pdf_dir = project_root / "data" / "pdfs"
        if txt_dir is None:
            txt_dir = project_root / "data" / "txt"
        
        self.pdf_dir = Path(pdf_dir)
        self.txt_dir = Path(txt_dir)
        self.txt_dir.mkdir(parents=True, exist_ok=True)

    def has_center_crossing_table(self, page: fitz.Page) -> bool:
        """
        [NEW] Checks if there is a table that crosses the vertical center of the page.
        If yes, we should NOT split the page into columns.
        """
        try:
            # PyMuPDF's built-in table finder
            tables = page.find_tables()
            if not tables.tables:
                return False
                
            page_width = page.rect.width
            mid_x = page_width / 2
            
            for table in tables:
                x0, y0, x1, y1 = table.bbox
                # If table starts before the middle and ends after the middle
                if x0 < mid_x and x1 > mid_x:
                    return True
            return False
        except Exception:
            # If table detection fails, assume no table to be safe
            return False
    
    def detect_columns(self, page: fitz.Page, threshold: float = 0.4) -> bool:
        """
        Detect if a page has double columns by analyzing text block positions.
        Returns True if double-column layout detected.
        """
        # --- NEW LOGIC: TABLE SAFETY CHECK ---
        # If a table spans the whole width, force Single Column mode.
        if self.has_center_crossing_table(page):
            return False
        # -------------------------------------

        blocks = page.get_text("dict")["blocks"]
        if not blocks: return False
        
        # Get text blocks only
        text_blocks = [b for b in blocks if b["type"] == 0]
        if len(text_blocks) < 2: return False
        
        page_width = page.rect.width
        x_centers = []
        
        for block in text_blocks:
            x0, y0, x1, y1 = block["bbox"]
            x_center = (x0 + x1) / 2
            x_centers.append(x_center)
        
        if not x_centers: return False
        
        # Normalize coordinates
        x_centers = np.array(x_centers) / page_width
        
        # Look for clustering (Left side < 0.5, Right side > 0.5)
        left_blocks = sum(1 for x in x_centers if x < 0.45)
        right_blocks = sum(1 for x in x_centers if x > 0.55)
        
        # If we have distinct blocks on both sides
        if left_blocks > 0 and right_blocks > 0:
            sorted_x = sorted(x_centers)
            gaps = [sorted_x[i+1] - sorted_x[i] for i in range(len(sorted_x)-1)]
            
            if gaps:
                max_gap = max(gaps)
                max_gap_idx = gaps.index(max_gap)
                gap_position = sorted_x[max_gap_idx]
                
                # If there is a large gap in the middle (0.3 - 0.7)
                if 0.3 < gap_position < 0.7 and max_gap > 0.1: # Threshold tuned for text
                    return True
        
        return False
    
    def extract_single_column(self, page: fitz.Page) -> str:
        """Extract text from a single-column page."""
        return page.get_text("text")
    
    def extract_double_column(self, page: fitz.Page) -> str:
        """
        Extract text from a double-column page in proper reading order.
        Handles 'Full Width' headers (Titles) that span both columns.
        """
        blocks = page.get_text("dict")["blocks"]
        text_blocks = [b for b in blocks if b["type"] == 0]
        
        if not text_blocks: return ""
        
        page_width = page.rect.width
        mid_point = page_width / 2
        
        # Categories
        full_width_blocks = []
        left_column = []
        right_column = []
        
        for block in text_blocks:
            x0, y0, x1, y1 = block["bbox"]
            
            # Extract text
            block_text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "") + " "
                block_text += "\n"
            
            # CHECK: Does this block span across the center? (e.g. A Title)
            # If it covers > 70% of the page width, treat as full width
            block_width = x1 - x0
            if block_width > (page_width * 0.7):
                full_width_blocks.append((y0, block_text))
                continue
                
            # Otherwise, sort into columns
            x_center = (x0 + x1) / 2
            if x_center < mid_point:
                left_column.append((y0, block_text))
            else:
                right_column.append((y0, block_text))
        
        # Sort all lists by Vertical (Y) position
        full_width_blocks.sort(key=lambda x: x[0])
        left_column.sort(key=lambda x: x[0])
        right_column.sort(key=lambda x: x[0])
        
        # --- STITCHING STRATEGY ---
        text = ""
        
        # 1. Add Full Width Blocks (Titles)
        for _, t in full_width_blocks: text += t + "\n"
        
        # 2. Add Left Column
        for _, t in left_column: text += t
        
        # 3. Add Right Column
        for _, t in right_column: text += t
        
        return text
    
    def extract_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF, injecting Page Markers for downstream cleaning."""
        try:
            doc = fitz.open(pdf_path)
            full_text = []
            
            for i, page in enumerate(doc):
                page_num = i + 1
                
                # 1. Add The Marker (CRITICAL for clean_txt.py)
                full_text.append(f"==== PAGE {page_num} ====")
                
                # 2. Detect & Extract
                is_double = self.detect_columns(page)
                
                if is_double:
                    # print(f"  [Page {page_num}] 2-Column Detected")
                    page_text = self.extract_double_column(page)
                else:
                    # print(f"  [Page {page_num}] Single Column / Table")
                    page_text = self.extract_single_column(page)
                
                full_text.append(page_text)
                full_text.append("\n") # Spacer
            
            doc.close()
            return "\n".join(full_text)
        
        except Exception as e:
            print(f"Error extracting {pdf_path.name}: {str(e)}")
            return ""
    
    def process_all_pdfs(self):
        if not self.pdf_dir.exists():
            print(f"Error: PDF directory '{self.pdf_dir}' missing.")
            return
            
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        if not pdf_files:
            print("No PDF files found.")
            return
            
        print(f"🚀 Processing {len(pdf_files)} PDFs using PyMuPDF...")
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}")
            text = self.extract_pdf(pdf_path)
            
            if text:
                txt_path = self.txt_dir / (pdf_path.stem + ".txt")
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"   ✅ Saved: {txt_path.name}")
            else:
                print(f"   ❌ Failed to extract text.")

def main():
    extractor = SmartPDFExtractor()
    extractor.process_all_pdfs()

if __name__ == "__main__":
    main()
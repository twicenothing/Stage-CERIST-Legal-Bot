import pdfplumber
import os

# --- CONFIGURATION ---
# This ensures we work relative to the script location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
TXT_DIR = os.path.join(BASE_DIR, "data", "txt")

def is_two_column(page):
    """
    Analyzes the page layout to decide if it is 2-column or 1-column.
    """
    width = page.width
    height = page.height
    
    # Check the center strip (45% to 55% of page width)
    center_start = width * 0.45
    center_end = width * 0.55
    
    # Get all words
    words = page.extract_words()
    
    if not words: return False # Empty page
        
    # Count words that cross the "Danger Zone" (the center gutter)
    words_in_center = 0
    for word in words:
        word_x0 = word['x0']
        word_x1 = word['x1']
        
        # If the word overlaps the center strip
        if (word_x0 < center_end) and (word_x1 > center_start):
            words_in_center += 1
            
    # If fewer than 3 words touch the center, it's a 2-column page.
    return words_in_center < 3

def extract_smart(pdf_filename):
    pdf_path = os.path.join(PDF_DIR, pdf_filename)
    txt_filename = pdf_filename.replace(".pdf", ".txt")
    txt_path = os.path.join(TXT_DIR, txt_filename)
    
    full_text = []
    print(f"📄 Processing: {pdf_filename}...")
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                
                # --- DECISION LOGIC ---
                if is_two_column(page):
                    # ✂️ SPLIT MODE (Decrees, Text)
                    width = page.width
                    height = page.height
                    
                    # Crop Left and Right
                    left_bbox = (0, 0, width / 2, height)
                    right_bbox = (width / 2, 0, width, height)
                    
                    left_text = page.crop(left_bbox).extract_text(x_tolerance=2) or ""
                    right_text = page.crop(right_bbox).extract_text(x_tolerance=2) or ""
                    
                    # Stitch: Left then Right
                    page_content = left_text + "\n" + right_text
                    
                else:
                    # 📄 STANDARD MODE (Tables, Sommaire, Old Scans)
                    page_content = page.extract_text(x_tolerance=2) or ""

                # Add Markers
                full_text.append(f"==== PAGE {page_num} ====")
                full_text.append(page_content)
                full_text.append("\n") # Breathable space

        # OVERWRITE the existing text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_text))
        
        print(f"    ✅ Overwrote: {txt_filename}")

    except Exception as e:
        print(f"    ❌ Error processing {pdf_filename}: {e}")

def main():
    # Ensure directory exists
    if not os.path.exists(TXT_DIR): os.makedirs(TXT_DIR)
    
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ No PDFs found in data/pdfs/")
        return

    print(f"🚀 Starting Smart Extraction on {len(pdf_files)} files...")
    print(f"📂 Output Folder: {TXT_DIR}")
    
    for pdf_file in pdf_files:
        extract_smart(pdf_file)

if __name__ == "__main__":
    main()
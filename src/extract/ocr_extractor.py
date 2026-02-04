import os
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# --- CONFIGURATION (Keep your paths) ---
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR'
pytesseract.pytesseract.tesseract_cmd = os.path.join(TESSERACT_PATH, 'tesseract.exe')
os.environ['TESSDATA_PREFIX'] = os.path.join(TESSERACT_PATH, 'tessdata')

# Update this path if needed
POPPLER_PATH = r'C:\poppler-25.12.0\Library\bin' 

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/pdfs"))
TXT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/txt"))

def process_pdf(pdf_path, output_path):
    print(f"Processing: {os.path.basename(pdf_path)}")
    try:
        # Get all pages (skipping page 1 as requested)
        images = convert_from_path(pdf_path, dpi=300, first_page=2, poppler_path=POPPLER_PATH)
        
        full_text = []
        
        for i, img in enumerate(images):
            # Standard OCR without manual cropping
            # Tesseract usually auto-detects text blocks
            text = pytesseract.image_to_string(img, lang='fra')
            
            full_text.append(f"--- Page {i+2} ---")
            full_text.append(text)
            full_text.append("\n" + "="*50 + "\n") # Visual separator

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(full_text))
            
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)

    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))
            process_pdf(pdf_path, txt_path)

if __name__ == "__main__":
    main()
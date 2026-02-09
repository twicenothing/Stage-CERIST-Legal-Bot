import os
from pypdf import PdfReader  # pip install pypdf (recommended) or PyPDF2

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Input folder
PDF_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/pdfs"))
# Output folder
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/txt"))

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Processing PDFs from: {PDF_DIR}")
print(f"Outputting TXT files to: {OUTPUT_DIR}\n")

# Process each PDF file
for filename in os.listdir(PDF_DIR):
    if filename.lower().endswith('.pdf'):
        pdf_path = os.path.join(PDF_DIR, filename)
        try:
            reader = PdfReader(pdf_path)
            full_text = ""

            # Skip first page (index 0), start from page 1 (which is actual page 2)
            for page_num in range(1, len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    # Add page header with actual page number (page_num + 1)
                    full_text += f"--- Page {page_num + 1} ---\n\n"
                    full_text += page_text + "\n\n"

            # Save to .txt file (same base name)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(OUTPUT_DIR, txt_filename)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(full_text)

            print(f"✅ Processed: {filename} → {txt_filename} ({len(reader.pages)-1} pages extracted)")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

print("\nAll PDFs processed!")
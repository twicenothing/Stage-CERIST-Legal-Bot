from pathlib import Path
import fitz  # PyMuPDF
from ollama import Client

# -----------------------------
# Config
# -----------------------------
OLLAMA_HOST = "http://127.0.0.1:11434"
MODEL_NAME = "mistral-small3.1:latest"

pdf_path = Path("../../data/pdf/2005/F2005001.pdf")   # change if needed
page_number = 1                                 # 1-based page number
output_image = Path("../../data/pdf/2005/F2_page1.png")

# -----------------------------
# Convert PDF page to PNG
# -----------------------------
doc = fitz.open(pdf_path)
page = doc.load_page(page_number - 1)  # PyMuPDF uses 0-based indexing

# Zoom factor for better readability
zoom = 3.0
matrix = fitz.Matrix(zoom, zoom)

pix = page.get_pixmap(matrix=matrix, alpha=False)
pix.save(output_image)

doc.close()

print(f"✅ Rendered page {page_number} to: {output_image}")

# -----------------------------
# Send image to Ollama
# -----------------------------
client = Client(host=OLLAMA_HOST)

res = client.chat(
    model=MODEL_NAME,
    messages=[
        {
            "role": "user",
            "content": "Lis cette page du Journal Officiel et dis-moi ce qu'elle contient.",
            "images": [str(output_image)],
        }
    ],
    options={
        "temperature": 0.0,
    },
)

print("\n📄 Model response:\n")
print(res["message"]["content"])
import os
import json
import re

# --- CONFIGURATION ---
RAW_TEXT_DIR = "../../data/txt"       # Tes fichiers TXT issus de l'OCR
OUTPUT_JSON_DIR = "../../data/json_llm_extracted" # On garde le même dossier de sortie pour pas casser la suite
CHUNK_SIZE = 1500  # Nombre de caractères par morceau (environ 300-400 mots)
OVERLAP = 200      # Chevauchement pour ne pas couper une phrase en deux

def clean_text(text):
    """Nettoyage léger : on garde tout, on enlève juste les sauts de ligne bizarres"""
    # Remplace les sauts de ligne multiples par un seul
    text = re.sub(r'\n+', '\n', text)
    # Enlève les caractères non-imprimables bizarres
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]+', ' ', text)
    return text.strip()

def create_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        
        # Petit ajustement pour ne pas couper au milieu d'un mot
        if end < text_len:
            # On cherche le dernier espace pour couper proprement
            last_space = text.rfind(' ', start, end)
            if last_space != -1:
                end = last_space
        
        chunk = text[start:end]
        chunks.append(chunk)
        
        # On avance, mais on recule un peu (overlap) pour le contexte
        start = end - overlap
        
        # Sécurité pour éviter boucle infinie si overlap >= chunk_size
        if start >= end:
            start = end
            
    return chunks

def main():
    if not os.path.exists(OUTPUT_JSON_DIR):
        os.makedirs(OUTPUT_JSON_DIR)

    files = [f for f in os.listdir(RAW_TEXT_DIR) if f.endswith(".txt")]
    print(f"📦 Démarrage du découpage sécurisé sur {len(files)} fichiers...")

    for filename in files:
        file_path = os.path.join(RAW_TEXT_DIR, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            clean_content = clean_text(raw_text)
            text_chunks = create_chunks(clean_content, CHUNK_SIZE, OVERLAP)
            
            # Structure JSON compatible avec ton indexeur actuel
            json_output = {
                "source_file": filename,
                "total_chunks": len(text_chunks),
                "documents": [] # Ton indexeur s'attend peut-être à une liste
            }

            for i, chunk in enumerate(text_chunks):
                # On crée un "faux" objet structuré pour que ton indexeur soit content
                doc_struct = {
                    "id": f"{filename}_chunk_{i}",
                    "title": f"Extrait {i+1} de {filename}", # Titre générique
                    "content": chunk, # <-- LE PLUS IMPORTANT : LE TEXTE BRUT
                    "page_number": "N/A" # Difficile à savoir sans OCR complexe
                }
                json_output["documents"].append(doc_struct)

            # Sauvegarde
            output_filename = filename.replace(".txt", ".json")
            output_path = os.path.join(OUTPUT_JSON_DIR, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_output, f, ensure_ascii=False, indent=4)
                
            print(f"✅ {filename} -> {len(text_chunks)} morceaux générés.")

        except Exception as e:
            print(f"❌ Erreur sur {filename}: {e}")

    print("\n🎉 Terminé ! Toutes les données sont préservées.")
    print("👉 Maintenant, relance 'indexer.py' pour mettre à jour ChromaDB.")

if __name__ == "__main__":
    main()
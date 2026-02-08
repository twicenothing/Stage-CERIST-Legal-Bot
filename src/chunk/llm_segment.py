import os
import json
import ollama
import re
from tqdm import tqdm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/cleaned"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/json_llm_extracted"))

# On réduit un peu la taille pour éviter que le modèle s'embrouille
CHUNK_SIZE = 3500 
OVERLAP = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_with_tags(text_chunk):
    # Prompt simple et robuste : pas de accolades, juste des balises
    prompt = f"""
    Tu es un extracteur juridique. Analyse ce texte du Journal Officiel.
    
    TACHE:
    Sépare chaque texte juridique (Décret, Arrêté) distinct.
    
    FORMAT OBLIGATOIRE POUR CHAQUE TEXTE :
    [[START]]
    ID: (Copie le titre exact ici, ex: Décret exécutif n° 25-10...)
    CONTENT: (Copie tout le contenu du texte ici)
    [[END]]
    
    Règles :
    1. Ne change pas le texte du contenu.
    2. Si un titre est cassé (ex: "k Décret"), répare-le dans la ligne ID.
    
    TEXTE A TRAITER :
    {text_chunk}
    """
    
    try:
        response = ollama.chat(model='mistral', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        print(f"❌ Erreur Ollama : {e}")
        return ""

def parse_tagged_response(response_text):
    """Découpe le texte brut grâce aux balises [[START]] et [[END]]"""
    documents = []
    
    # On découpe par blocs [[START]] ... [[END]]
    # Le flag DOTALL permet au point (.) de capturer aussi les retours à la ligne
    pattern = re.compile(r"\[\[START\]\]\s*ID:\s*(.*?)\s*CONTENT:\s*(.*?)\s*\[\[END\]\]", re.DOTALL)
    
    matches = pattern.findall(response_text)
    
    for match in matches:
        title = match[0].strip()
        content = match[1].strip()
        
        # Petit nettoyage si le modèle bafouille
        if len(title) > 5 and len(content) > 20:
            documents.append({
                "official_id": title,
                "text": content
            })
            
    return documents

def process_file(file_path):
    filename = os.path.basename(file_path)
    output_path = os.path.join(OUTPUT_DIR, filename.replace(".txt", ".jsonl"))

    if os.path.exists(output_path):
        print(f"⏩ Déjà fait : {filename}")
        return

    print(f"📄 Traitement : {filename}")
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    extracted_count = 0
    start = 0
    
    # On ouvre le fichier de sortie en mode append au cas où
    with open(output_path, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=len(full_text), unit="char")
        
        while start < len(full_text):
            end = min(start + CHUNK_SIZE, len(full_text))
            chunk = full_text[start:end]
            
            # 1. Appel LLM
            raw_response = extract_with_tags(chunk)
            
            # 2. Parsing Regex (Indestructible)
            docs = parse_tagged_response(raw_response)
            
            # 3. Debug : Si rien trouvé, on affiche un petit warning
            if not docs and len(raw_response) > 100:
                # Parfois le modèle ne met pas les balises, on loggue juste pour info
                pass 

            for doc in docs:
                final_doc = {
                    "source": filename,
                    "official_id": doc['official_id'],
                    "text": doc['text'],
                    "journal_date": "Inconnue", # Sera rempli par l'extracteur de date si besoin
                    "type": "Decret" if "Décret" in doc['official_id'] else "Arrêté"
                }
                f_out.write(json.dumps(final_doc, ensure_ascii=False) + "\n")
                extracted_count += 1
            
            start += (CHUNK_SIZE - OVERLAP)
            pbar.update(CHUNK_SIZE - OVERLAP)
        
        pbar.close()
    
    print(f"   ✅ {extracted_count} documents extraits.")

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]
    print(f"🚀 Lancement RobustExtractor sur {len(files)} fichiers...")
    for f in files:
        process_file(os.path.join(INPUT_DIR, f))

if __name__ == "__main__":
    main()
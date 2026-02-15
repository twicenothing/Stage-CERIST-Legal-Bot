import re
import os

# Chemin direct vers le fichier problématique
FILE_PATH = "data/txt/F2025007.txt"

def clean_text(text):
    # On garde la même logique que ton script principal
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]+', ' ', text)
    return text.strip()

def analyze_chunking():
    if not os.path.exists(FILE_PATH):
        print(f"❌ Fichier introuvable : {FILE_PATH}")
        return

    print(f"📄 Lecture de {FILE_PATH}...")
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        raw = f.read()
    
    text = clean_text(raw)
    
    # Le Regex utilisé dans ton safe_chunker
    header_pattern = re.compile(
        r'(Décret\s+(?:présidentiel|exécutif)|Arrêté|Décision)\s+(?:n[°o\.]?)?\s*(\d+[-‐‑]\d+|\d{1,4}(?!\d))', 
        re.IGNORECASE
    )

    print("\n🔍 --- TEST DE DÉTECTION ---")
    matches = list(header_pattern.finditer(text))
    
    for i, m in enumerate(matches):
        match_str = m.group(0)
        start = m.start()
        
        # On regarde le contexte AVANT (c'est là que ça se joue)
        context_before = text[max(0, start-50):start]
        context_clean = context_before.lower().replace("\n", " ").strip()
        
        print(f"\n🔹 Match #{i+1} trouvé : '{match_str}'")
        print(f"   📍 Position : {start}")
        print(f"   👀 Contexte avant (50 chars) : [{context_clean}]")
        
        # Simulation de ta logique Anti-Visa
        is_visa = False
        if "vu le" in context_clean or "vu l'" in context_clean or "application du" in context_clean:
            is_visa = True
            print("   🛡️  FILTRE ACTIVÉ : C'est un Visa (sera ignoré).")
        else:
            print("   ✅ ACCEPTÉ : C'est un nouveau décret.")

        # FOCUS SUR LE PROBLEME (24-10)
        if "24-10" in match_str:
            print("   👉 ANALYSE CRITIQUE SUR 24-10 :")
            if not is_visa:
                print("   ⚠️  ALERTE : Ce décret aurait dû être ignoré mais il est accepté !")
                print("   ⚠️  Cause probable : Le texte avant n'est pas exactement 'vu le' (peut-être 'vu  le' ou 'v u le' ?)")

if __name__ == "__main__":
    analyze_chunking()
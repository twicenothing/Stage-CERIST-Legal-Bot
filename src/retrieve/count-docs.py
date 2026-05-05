import chromadb
from dotenv import load_dotenv
import os

load_dotenv()
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")

# Connexion à la base
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# Récupération du nombre total de vecteurs (chunks)
total_chunks = collection.count()

print(f"📊 Ta base de données contient exactement {total_chunks} chunks.")
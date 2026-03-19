import chromadb
from sentence_transformers import SentenceTransformer
import requests
import time  

# --- 1. CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 3  

OLLAMA_URL = "http://127.0.0.1:11435/api/chat"  
LLM_MODEL = "deepseek-r1:32b"  

# (Dataset)
dataset = {
    "samples" : [
        {"id": 1, "question" : "Pourquoi la Cour constitutionnelle a-t-elle rejeté la saisine des parlementaires concernant l’interprétation de l’article 158 de la Constitution ?", "answer" : "La Cour a rejeté la demande sur le fond car elle estime que les dispositions de l'article 158 sont déjà parfaitement claires, rigides et dénuées de toute ambiguïté. Interpréter un texte déjà explicite n'était donc pas justifié.", "source" : "F2025009.json", "parent_title" : """Avis n° 01 A.C.C/I.C/25 du 30 Rajab 1446 correspondant au 30 janvier 2025 """},
        {"id": 2, "question" : "Quel est le danger potentiel souligné par les juges s'ils acceptaient d'expliquer une disposition constitutionnelle qui est déjà évidente ?", "answer" : "La Cour souligne qu'une interprétation extensive de dispositions claires risquerait d'entraîner une modification indirecte de la Constitution en dehors des voies légales, créant ainsi une forme de révision constitutionnelle parallèle par le juge.", "source" : "F2025009.json", "parent_title" : """Avis n° 01 A.C.C/I.C/25 du 30 Rajab 1446 correspondant au 30 janvier 2025 relatif à l’interprétation des dispositions de l’article 158 de la Constitution. """},
        {"id": 3, "question" : "Quel est l'impact pratique de l'arrêté de janvier 2025 sur la gestion quotidienne des fonds alloués à l'Inspection générale des finances, et quelle capacité d'action spécifique est transférée à M. Saïd Touakni ?", "answer" : "Ce texte permet de décentraliser et de fluidifier l'exécution budgétaire de l'Inspection générale des finances (IGF). Il autorise M. Saïd Touakni à agir légalement au nom du ministre des Finances pour valider et signer tous les documents liés aux dépenses (y compris les ordres de paiement), mais uniquement dans le périmètre strict du budget propre à l'IGF.", "source" : "F2025009.json","parent_title":"""Arrêté du 30 Rajab 1446 correspondant au 30 janvier 2025"""},
        {"id": 4, "question" : "À quel programme, sous-programme et titre exacts est applicable le montant ouvert de trente-neuf millions de dinars (39.000.000 DA) pour le portefeuille du ministère des transports ?", "answer" : "Ce montant est applicable au programme « Administration générale », au sous-programme « Soutien administratif » et au titre 2 « Dépenses de fonctionnement des services ».", "source" : "F2025005.json", "parent_title": """Décret présidentiel n° 24-439 du 29 Joumada Ethania 1446"""},
        {"id": 5, "question" : "Quelle est la nouvelle échéance accordée aux groupements d'agriculteurs pour se mettre en règle avec la législation de 1996 qui les régit ?", "answer" : "Le délai de mise en conformité pour les coopératives agricoles et leurs unions (vis-à-vis du décret de 1996) a été repoussé au 31 décembre 2025.", "source" : "F2025009.json", "parent_title" : """Décret exécutif n° 25-73 du 11 Chaâbane 1446 correspondant au 10 février 2025"""},
        {"id": 6, "question" : "Au sein de la commission sectorielle chargée de la tutelle pédagogique sur l'école supérieure de la sécurité sociale, que se passe-t-il si la plupart des membres sont absents lors d'une réunion, et comment les décisions sont-elles tranchées en cas d'égalité des votes ?", "answer" : "Si le quorum exigé des deux tiers (2/3) des membres n'est pas atteint lors de la première réunion, une seconde session doit être convoquée dans les huit (8) jours suivants. Lors de cette seconde réunion, la commission peut valablement délibérer quel que soit le nombre de personnes présentes. Si les votes sont partagés à égalité, c'est la voix du président qui est prépondérante pour trancher.", "source" : "F2025005.json", "parent_title":"""Arrêté interministériel du 7 Joumada Ethania 1446 correspondant au 9 décembre 2024"""},
        {"id": 7, "question" : "La liste des membres du Conseil national économique, social et environnemental (CNESE) nommés en janvier 2025 est-elle complète, et quelle est la durée de leur mandat ?", "answer" : "Non, la liste n'est pas exhaustive et les membres restants seront nommés ultérieurement (selon l'article 3). Pour les membres déjà désignés dans ce texte, la durée de leur mandat est fixée à quatre (4) ans (selon l'article 1er).", "source" : "F2025005.json", "parent_title" : """Décision du 14 Rajab 1446 correspondant au 14 janvier 2025"""},
        {"id": 8, "question" : "De quelles manières exactes le ministre de la jeunesse doit-il intervenir en faveur de la jeunesse algérienne établie hors du pays, selon le décret de février 2025 fixant ses attributions ?", "answer" : "Le ministre a trois responsabilités principales envers la communauté nationale à l'étranger, réparties dans différents domaines d'action :\nIdentité (Art. 2) : Il doit proposer et développer des mesures pour renforcer leur esprit d'appartenance nationale.\nStratégie (Art. 3) : Il est chargé d'élaborer une stratégie d'action spécifique à leur profit, en coordination avec d'autres secteurs ministériels.\nRayonnement et Talents (Art. 7) : Dans le cadre des relations internationales, il doit mettre en œuvre des mesures pour valoriser les compétences et les talents des jeunes issus de cette communauté.", "source" : "F2025010.json","parent_title":"""Décret exécutif n° 25-74 du 12 Chaâbane 1446 correspondant au 11 février 2025"""}
    ]
}

# --- 2. FONCTION LLM ---
def call_llm(prompt_text, system_instruction=""):
    messages = []
    if system_instruction.strip():
        messages.append({"role": "system", "content": system_instruction})
    
    messages.append({"role": "user", "content": prompt_text})

    payload = {
        "model": LLM_MODEL, 
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192  
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status() 
        data = response.json()
        return data["message"]["content"]
    except Exception as e:
        print(f"❌ Erreur de connexion à Ollama : {e}")
        return "" 

# --- 3. SYSTÈME RAG (NOUVEAU PROMPT CHAIN-OF-THOUGHT) ---
def retrieve_and_generate(question, collection, model):
    query_emb = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K
    )
    
    retrieved_docs = results['documents'][0]
    
    formatted_context = ""
    for i, doc in enumerate(retrieved_docs):
        formatted_context += f"Document [{i+1}]:\n{doc}\n\n"
    
    system_prompt = """Tu es un assistant juridique expert en droit administratif algérien. Ta seule et unique mission est de répondre aux questions de l'utilisateur en te basant STRICTEMENT et EXCLUSIVEMENT sur les documents de référence fournis.

MÉTHODOLOGIE OBLIGATOIRE :
Tu es un modèle de raisonnement. Tu vas automatiquement utiliser ta balise <think> pour réfléchir avant de répondre. 
Dans cette réflexion :
1. Identifie les concepts clés de la question.
2. Cherche attentivement ces concepts dans les documents fournis.
3. Détermine si l'information s'y trouve réellement.

RÈGLES POUR LA RÉPONSE FINALE (Après la balise <think>) :
1. RÈGLE CRITIQUE : Si ton analyse conclut que la réponse ne figure pas dans les documents, ta réponse finale DOIT être EXACTEMENT : "Les documents fournis ne contiennent pas cette information."
2. Si la réponse s'y trouve, réponds de manière directe, factuelle et précise.
3. Cite toujours le numéro de l'Article ou l'intitulé du texte de loi justifiant ta réponse."""

    user_prompt = f"""Voici les documents de référence :

{formatted_context}

Question de l'utilisateur : {question}

Réponse :"""
    
    answer = call_llm(user_prompt, system_instruction=system_prompt)
    return answer, formatted_context

# --- 4. EXÉCUTION DU TEST MANUEL ---
def run_manual_test():
    print("🔄 Initialisation de ChromaDB et BGE-M3...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    print(f"\n🚀 Démarrage du test manuel avec {LLM_MODEL} (TOP_K={TOP_K})...\n")
    
    for sample in dataset["samples"][:8]:
        q_id = sample["id"]
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print("\n" + "="*100)
        print(f"🛑 QUESTION {q_id}")
        print("="*100)
        print(f"📝 {question}")
        
        # RAG
        generated_answer, full_context = retrieve_and_generate(question, collection, embedding_model)
        
        print("\n" + "-"*100)
        print(f"📚 CONTEXTE RÉCUPÉRÉ (TOP {TOP_K}) :") 
        print("-"*100)
        # 👇 I UNCOMMENTED THIS LINE
        print(full_context)
        
        print("\n" + "-"*100)
        print("🎯 RÉPONSE DE RÉFÉRENCE (GROUND TRUTH) :")
        print("-"*100)
        print(ground_truth)
        
        print("\n" + "-"*100)
        print("🤖 RÉPONSE GÉNÉRÉE PAR LE MODÈLE (AVEC ANALYSE) :")
        print("-"*100)
        print(generated_answer)
        print("\n")
        
        time.sleep(1)
        
        # 👇 I ADDED THIS BREAK SO IT STOPS AFTER QUESTION 1
        break 

if __name__ == "__main__":
    run_manual_test()
import json
import re
import ollama
import chromadb
import string
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
OLLAMA_MODEL = "mistral"

# Les ID des questions qui ont échoué ou perdu des points au test précédent
FAILED_IDS = [1, 3, 5, 6, 15, 16, 17, 18]

# (Assure-toi d'avoir ton combined_golden_dataset complet ici)
combined_golden_dataset = {
    "dataset_name": "Evaluation RAG - Journaux Officiels Algériens",
    "samples": [
        {
            "id": 1,
            "type": "mot-clé",
            "question": "Quel est le montant exact en dinars algériens (DA) annulé pour les 'Dépenses imprévues' en vertu du décret n° 24-432 ?",
            "expected_answer": "En vertu du décret n° 24-432, les montants annulés pour les 'Dépenses imprévues' (imputables au titre 7) sont de 29.711.200.000 DA en autorisations d'engagement et de 4.500.000.000 DA en crédits de paiement."
        },
        {
            "id": 2,
            "type": "mot-clé",
            "question": "Quelles sont les deux personnes désignées comme membres de la Cour constitutionnelle par le décret n° 25-01 ?",
            "expected_answer": "Les deux membres désignés sont Mme Leila ASLAOUI et M. Mosbah MENAS."
        },
        {
            "id": 3,
            "type": "sémantique",
            "question": "D'après les transferts de crédits du décret n° 24-432, quels secteurs d'infrastructures ont été les principaux bénéficiaires des crédits nouvellement ouverts ?",
            "expected_answer": "Les principaux bénéficiaires ont été les infrastructures routières et autoroutières (spécifiquement l'entretien routier avec 22.896.200.000 DA) et les infrastructures ferroviaires (2.300.000.000 DA)."
        },
        {
            "id": 4,
            "type": "mot-clé",
            "question": "Identifiez les sociétés internationales qui ont signé le contrat d'exploitation d'hydrocarbures pour le périmètre 'Menzel Lejmat' le 15 juin 2023.",
            "expected_answer": "Les sociétés sont 'PT Pertamina Algeria Eksplorasi Produksi' et 'REPSOL EXPLORACION 405A, S.A.'."
        },
        {
            "id": 5,
            "type": "sémantique",
            "question": "Comment le décret n° 25-03 a-t-il modifié la durée et la structure des mandats des membres du Conseil National Economique, Social et Environnemental (CNESE) ?",
            "expected_answer": "Le décret établit que les membres du CNESE sont désignés pour un mandat de quatre ans, renouvelable une seule fois. De plus, les membres du bureau et les présidents des commissions permanentes sont désormais élus pour un mandat de deux ans non renouvelable."
        },
        {
            "id": 6,
            "type": "mot-clé",
            "question": "Quelle est la catégorie et l'indice minimal pour le grade de 'Professeur émérite de l’enseignement secondaire' ?",
            "expected_answer": "Selon la grille de classification du personnel enseignant, un 'Professeur émérite de l’enseignement secondaire' est classé à la catégorie 17 avec un indice minimal de 962."
        },
        {
            "id": 7,
            "type": "mot-clé",
            "question": "En vertu de l'article 17, quelle est la durée maximale et la fréquence du 'congé de mobilité professionnelle' pour les enseignants ?",
            "expected_answer": "Le congé de mobilité professionnelle est accordé une seule fois au cours de la carrière de l'enseignant pour une durée maximale d'un (1) an, non renouvelable."
        },
        {
            "id": 8,
            "type": "sémantique",
            "question": "Quelles sont les conditions physiques et de santé spécifiques requises pour les candidats au recrutement dans les grades d'enseignement du secteur de l'éducation nationale ?",
            "expected_answer": "Les candidats doivent avoir l'aptitude d'accomplir leurs missions, ce qui inclut spécifiquement d'avoir une élocution, une ouïe et une vue intactes. Ils doivent également être indemnes de tout empêchement de santé pouvant entraver leurs fonctions et sont tenus de subir un examen médical spécialisé avant le recrutement."
        },
        {
            "id": 9,
            "type": "mot-clé",
            "question": "Quels sont les trois responsables spécifiques légalement tenus d'assurer une présence de jour et de nuit dans leur établissement en cas de nécessité ?",
            "expected_answer": "En vertu de l'article 27, le directeur de l'établissement éducatif, le censeur et l'intendant sont tenus d'être présents de jour comme de nuit en cas de nécessité, même en dehors des heures officielles de travail."
        },
        {
            "id": 10,
            "type": "sémantique",
            "question": "Selon les articles 37 et 39, quelles sont les trois issues possibles pour un enseignant stagiaire à la suite de son 'examen de titularisation' à la fin de son stage probatoire ?",
            "expected_answer": "Suite à l'évaluation par une commission qualifiée, le stagiaire peut être : 1) titularisé (confirmé dans son poste), 2) astreint à une prorogation de stage une seule fois pour la même durée, ou 3) licencié sans préavis ni indemnité."
        },
        {
            "id": 11,
            "type": "mot-clé",
            "question": "Quel montant exact en autorisations d'engagement est transféré au portefeuille de programmes de la Présidence de la République en vertu du décret n° 24-440 ?",
            "expected_answer": "Un montant de 48.300.000.000 DA (quarante-huit milliards trois cents millions de dinars) en autorisations d'engagement est ouvert et applicable au portefeuille de programmes de la Présidence de la République."
        },
        {
            "id": 12,
            "type": "sémantique",
            "question": "Selon le décret n° 25-60, quels sont les éléments principaux qui doivent être inclus dans un plan de confortement prioritaire pour les infrastructures exposées aux risques de catastrophes ?",
            "expected_answer": "Le plan doit inclure la collecte d'informations et de données, l'examen visuel et les essais en laboratoire, l'analyse et la modélisation, la budgétisation des opérations de confortement, les projets d'exécution des stratégies de confortement, la surveillance de l'intégrité structurelle et l'évaluation des mesures de confortement et des risques de catastrophes."
        },
        {
            "id": 13,
            "type": "mot-clé",
            "question": "Quels sont les deux sous-comités techniques inclus dans la commission intersectorielle d'évaluation des dommages établie par le décret n° 25-61 ?",
            "expected_answer": "La commission comprend un sous-comité chargé de l'évaluation des dommages et des estimations financières pour la phase de relèvement, et un sous-comité chargé de l'élaboration de recommandations pour reconstruire et réhabiliter en mieux, sur la base des études des organismes compétents."
        },
        {
            "id": 14,
            "type": "sémantique",
            "question": "Comment l'article 2 du décret n° 25-62 définit-il les 'déchets de catastrophe' ?",
            "expected_answer": "Les déchets de catastrophe sont définis comme des matériaux, objets, résidus et toutes autres substances devenus impropres à la consommation et inutilisables en l'état, susceptibles de nuire à la santé humaine, à l'environnement et à la salubrité publique."
        },
        {
            "id": 15,
            "type": "mot-clé",
            "question": "En vertu du décret n° 25-62, quelle autorité spécifique est chargée d'assurer le secrétariat de la commission de wilaya chargée du plan de gestion des déchets de catastrophe ?",
            "expected_answer": "Le secrétariat de la commission est assuré par la direction de l'environnement de la wilaya."
        },
        {
            "id": 16,
            "type": "mot-clé",
            "question": "Selon l'article 158 de la Constitution mentionné dans l'avis de la Cour constitutionnelle, quel est le délai maximum de réponse pour les questions écrites et orales adressées au Gouvernement ?",
            "expected_answer": "Le délai maximum de réponse est de trente (30) jours pour les questions écrites et orales."
        },
        {
            "id": 17,
            "type": "sémantique",
            "question": "Pourquoi la Cour constitutionnelle a-t-elle rejeté la demande parlementaire d'interprétation de l'article 158 de la Constitution dans son Avis n° 01 ?",
            "expected_answer": "La Cour a rejeté la saisine parce que les dispositions de l'article 158 sont claires dans tous leurs alinéas et ne contiennent aucune ambiguïté, contradiction ou obscurité nécessitant une interprétation."
        },
        {
            "id": 18,
            "type": "sémantique",
            "question": "En vertu du décret n° 24-441, quels transferts financiers spécifiques ont été effectués dans le budget de l'Etat 2024 concernant le secteur agricole ?",
            "expected_answer": "Un montant de 44.369.000.000 DA a été annulé du budget des 'Dépenses imprévues' géré par le ministre des finances. Ce même montant a ensuite été ouvert et alloué au programme 'Agriculture et développement rural' de l'ex-ministère de l'Agriculture et du Développement rural, ciblant spécifiquement le sous-programme de développement rural et de gestion durable des territoires."
        },
        {
            "id": 19,
            "type": "mot-clé",
            "question": "Qui a reçu la médaille de l'ordre du mérite national au rang de 'Achir' en vertu du décret n° 25-70 ?",
            "expected_answer": "La médaille a été décernée à M. Mohamed Tarek Belaribi, ministre de l'Habitat, de l'Urbanisme et de la Ville."
        },
        {
            "id": 20,
            "type": "sémantique",
            "question": "Comment le décret n° 25-71 modifie-t-il la structure administrative locale de la commune d'El Khroub ?",
            "expected_answer": "Le décret organise l'ensemble du territoire de la commune d'El Khroub en huit (8) délégations communales spécifiques. Simultanément, il ordonne que les antennes communales existant précédemment et implantées sur son territoire soient supprimées."
        }
    ]
}

# --- FONCTIONS UTILITAIRES ---
def clean_json_string(raw_string):
    match = re.search(r'\{.*\}', raw_string.strip(), re.DOTALL)
    if match:
        return match.group(0)
    return "{}" 

def ask_ollama(system_prompt, user_prompt, model=OLLAMA_MODEL, json_format=False):
    # Pour le super prompt, on met tout dans le user_prompt pour imiter ta prod
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    response = ollama.chat(
        model=model,
        messages=messages,
        format='json' if json_format else '',
        options={"temperature": 0.0}
    )
    content = response['message']['content']
    if json_format:
        content = clean_json_string(content)
    return content

def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

def reciprocal_rank_fusion(results_dict, weights=None, k=60):
    if weights is None:
        weights = {system: 1.0 for system in results_dict.keys()}
        
    fused_scores = {}
    for system_name, doc_list in results_dict.items():
        weight = weights.get(system_name, 1.0)
        
        for rank, (doc_id, doc_text, metadata) in enumerate(doc_list):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = {"score": 0.0, "text": doc_text, "meta": metadata}
            fused_scores[doc_id]["score"] += weight * (1 / (k + rank + 1))
            
    return sorted(fused_scores.items(), key=lambda x: x[1]["score"], reverse=True)

# --- BOUCLE D'ÉVALUATION CIBLÉE ---
def evaluate_rag():
    print(f"🔌 Connexion à ChromaDB ({CHROMA_PATH})...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print("🤖 Chargement du modèle BAAI/bge-m3 sur GPU...")
    embedding_model = SentenceTransformer("BAAI/bge-m3", device="cuda")

    print("📚 Chargement de l'index BM25...")
    all_docs = collection.get()
    documents_all = all_docs['documents']
    ids_all = all_docs['ids']
    metadatas_all = all_docs['metadatas']
    
    if not documents_all:
        print("⚠️ Base ChromaDB vide.")
        return

    tokenized_corpus = [normalize_text(doc) for doc in documents_all]
    bm25 = BM25Okapi(tokenized_corpus)

    # Filtrer le dataset pour ne garder que les questions ayant échoué
    failed_samples = [s for s in combined_golden_dataset["samples"] if s["id"] in FAILED_IDS]

    results_log = []
    total_score = 0
    retrieval_success_count = 0

    print(f"\n🚀 Début de la RE-ÉVALUATION sur les {len(failed_samples)} questions problématiques...\n")

    for sample in failed_samples:
        print(f"[{sample['id']}] Question: {sample['question']}")
        query = sample["question"]
        
        # --- ETAPE 1 : RETRIEVAL HYBRIDE ---
        # Vector Search
        q_embed = embedding_model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        vec_list = []
        if vec_res['ids'] and len(vec_res['ids']) > 0:
            for i in range(len(vec_res['ids'][0])):
                vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

        # Keyword Search
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:10]
        kw_list = []
        for idx in top_n_indices:
            if doc_scores[idx] > 0:
                kw_list.append((ids_all[idx], documents_all[idx], metadatas_all[idx]))

        # ⚖️ NOUVEAUX POIDS RRF : On donne un léger avantage au sens (Vecteur)
        best_weights = {"keyword": 0.4, "vector": 0.6}
        final_results = reciprocal_rank_fusion(
            {"vector": vec_list, "keyword": kw_list}, 
            weights=best_weights
        )
        
        # Préparation du contexte (Format "Prod")
        context_pieces = []
        for rank, (doc_id, data) in enumerate(final_results[:3]):
            text = data['text']
            context_pieces.append(f"DOCUMENT {rank+1}\nCONTENU: {text}")
        
        full_context = "\n\n---\n\n".join(context_pieces)

        # AFFICHAGE COMPLET DES DOCUMENTS
        print("\n📑 --- DOCUMENTS RÉCUPÉRÉS PAR LE RAG (TOP 3) ---")
        print(full_context)
        print("-------------------------------------------------\n")

        # --- ETAPE 2 : GENERATION AVEC LE SUPER PROMPT ---
        super_prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
        Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.

        ⚠️ RÈGLES STRICTES DE RÉDACTION :
        1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
        2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement (ex: "Oui.", "C'est M. X.", "Il s'agit du décret..."), puis justifie en citant la base légale.
        3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant tels qu'ils apparaissent dans la Source (ex: "Selon l'article 3 du décret n° 25-74..."). Ne dis JAMAIS "D'après le document 1" ou "Selon la source fournie".
        4. PRÉCISION CHIRURGICALE : 
        - Noms et Fonctions : Reproduis fidèlement les noms propres, institutions et intitulés officiels.
        - Nombres et Budgets : Écris les montants, durées ou quantités exactes. S'il s'agit d'un budget, fais bien la distinction entre les crédits "annulés" et "ouverts/appliqués".
        - Énumérations : Utilise une liste numérotée si la réponse contient plusieurs éléments distincts (comme des sous-directions ou des délégations).
        5. NETTOYAGE VISUEL : Ignore les pointillés ("....") ou les mentions d'édition type "(sans changement)" présents dans le texte brut.

        CONTEXTE FOURNI :
        {full_context}

        QUESTION DE L'UTILISATEUR :
        {query}

        RÉPONSE :
        """
        
        # On passe le super prompt directement
        mistral_answer = ask_ollama(system_prompt="", user_prompt=super_prompt)

        # --- ETAPE 3 : EVALUATION DU RETRIEVAL ---
        retrieval_judge_prompt = f"""Tu es un évaluateur strict. Le CONTEXTE FOURNI contient-il les informations nécessaires pour produire la REPONSE ATTENDUE ?
Réponds uniquement par un objet JSON valide avec une clé "retrieval_success" (valeur booléenne true ou false).

CONTEXTE FOURNI: {full_context}
REPONSE ATTENDUE: {sample['expected_answer']}
"""
        retrieval_eval_str = ask_ollama("Tu dois répondre au format JSON.", retrieval_judge_prompt, json_format=True)
        try:
            retrieval_eval = json.loads(retrieval_eval_str)
            retrieval_pass = retrieval_eval.get("retrieval_success", False)
        except json.JSONDecodeError:
            retrieval_pass = False
            
        if retrieval_pass:
            retrieval_success_count += 1

        # --- ETAPE 4 : EVALUATION DE LA REPONSE ---
        answer_judge_prompt = f"""Tu es un juge impitoyable. Compare la REPONSE GENEREE avec la REPONSE ATTENDUE.
- Si la REPONSE GENEREE manque un chiffre exact, une date, ou un nom de la REPONSE ATTENDUE, pénalise sévèrement.
- Attribue un score de 0 (Faux/Incomplet) à 5 (Parfaitement exact).

REPONSE ATTENDUE: {sample['expected_answer']}
REPONSE GENEREE: {mistral_answer}

Renvoie uniquement un objet JSON valide avec deux clés :
"score" (entier de 0 à 5)
"reasoning" (courte explication)
"""
        answer_eval_str = ask_ollama("Tu dois répondre au format JSON.", answer_judge_prompt, json_format=True)
        try:
            answer_eval = json.loads(answer_eval_str)
            raw_score = answer_eval.get("score", 0)
            if isinstance(raw_score, str):
                raw_score = raw_score.split("/")[0].strip()
            score = int(raw_score)
            reasoning = answer_eval.get("reasoning", "Pas de raison.")
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 0
            reasoning = "Erreur de parsing du juge."
            
        total_score += score

        # AFFICHAGE COMPLET DE LA REPONSE
        print(f"   🔎 Contexte Trouvé : {'✅ Oui' if retrieval_pass else '❌ Non'}")
        print(f"   🤖 Réponse Mistral complète :\n{mistral_answer.strip()}\n")
        print(f"   ⚖️  Score          : {score}/5")
        if score < 5:
            print(f"   📝 Raison          : {reasoning}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_rag()
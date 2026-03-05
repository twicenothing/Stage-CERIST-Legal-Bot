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

combined_golden_dataset = {
    "dataset_name": "Evaluation RAG - Journaux Officiels Algériens (F2025001, F2025004, F2025007, F2025009)",
    "samples": [
        # --- À partir du fichier F2025001 ---
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
        
        # --- À partir du fichier F2025004 ---
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
        
        # --- À partir du fichier F2025007 ---
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
        
        # --- À partir du fichier F2025009 ---
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
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
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

# --- BOUCLE D'ÉVALUATION PRINCIPALE ---
def evaluate_rag():
    print(f"🔌 Connexion à ChromaDB ({CHROMA_PATH})...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print("🤖 Chargement du modèle BAAI/bge-m3...")
    embedding_model = SentenceTransformer("BAAI/bge-m3", model_kwargs={"use_safetensors": True})

    # PRÉPARATION BM25 (Comme en Prod)
    print("📚 Chargement de l'index BM25...")
    all_docs = collection.get()
    documents_all = all_docs['documents']
    ids_all = all_docs['ids']
    metadatas_all = all_docs['metadatas']
    
    if not documents_all:
        print("⚠️ Base ChromaDB vide. Fin de l'évaluation.")
        return

    tokenized_corpus = [normalize_text(doc) for doc in documents_all]
    bm25 = BM25Okapi(tokenized_corpus)

    results_log = []
    total_score = 0
    retrieval_success_count = 0

    print(f"\n🚀 Début de l'évaluation Hybride sur {len(combined_golden_dataset['samples'])} questions...\n")

    for sample in combined_golden_dataset["samples"]:
        print(f"[{sample['id']}] Question: {sample['question']}")

        # --- ETAPE 1 : RETRIEVAL HYBRIDE (Miroir de la Prod) ---
        query = sample["question"]
        
        # A. Vector Search (Top 10)
        q_embed = embedding_model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        
        vec_list = []
        if vec_res['ids'] and len(vec_res['ids']) > 0:
            for i in range(len(vec_res['ids'][0])):
                vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

        # B. Keyword Search BM25 (Top 10)
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:10]
        
        kw_list = []
        for idx in top_n_indices:
            if doc_scores[idx] > 0:
                kw_list.append((ids_all[idx], documents_all[idx], metadatas_all[idx]))

        # C. RRF Fusion (Poids Prod : 0.7 Keyword / 0.3 Vector)
        best_weights = {"keyword": 0.7, "vector": 0.3}
        final_results = reciprocal_rank_fusion(
            {"vector": vec_list, "keyword": kw_list}, 
            weights=best_weights
        )
        
        # D. Extraction des 3 meilleurs pour le contexte
        context_pieces = []
        for rank, (doc_id, data) in enumerate(final_results[:3]):
            text = data['text']
            context_pieces.append(text)
        
        retrieved_context = "\n\n---\n\n".join(context_pieces)

        # --- ETAPE 2 : GENERATION (Mistral) ---
        rag_system_prompt = "Tu es un assistant juridique expert. Réponds à la question en te basant UNIQUEMENT sur le contexte fourni. Si l'information n'y est pas, dis-le."
        rag_user_prompt = f"Contexte:\n{retrieved_context}\n\nQuestion: {sample['question']}"
        mistral_answer = ask_ollama(rag_system_prompt, rag_user_prompt)

        # --- ETAPE 3 : EVALUATION DU RETRIEVAL (Juge) ---
        retrieval_judge_prompt = f"""Tu es un évaluateur strict. Le CONTEXTE FOURNI contient-il les informations nécessaires pour produire la REPONSE ATTENDUE ?
Réponds uniquement par un objet JSON valide avec une clé "retrieval_success" (valeur booléenne true ou false).

CONTEXTE FOURNI: {retrieved_context}
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

        # --- ETAPE 4 : EVALUATION DE LA REPONSE (Juge) ---
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

        # --- LOGGING ---
        print(f"   🔎 Contexte Hybride : {'✅ Oui' if retrieval_pass else '❌ Non'}")
        print(f"   🤖 Réponse Mistral  : {mistral_answer.strip()[:100]}...")
        print(f"   ⚖️  Score          : {score}/5")
        if score < 5:
            print(f"   📝 Raison        : {reasoning}")
        print("-" * 50)

        results_log.append({
            "id": sample["id"],
            "question": sample["question"],
            "retrieval_pass": retrieval_pass,
            "score": score,
            "reasoning": reasoning
        })

    # --- BILAN FINAL ---
    print("\n" + "="*50)
    print("📊 BILAN DE L'ÉVALUATION HYBRIDE (Vector + BM25)")
    print("="*50)
    print(f"Taux de succès du Retrieval : {retrieval_success_count}/{len(combined_golden_dataset['samples'])} ({(retrieval_success_count/len(combined_golden_dataset['samples']))*100:.1f}%)")
    
    max_score = len(combined_golden_dataset['samples']) * 5
    print(f"Score global de Génération  : {total_score}/{max_score} ({(total_score/max_score)*100:.1f}%)")
    
    with open("evaluation_results_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)
    print("📁 Résultats sauvegardés dans 'evaluation_results_hybrid.json'")

if __name__ == "__main__":
    evaluate_rag()
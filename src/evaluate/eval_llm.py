import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
import ollama
import re

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"

# --- LE GOLDEN DATASET (Avec Expected Answers) ---
golden_dataset = [
    {
        "query": "Quel décret exécutif a officiellement abrogé les attributions du ministre de la jeunesse et des sports de 2016 ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "Il s'agit du décret exécutif n° 16-84 du 1er mars 2016, qui a été abrogé par l'article 10 du décret exécutif n° 25-74 du 11 février 2025."
    },
    {
        "query": "Le ministre de la jeunesse est-il responsable de la stratégie pour les jeunes de la communauté nationale à l'étranger ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "Oui. Selon l'article 3 du décret n° 25-74, il est chargé d'élaborer une stratégie d'action au profit des jeunes de la communauté nationale à l'étranger en coordination avec les secteurs concernés."
    },
    {
        "query": "Quelles sont les trois sous-directions rattachées à la direction du développement des compétences, des initiatives des jeunes et de l’insertion ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "La direction comprend : 1. La sous-direction des programmes de développement des compétences de vie et numériques des jeunes. 2. La sous-direction de la promotion des initiatives, de la créativité et du leadership. 3. La sous-direction de la protection et de l'insertion des jeunes."
    },
    {
        "query": "Qui assiste le secrétaire général du ministère de la jeunesse dans ses fonctions ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "Selon l'article 1er du décret n° 25-75, le secrétaire général est assisté de deux (2) directeurs d’études."
    },
    {
        "query": "Quelle structure est spécifiquement chargée de la gestion du parc automobile et de l'entretien des bâtiments de l'administration centrale ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "C'est la sous-direction des moyens généraux, qui est rattachée à la direction de l'administration générale (Article 10, point d)."
    },
    {
        "query": "À quel rythme et auprès de quelles instances le ministre de la jeunesse doit-il présenter les résultats de ses activités ?",
        "expected_doc_id": "F2025010.json",
        "expected_answer": "Il rend compte au Premier ministre (ou Chef du Gouvernement), au Gouvernement et au Conseil des ministres, selon les formes, modalités et échéances établies (Article 1er du décret n° 25-74)."
    },
    {
        "query": "À qui a été décernée la médaille de l'ordre du mérite national au rang de « Achir » selon le décret présidentiel n° 25-70 ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "La médaille a été décernée à M. Mohamed Tarek Belaribi, ministre de l'habitat, de l’urbanisme et de la ville."
    },
    {
        "query": "Quel ministre a reçu une distinction honorifique de l'ordre du mérite national début février 2025 ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "C'est M. Mohamed Tarek Belaribi, le ministre de l'habitat, de l’urbanisme et de la ville."
    },
    {
        "query": "Combien de délégations communales composent le territoire de la commune d’El Khroub selon le décret exécutif n° 25-71 ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "Le territoire de la commune d’El Khroub est organisé dans sa totalité en huit (8) délégations communales."
    },
    {
        "query": "Quelles sont les limites géographiques de la délégation communale 'Ain Nahas' au nord et à l'ouest ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "Au Nord et à l'Ouest, la délégation 'Ain Nahas' est délimitée par les limites territoriales de la commune de Constantine."
    },
    {
        "query": "Pour quel sous-programme spécifique les crédits de 44.369.000.000 DA ont-ils été ouverts fin 2024 ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "Ils ont été ouverts et appliqués au sous-programme « Développement rural et gestion équilibrée et durable des territoires »."
    },
    {
        "query": "Quelle association nationale Fatiha Khelfi représente-t-elle d'après la liste de nominations ?",
        "expected_doc_id": "F2025009.json",
        "expected_answer": "Elle est la représentante de l'association nationale des affaires de la femme divorcée, de la veuve et de l'enfance."
    },
    {
        "query": "Quel est le montant exact en crédits de paiement qui a été annulé puis ouvert au profit de la Présidence de la République selon le décret présidentiel n° 24-440 ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "Le montant en crédits de paiement est de vingt milliards de dinars (20.000.000.000 DA)."
    },
    {
        "query": "Quelle autorité préside la commission nationale chargée d'approuver les projets visant à préserver les infrastructures face aux séismes et autres risques ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "La commission nationale est présidée par le ministre chargé de l’habitat ou son représentant (Article 10 du décret sur les plans de confortement)."
    },
    {
        "query": "Qui a été nommé directeur général de l'office national des statistiques (ONS) à la fin du mois de janvier 2025 ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "C'est M. Taoufik Hadj-Messaoud qui a été nommé à ce poste."
    },
    {
        "query": "Quel poste a été attribué à Maamar Benlahcene après la fin de ses fonctions de secrétaire général à la Haute autorité de transparence ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "Il a été nommé chargé de mission à la Présidence de la République."
    },
    {
        "query": "D'après les textes encadrant la gestion locale des catastrophes, quelle direction de wilaya est spécifiquement chargée d'assurer le secrétariat de la commission dédiée aux déchets ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "Le secrétariat de cette commission est assuré par la direction de l’environnement de wilaya."
    },
    {
        "query": "Quelle est la durée du mandat des membres siégeant à la commission nationale des plans de confortement des bâtiments stratégiques ?",
        "expected_doc_id": "F2025007.json",
        "expected_answer": "Les membres sont nommés pour une durée de trois (3) ans, renouvelable (Article 11)."
    }
]

# --- 1. RECHERCHE ET GÉNÉRATION (RAG) ---
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

def generate_answer(query, ranked_results, top_k=3):
    if not ranked_results:
        return "Les documents fournis ne contiennent pas cette information."

    # Préparation du contexte réduit à top_k (3)
    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:top_k]):
        text = data['text']
        # Utilisation de la structure propre issue de notre nouvelle indexation
        context_pieces.append(f"--- DOCUMENT {rank+1} ---\n{text}")

    full_context = "\n\n".join(context_pieces)

    prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
    Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.

    ⚠️ RÈGLES STRICTES DE RÉDACTION :
    1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
    2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement (ex: "Oui.", "C'est M. X.", "Il s'agit du décret..."), puis justifie en citant la base légale.
    3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant tels qu'ils apparaissent dans la Source.
    4. PRÉCISION CHIRURGICALE : 
       - Noms et Fonctions : Reproduis fidèlement les noms propres et institutions.
       - Nombres et Budgets : Écris les montants, durées ou quantités exactes. 
       - Énumérations : Utilise une liste numérotée si la réponse contient plusieurs éléments distincts.
    5. NETTOYAGE VISUEL : Ignore les pointillés ("....") présents dans le texte brut.

    CONTEXTE FOURNI :
    {full_context}

    QUESTION DE L'UTILISATEUR :
    {query}

    RÉPONSE :
    """
    
    try:
        response = ollama.chat(
            model='mistral', 
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.1} # Température basse pour la précision factuelle
        )
        return response['message']['content']
    except Exception as e:
        return f"Erreur Ollama : {e}"

# --- 2. FONCTION DU JUGE (LLM-AS-A-JUDGE) ---
def evaluate_mistral_answer(query, expected_answer, generated_answer, judge_model="mistral"):
    judge_prompt = f"""Tu es un évaluateur impartial expert en droit.
    Ta mission est d'évaluer si la réponse générée par un assistant correspond à la réponse attendue pour une question donnée.
    
    Critères d'évaluation :
    1. Exactitude des faits (les montants, noms, dates doivent être identiques).
    2. Fidélité au sens de la réponse attendue.
    
    QUESTION : {query}
    RÉPONSE ATTENDUE (Référence) : {expected_answer}
    RÉPONSE GÉNÉRÉE (À évaluer) : {generated_answer}
    
    Donne une note entre 0 et 10.
    0 = Totalement faux, information manquante, refus de répondre, ou hallucination.
    5 = Partiellement vrai mais manque une information clé (ex: le montant est bon mais pas la citation de l'article).
    10 = Parfaitement correct, même si la formulation est légèrement différente de la réponse attendue.
    
    Réponds UNIQUEMENT avec le score sous ce format exact : "SCORE: X". Ne justifie pas ta note.
    """
    
    try:
        response = ollama.chat(
            model=judge_model, 
            messages=[{'role': 'user', 'content': judge_prompt}],
            options={'temperature': 0.0} 
        )
        judge_output = response['message']['content'].strip()
        
        match = re.search(r'SCORE:\s*(\d+)', judge_output, re.IGNORECASE)
        if match:
            return int(match.group(1))
        else:
            return 0
    except Exception:
        return 0

# --- 3. SCRIPT D'ÉVALUATION PRINCIPAL ---
def main():
    print("🔄 Initialisation du RAG pour l'évaluation des réponses...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    
    # Si tu as un GPU, on enlève device="cpu"
    model = SentenceTransformer(MODEL_NAME, model_kwargs={"use_safetensors": True})

    all_docs = collection.get()
    documents = all_docs['documents']
    ids = all_docs['ids']
    metadatas = all_docs['metadatas']
    
    tokenized_corpus = [normalize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"✅ Système prêt. Lancement de l'évaluation sur {len(golden_dataset)} questions...\n")
    print("="*80)

    total_score = 0
    weights = {"keyword": 0.7, "vector": 0.3} # Nos poids optimisés

    for idx, item in enumerate(golden_dataset):
        query = item["query"]
        expected_answer = item["expected_answer"]
        
        print(f"[{idx+1}/{len(golden_dataset)}] ❓ Q: {query}")
        
        # --- ETAPE 1 : RETRIEVAL ---
        q_embed = model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        vec_list = [(vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]) for i in range(len(vec_res['ids'][0]))]
        
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n = np.argsort(doc_scores)[::-1][:10]
        kw_list = [(ids[idx], documents[idx], metadatas[idx]) for idx in top_n if doc_scores[idx] > 0]
        
        final_results = reciprocal_rank_fusion({"vector": vec_list, "keyword": kw_list}, weights=weights)
        
        # --- ETAPE 2 : GENERATION (top_k=3) ---
        generated_answer = generate_answer(query, final_results, top_k=3)
        
        # --- ETAPE 3 : EVALUATION PAR LE JUGE ---
        score = evaluate_mistral_answer(query, expected_answer, generated_answer, judge_model="mistral")
        total_score += score
        
        print(f"   🤖 Mistral : {generated_answer}")
        print(f"   ⚖️  Juge    : {score}/10")
        print("-" * 80)

    # --- RESULTATS FINAUX ---
    average_score = total_score / len(golden_dataset)
    print("\n" + "★"*40)
    print(f"🏆 RAG GENERATION SCORE FINAL : {average_score:.2f} / 10")
    print("★"*40)
    
    if average_score >= 8.5:
        print("🚀 EXCELLENT ! Ton système est prêt pour la production.")
    elif average_score >= 7.0:
        print("👍 BON ! Quelques petites erreurs, mais globalement solide.")
    else:
        print("⚠️ MOYEN. Il faut peut-être ajuster le chunking, tester Llama 3 ou revoir le Top K.")

if __name__ == "__main__":
    main()
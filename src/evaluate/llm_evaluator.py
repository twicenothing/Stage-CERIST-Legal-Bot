import json
import time
import string
import numpy as np
import chromadb
import ollama
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from colorama import Fore, Style, init

# Initialisation des couleurs pour le terminal
init(autoreset=True)

# ===================================================================
# --- CONFIGURATION ---
# ===================================================================
# RAG Config
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RAG_MODEL = "mistral" # Modèle qui génère la réponse

# Judge Config
JUDGE_MODEL = "mistral" # Modèle qui évalue la réponse

# Poids du RRF (Mis à jour selon ta demande)
RRF_WEIGHTS = {"keyword": 0.3, "vector": 0.7}

# ===================================================================
# --- DATASET ---
# ===================================================================
dataset = {
    "samples" : [
        {"id": 1, "question" : "Pourquoi la juridiction suprême a-t-elle refusé d'accéder à la requête des parlementaires concernant l'article 158 ?", "answer" : "La Cour a rejeté la demande sur le fond car elle estime que les dispositions de l'article 158 sont déjà parfaitement claires, rigides et dénuées de toute ambiguïté. Interpréter un texte déjà explicite n'était donc pas justifié.", "source" : "F2025009.json"},
        {"id": 2, "question" : "Quel est le danger potentiel souligné par les juges s'ils acceptaient d'expliquer une disposition constitutionnelle qui est déjà évidente ?", "answer" : "La Cour souligne qu'une interprétation extensive de dispositions claires risquerait d'entraîner une modification indirecte de la Constitution en dehors des voies légales, créant ainsi une forme de révision constitutionnelle parallèle par le juge.", "source" : "F2025009.json"},
        {"id": 3, "question" : "Quel est l'impact pratique de l'arrêté de janvier 2025 sur la gestion quotidienne des fonds alloués à l'Inspection générale des finances, et quelle capacité d'action spécifique est transférée à M. Saïd Touakni ?", "answer" : "Ce texte permet de décentraliser et de fluidifier l'exécution budgétaire de l'Inspection générale des finances (IGF). Il autorise M. Saïd Touakni à agir légalement au nom du ministre des Finances pour valider et signer tous les documents liés aux dépenses (y compris les ordres de paiement), mais uniquement dans le périmètre strict du budget propre à l'IGF.", "source" : "F2025009.json"},
        {"id": 4, "question" : "À quel programme, sous-programme et titre exacts est applicable le montant ouvert de trente-neuf millions de dinars (39.000.000 DA) pour le portefeuille du ministère des transports ?", "answer" : "Ce montant est applicable au programme « Administration générale », au sous-programme « Soutien administratif » et au titre 2 « Dépenses de fonctionnement des services ».", "source" : "F2025005.json"},
        {"id": 5, "question" : "Quelle est la nouvelle échéance accordée aux groupements d'agriculteurs pour se mettre en règle avec la législation de 1996 qui les régit ?", "answer" : "Ce montant est applicable au programme « Administration générale », au sous-programme « Soutien administratif » et au titre 2 « Dépenses de fonctionnement des services ».", "source" : "F2025009.json"},
        {"id": 6, "question" : "Au sein de la commission sectorielle chargée de la tutelle pédagogique sur l'école supérieure de la sécurité sociale, que se passe-t-il si la plupart des membres sont absents lors d'une réunion, et comment les décisions sont-elles tranchées en cas d'égalité des votes ?", "answer" : "Si le quorum exigé des deux tiers (2/3) des membres n'est pas atteint lors de la première réunion, une seconde session doit être convoquée dans les huit (8) jours suivants. Lors de cette seconde réunion, la commission peut valablement délibérer quel que soit le nombre de personnes présentes. Si les votes sont partagés à égalité, c'est la voix du président qui est prépondérante pour trancher.", "source" : "F2025005.json"},
        {"id": 7, "question" : "La liste des membres du Conseil national économique, social et environnemental (CNESE) nommés en janvier 2025 est-elle complète, et quelle est la durée de leur mandat ?", "answer" : "Non, la liste n'est pas exhaustive et les membres restants seront nommés ultérieurement (selon l'article 3). Pour les membres déjà désignés dans ce texte, la durée de leur mandat est fixée à quatre (4) ans (selon l'article 1er).", "source" : "F2025005.json"},
        {"id": 8, "question" : "De quelles manières exactes le ministre de la jeunesse doit-il intervenir en faveur de la jeunesse algérienne établie hors du pays, selon le décret de février 2025 fixant ses attributions ?", "answer" : "Le ministre a trois responsabilités principales envers la communauté nationale à l'étranger, réparties dans différents domaines d'action :\nIdentité (Art. 2) : Il doit proposer et développer des mesures pour renforcer leur esprit d'appartenance nationale.\nStratégie (Art. 3) : Il est chargé d'élaborer une stratégie d'action spécifique à leur profit, en coordination avec d'autres secteurs ministériels.\nRayonnement et Talents (Art. 7) : Dans le cadre des relations internationales, il doit mettre en œuvre des mesures pour valoriser les compétences et les talents des jeunes issus de cette communauté.", "source" : "F2025010.json"},
        {"id": 9, "question" : "Monsieur Hamid Benazouz a reçu l'autorisation de valider de nombreuses opérations financières et administratives à la place de la ministre. Cependant, quelle est la limite stricte de cette délégation et quel type de document n'a-t-il absolument pas le droit de signer ?", "answer" : "Bien qu'il puisse signer en son nom les actes, les décisions, les ordonnances de paiement et les pièces justificatives de dépenses, la délégation exclut formellement la signature des arrêtés (conformément à l'Article 1er).", "source" : "F2025010.json"},
        {"id": 10, "question" : "Dans le cadre de la convention d'extradition signée en 2021 entre l'Algérie et la Tunisie, un individu recherché pour des actes terroristes ou pour une tentative d'assassinat sur un membre du Gouvernement peut-il bloquer son extradition en affirmant qu'il s'agit d'un crime politique ?", "answer" : "Non. Selon l'Article 4 de la convention, bien que l'extradition soit normalement refusée pour les infractions politiques, les actes terroristes et les attentats à la vie ou à l'intégrité physique d'un Chef d'État, de sa famille ou d'un membre du Gouvernement sont explicitement exclus de la qualification d'infraction politique.", "source" : "F2025008.json"},
        {"id": 11, "question" : "Selon l'accord de décembre 2023 entre l'Algérie et l'Indonésie, un diplomate algérien officiellement affecté pour travailler à l'ambassade d'Algérie à Jakarta a-t-il besoin d'un visa pour sa première entrée sur le territoire indonésien avec son passeport diplomatique ?", "answer" : "Oui. Bien que la règle générale de cet accord exempte les passeports diplomatiques de visa pour des séjours de moins de 30 jours (Article 1), l'Article 4 prévoit une exception stricte : les personnes officiellement affectées à une mission diplomatique ou consulaire doivent impérativement obtenir un visa d'entrée approprié avant leur arrivée. Ce n'est qu'ensuite qu'elles pourront circuler sans visa pendant la durée de leur mission.", "source" : "F2025008.json"},
        {"id": 12, "question" : "D'après le décret présidentiel n° 24-440 du 31 décembre 2024, quels sont les montants exacts en autorisations d'engagement et en crédits de paiement qui ont été transférés à la Présidence de la République, et de quelle rubrique budgétaire spécifique du ministère des finances provenaient ces fonds ?", "answer" : "Selon le décret, 48,3 milliards de dinars (48.300.000.000 DA) en autorisations d'engagement et 20 milliards de dinars (20.000.000.000 DA) en crédits de paiement ont été transférés. Ces fonds proviennent de l'annulation de crédits sur la dotation « Montant non assigné », imputable au titre 7 « Dépenses imprévues » gérée par le ministre des finances.", "source" : "F2025007.json"},
        {"id": 13, "question" : "D'après l'arrêté du ministère de la culture et des arts du 21 janvier 2025, dans quelle ville algérienne le festival culturel international du théâtre du Sahara est-il officiellement institutionnalisé, et à quelle fréquence cet événement doit-il se tenir ?", "answer" : "Selon l'article 1er de cet arrêté, le festival est institutionnalisé dans la ville d'Adrar, et il s'agit d'un événement à périodicité annuelle.", "source" : "F2025007.json"},
        {"id": 14, "question": "Dans le cadre du décret présidentiel n° 25-57 du 23 janvier 2025, quel est l'objet précis de l'accord bilatéral ratifié par l'Algérie, avec quel pays a-t-il été conclu, et à quelle date cet accord avait-il été initialement signé ?", "answer": "Le décret ratifie l'accord de coopération culturelle et scientifique conclu entre l'Algérie et la République fédérale d'Allemagne. Ce document avait été initialement signé à Alger le 13 juin 2022, soit plus de deux ans avant sa ratification par ce décret.", "source": "F025006.json"},
        {"id": 15, "question": "D'après l'annexe de l'accord de coopération culturelle et scientifique entre l'Algérie et l'Allemagne, à quelles conditions strictes un expert allemand détaché en Algérie peut-il importer son véhicule personnel sans payer de droits de douane, et quand aura-t-il le droit de le revendre sur place ?", "answer": "L'expert peut importer son véhicule en franchise de droits de douane à deux conditions : le véhicule doit avoir été utilisé pendant au moins 6 mois avant le transfert, et il doit être dédouané dans les 12 mois suivant son installation. Pour le revendre (ou le céder gratuitement) en Algérie, il doit obligatoirement attendre un délai de 12 mois, sauf s'il décide de payer les droits de douane au préalable (Annexe, Paragraphe 3).", "source": "F025006.json"},
        {"id": 16, "question": "Selon le décret exécutif du 26 janvier 2025, quelles sont les sous-directions exactes confiées respectivement à Lynda Ghoul et Farid Chaoui au sein du ministère algérien de la solidarité nationale, et qui a été nommé à la tête des systèmes d'information ?", "answer": "Lynda Ghoul a été nommée sous-directrice de l'enfance et de l'adolescence en difficulté sociale et en danger moral, tandis que Farid Chaoui a pris la sous-direction de la petite enfance et de l'enfance privée de famille. La sous-direction de la communication et des systèmes d'information a quant à elle été confiée à Ali Abderraouf El-Haffaf.", "source": "F025006.json"},
        {"id": 17, "question": "Dans le rectificatif publié au Journal Officiel n° 82 de décembre 2024 concernant l'avis n° 03/A.C.C/I.C/24 de la Cour constitutionnelle sur l'article 122 de la Constitution, quelle précision juridique majeure a été ajoutée concernant la restriction d'accès aux deux chambres du Parlement ?", "answer": "Le rectificatif ajoute la notion de désignation. Au lieu de limiter la restriction au seul fait de « se porter candidat » (qui ne concerne que les élus), le texte corrigé précise désormais « que nul ne peut se porter candidat ou être désigné ». Cela englobe donc formellement à la fois les parlementaires issus d'élections et ceux nommés par décret.", "source": "F025005.json"},
        {"id": 18, "question": "Selon l'arrêté interministériel du 9 décembre 2024 relatif à l'école supérieure de la sécurité sociale, que se passe-t-il très exactement si le quorum des deux tiers (2/3) n'est pas atteint lors d'une réunion de la commission sectorielle, et comment les décisions sont-elles tranchées en cas d'égalité parfaite des voix lors d'un vote ?", "answer": "D'après l'article 7, si le quorum n'est pas atteint, une deuxième réunion doit être organisée dans les huit (8) jours suivants. Lors de cette seconde réunion, la commission peut délibérer valablement quel que soit le nombre de membres présents. En cas de partage égal des voix lors d'un vote, la voix du président de la commission est prépondérante (elle tranche la décision).", "source": "F025005.json"},
        {"id": 19, "question": "Selon le décret exécutif n° 25-55 du 21 janvier 2025, quel est le taux de l'indemnité de soutien scolaire accordé au personnel d'intendance par rapport au personnel de laboratoire, et ce dernier (le personnel de laboratoire) peut-il également percevoir l'indemnité de qualification ?", "answer": "D'après l'article 10 du décret, le personnel d'intendance et le personnel de laboratoire bénéficient tous deux du même taux, soit 15 % du traitement, pour l'indemnité de soutien scolaire et de remédiation pédagogique. En revanche, le personnel de laboratoire n'a pas droit à l'indemnité de qualification. L'article 7 précise en effet que cette indemnité est exclusivement servie aux personnels cités aux articles 3 et 4 (enseignants, direction, intendance...), excluant de fait les laborantins qui sont régis par l'article 5.", "source": "F025004.json"},
        {"id": 20, "question": "D'après l'arrêté interministériel du 5 janvier 2025, quelle catégorie spécifique de personnel et quelle administration exacte sont visées par la modification des effectifs et de la durée des contrats ?", "answer": "Cet arrêté cible exclusivement les agents contractuels qui exercent des activités d'entretien, de maintenance ou de service. Il s'applique uniquement au niveau de l'administration centrale du ministère de la santé (historiquement désigné comme ministère de la santé, de la population et de la réforme hospitalière).", "source": "F025004.json"},
        {"id": 21, "question": "D'selon le décret présidentiel n° 25-56 du 22 janvier 2025, à quelle date exacte se tiendra l'élection pour le renouvellement de la moitié des membres élus du Conseil de la Nation, et quelles sont les instances qui composent le collège électoral autorisé à voter ?", "answer": "D'après les articles 1 et 2 du décret, le collège électoral est convoqué pour le dimanche 9 mars 2025. Ce collège n'est pas composé de citoyens ordinaires, mais de l'ensemble des membres de l'assemblée populaire de wilaya (APW) et des membres des assemblées populaires communales (APC) de chaque wilaya.", "source": "F025003.json"},
        {"id": 22, "question": "D'après le décret présidentiel du 6 janvier 2025, qui a été nommé à la tête de la garde Républicaine algérienne, avec quel statut précis, et à partir de quelle date exacte cette nomination a-t-elle effectivement pris effet ?", "answer": "Le Général-major Tahar Ayad a été nommé commandant de la garde Républicaine avec le statut précis de commandant « par intérim ». Le point crucial est que, bien que le décret ait été signé le 6 janvier 2025, cette nomination a pris effet de manière rétroactive à compter du 23 décembre 2024.", "source": "F025003.json"},
        {"id": 23, "question": "D'après l'arrêté du 19 décembre 2024 portant nomination au conseil d'administration du musée national du moudjahid, qui a été désigné comme président de ce conseil et quel ministère représente-t-il ? De plus, quels sont les noms exacts des représentants nommés au titre de l'organisation nationale des enfants de chouhada ?", "answer": "Selon cet arrêté, le président du conseil d'administration est Alallou Abdelhamid, qui siège en tant que représentant du ministre des moudjahidine et des ayants droit. Par ailleurs, l'organisation nationale des enfants de chouhada est exceptionnellement représentée par deux membres : Abidli Mohamed Amine et Bakhouche Mokhtar.", "source": "F025003.json"},
        {"id": 24, "question": "Selon le décret présidentiel n° 24-433 du 31 décembre 2024, quels sont les montants exacts transférés respectivement en autorisations d'engagement et en crédits de paiement au profit de la Présidence de la République, et de quelle rubrique budgétaire spécifique ces fonds ont-ils été initialement annulés ?", "answer": "Le décret prévoit le transfert de 4.192.000.000 DA en autorisations d'engagement et de 4.936.300.000 DA en crédits de paiement vers le portefeuille de programmes de la Présidence de la République. Ces fonds ont été prélevés (annulés) sur les crédits gérés par le ministre des finances, plus précisément sur la dotation « Montant non assigné » relevant du titre 7 consacré aux « Dépenses imprévues ».", "source": "F025001.json"},
        {"id": 25, "question": "Selon le décret présidentiel n° 25-03 du 6 janvier 2025, quelle est la durée du mandat d'un membre du Conseil (et est-ce renouvelable ?), comparée à celle d'un membre du bureau ou d'un président de commission ? De plus, à combien de commissions permanentes un membre peut-il appartenir au maximum ?", "answer": "D'après l'article 8 modifié, le mandat d'un membre du Conseil est de quatre (4) ans, renouvelable une seule fois. En revanche, les membres du bureau (article 41) et les présidents des commissions permanentes (article 45) sont élus pour un mandat de deux (2) ans, non renouvelable. Enfin, l'article 45 précise qu'un membre du Conseil ne peut faire partie de plus de deux (2) commissions permanentes.", "source": "F025001.json"},
        {"id": 26, "question": "D'après l'arrêté du 25 novembre 2024, quel pouvoir précis a été délégué à M. Brahim Benbouza par le ministre de l'agriculture, du développement rural et de la pêche, et quelle est l'exception stricte à cette délégation ?", "answer": "M. Brahim Benbouza, en sa qualité de directeur de l'administration des moyens, a reçu délégation pour signer tous les actes et décisions au nom du ministre, dans la limite de ses attributions. Cependant, l'article 1er pose une exception stricte : cette délégation s'applique « à l'exclusion des arrêtés ». Il n'a donc pas le pouvoir de signer des arrêtés ministériels.", "source" : "F025001.json"}
    ]
}

# ===================================================================
# --- INITIALISATION GLOBALE DU RAG ---
# ===================================================================
print(f"🔄 Connexion à ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print("🤖 Chargement du modèle d'embedding...")
# Assure-toi que CUDA est dispo, sinon ça bascule sur CPU via sentence_transformers
model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")

print("📚 Chargement de l'index BM25...")
all_docs = collection.get()
documents = all_docs['documents']
ids = all_docs['ids']
metadatas = all_docs['metadatas']

def normalize_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation)).split()

tokenized_corpus = [normalize_text(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)
print("✅ Système RAG Prêt.\n")


# ===================================================================
# --- FONCTIONS UTILITAIRES RAG ---
# ===================================================================
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

def generate_rag_answer(query):
    # --- 1. Vector Search ---
    q_embed = model.encode([query]).tolist()
    vec_res = collection.query(query_embeddings=q_embed, n_results=10)
    
    vec_list = []
    if vec_res['ids']:
        for i in range(len(vec_res['ids'][0])):
            vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

    # --- 2. Keyword Search ---
    tokenized_query = normalize_text(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(doc_scores)[::-1][:10]
    
    kw_list = []
    for idx in top_n:
        if doc_scores[idx] > 0:
            kw_list.append((ids[idx], documents[idx], metadatas[idx]))

    # --- 3. RRF Fusion avec poids configurés ---
    ranked_results = reciprocal_rank_fusion(
        {"vector": vec_list, "keyword": kw_list}, 
        weights=RRF_WEIGHTS
    )

    if not ranked_results:
        return "Désolé, je n'ai trouvé aucun document pertinent dans la base de données."

    # --- 4. Préparation du prompt Mistral ---
    context_pieces = []
    for rank, (doc_id, data) in enumerate(ranked_results[:3]):
        meta = data['meta']
        text = data['text']
        source_title = meta.get('parent_title', meta.get('title', 'Document sans titre'))
        context_pieces.append(f"DOCUMENT {rank+1} (Source: {source_title})\nCONTENU: {text}")

    full_context = "\n\n---\n\n".join(context_pieces)

    prompt = f"""Tu es un assistant juridique expert en droit administratif algérien.
Ta mission est d'analyser les textes réglementaires fournis en contexte et de répondre à la question de l'utilisateur de manière directe, précise et factuelle.

⚠️ RÈGLES STRICTES DE RÉDACTION :
1. EXCLUSIVITÉ DU CONTEXTE : N'invente aucune information. Si la réponse ne se trouve pas dans le contexte, réponds uniquement : "Les documents fournis ne contiennent pas cette information."
2. STRUCTURE DIRECTE : Va droit au but. Donne la réponse immédiatement, puis justifie en citant la base légale.
3. CITATION JURIDIQUE : Cite TOUJOURS le numéro de l'Article et le numéro du Décret correspondant.
4. PRÉCISION CHIRURGICALE : Reproduis fidèlement les noms propres, montants, et utilise des listes si nécessaire.
5. NETTOYAGE VISUEL : Ignore les pointillés ("....") dans le texte brut.

CONTEXTE FOURNI :
{full_context}

QUESTION DE L'UTILISATEUR :
{query}

RÉPONSE :
"""

    # --- 5. Appel Ollama ---
    try:
        response = ollama.chat(
            model=RAG_MODEL, 
            messages=[{'role': 'user', 'content': prompt}],
            options={"temperature": 0.0} # Température à 0 pour éviter les hallucinations
        )
        return response['message']['content']
    except Exception as e:
        return f"Erreur Ollama (RAG) : {e}"

# ===================================================================
# --- FONCTION DU JUGE LLM ---
# ===================================================================
def evaluate_answer_with_ollama(question, ground_truth, generated_answer):
    prompt = f"""Tu es un expert juridique algérien très strict chargé d'évaluer les réponses d'un assistant IA.
    
    Tâche : Compare la réponse générée (Générée) avec la réponse de référence (Vraie Réponse).
    
    Critères d'évaluation :
    1. Précision Factuelle : La réponse générée contredit-elle la vraie réponse ?
    2. Complétude : La réponse générée contient-elle tous les éléments clés de la vraie réponse ?
    
    Question posée : {question}
    Vraie Réponse (Ground Truth) : {ground_truth}
    Réponse Générée par le RAG : {generated_answer}
    
    Renvoie UNIQUEMENT un objet JSON valide avec ce format exact :
    {{
        "reasoning": "Explication courte de ton évaluation justifiant la note.",
        "score": <un entier de 1 à 5> (1 = totalement faux/incomplet, 3 = partiel, 5 = parfait et complet)
    }}
    """
    
    try:
        response = ollama.chat(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "Tu es un juge impartial. Tu dois répondre uniquement avec un JSON valide."},
                {"role": "user", "content": prompt}
            ],
            format="json",
            options={"temperature": 0.0} 
        )
        
        result_str = response['message']['content']
        return json.loads(result_str)
        
    except Exception as e:
        return {"reasoning": f"Erreur du juge: {str(e)}", "score": 0}

# ===================================================================
# --- BOUCLE PRINCIPALE D'ÉVALUATION ---
# ===================================================================
def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}=== DÉBUT DE L'ÉVALUATION DE GÉNÉRATION ==={Style.RESET_ALL}")
    print(f"Modèle RAG : {Fore.YELLOW}{RAG_MODEL}{Style.RESET_ALL} | Modèle Juge : {Fore.YELLOW}{JUDGE_MODEL}{Style.RESET_ALL}")
    print(f"Poids RRF : {RRF_WEIGHTS}\n")
    
    total_score = 0
    max_score = len(dataset["samples"]) * 5
    results_log = []

    for i, sample in enumerate(dataset["samples"], 1):
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print(f"{Fore.MAGENTA}--- Question {i}/{len(dataset['samples'])} ---")
        print(f"{Style.BRIGHT}Q:{Style.RESET_ALL} {question}")
        
        # 1. Génération RAG
        print(f"{Fore.LIGHTBLACK_EX}Recherche et génération en cours...{Style.RESET_ALL}", end="\r")
        start_time = time.time()
        generated_answer = generate_rag_answer(question)
        generation_time = time.time() - start_time
        
        # 2. Évaluation
        print(f"{Fore.LIGHTBLACK_EX}Évaluation par le juge en cours...  {Style.RESET_ALL}", end="\r")
        eval_start_time = time.time()
        eval_result = evaluate_answer_with_ollama(question, ground_truth, generated_answer)
        eval_time = time.time() - eval_start_time
        
        score = eval_result.get("score", 0)
        reasoning = eval_result.get("reasoning", "Pas de raisonnement fourni.")
        total_score += score
        
        # 3. Affichage
        print(f" " * 50, end="\r") # Efface les messages d'attente
        print(f"{Fore.GREEN}Vraie Réponse:{Style.RESET_ALL} {ground_truth}")
        print(f"{Fore.CYAN}Génération ({generation_time:.1f}s):{Style.RESET_ALL} {generated_answer}")
        
        if score >= 4:
            score_color = Fore.GREEN
        elif score == 3:
            score_color = Fore.YELLOW
        else:
            score_color = Fore.RED
            
        print(f"{Style.BRIGHT}ÉVALUATION ({eval_time:.1f}s) :{Style.RESET_ALL} Score {score_color}{score}/5{Style.RESET_ALL} | {reasoning}\n")
        
        results_log.append({
            "id": sample.get("id", i),
            "score": score,
            "reasoning": reasoning
        })

    # Rapport Final
    average_score = (total_score / max_score) * 100 if max_score > 0 else 0
    
    print(f"{Fore.CYAN}{Style.BRIGHT}=== RAPPORT FINAL ==={Style.RESET_ALL}")
    print(f"Score Total : {total_score} / {max_score}")
    
    if average_score >= 80:
        print(f"Précision Globale : {Fore.GREEN}{Style.BRIGHT}{average_score:.2f}%{Style.RESET_ALL} 🚀")
    elif average_score >= 60:
        print(f"Précision Globale : {Fore.YELLOW}{Style.BRIGHT}{average_score:.2f}%{Style.RESET_ALL} ⚠️")
    else:
        print(f"Précision Globale : {Fore.RED}{Style.BRIGHT}{average_score:.2f}%{Style.RESET_ALL} ❌")

if __name__ == "__main__":
    main()
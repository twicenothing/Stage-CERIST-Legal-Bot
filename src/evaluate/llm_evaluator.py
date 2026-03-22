import chromadb
from sentence_transformers import SentenceTransformer
import time
import requests
import re # Pour extraire le score intelligemment

# --- 1. CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
MODEL_NAME = "BAAI/bge-m3"
TOP_K = 5  # 🚀 MODIFIÉ : Passé à 5 pour ne plus rater la fin des longs décrets (ex: Q7)

# 🚀 NOUVELLE CONFIGURATION SERVEUR & MODÈLE
OLLAMA_URL = "http://127.0.0.1:11435/api/chat"  # Ton serveur GPU personnel
LLM_MODEL = "mixtral:latest"  # 🚀 MODIFIÉ : Utilisation de Command R, l'expert RAG

# (Ton dataset)
dataset = {
    "samples" : [
        {"id": 1, "question" : "Pourquoi la juridiction suprême a-t-elle refusé d'accéder à la requête des parlementaires concernant l'article 158 ?", "answer" : "La Cour a rejeté la demande sur le fond car elle estime que les dispositions de l'article 158 sont déjà parfaitement claires, rigides et dénuées de toute ambiguïté. Interpréter un texte déjà explicite n'était donc pas justifié.", "source" : "F2025009.json", "parent_title" : """Avis n° 01 A.C.C/I.C/25 du 30 Rajab 1446 correspondant au 30 janvier 2025 """},
        {"id": 2, "question" : "Quel est le danger potentiel souligné par les juges s'ils acceptaient d'expliquer une disposition constitutionnelle qui est déjà évidente ?", "answer" : "La Cour souligne qu'une interprétation extensive de dispositions claires risquerait d'entraîner une modification indirecte de la Constitution en dehors des voies légales, créant ainsi une forme de révision constitutionnelle parallèle par le juge.", "source" : "F2025009.json", "parent_title" : """Avis n° 01 A.C.C/I.C/25 du 30 Rajab 1446 correspondant au 30 janvier 2025 relatif à l’interprétation des dispositions de l’article 158 de la Constitution. """},
        {"id": 3, "question" : "Quel est l'impact pratique de l'arrêté de janvier 2025 sur la gestion quotidienne des fonds alloués à l'Inspection générale des finances, et quelle capacité d'action spécifique est transférée à M. Saïd Touakni ?", "answer" : "Ce texte permet de décentraliser et de fluidifier l'exécution budgétaire de l'Inspection générale des finances (IGF). Il autorise M. Saïd Touakni à agir légalement au nom du ministre des Finances pour valider et signer tous les documents liés aux dépenses (y compris les ordres de paiement), mais uniquement dans le périmètre strict du budget propre à l'IGF.", "source" : "F2025009.json","parent_title":"""Arrêté du 30 Rajab 1446 correspondant au 30 janvier 2025"""},
        {"id": 4, "question" : "À quel programme, sous-programme et titre exacts est applicable le montant ouvert de trente-neuf millions de dinars (39.000.000 DA) pour le portefeuille du ministère des transports ?", "answer" : "Ce montant est applicable au programme « Administration générale », au sous-programme « Soutien administratif » et au titre 2 « Dépenses de fonctionnement des services ».", "source" : "F2025005.json", "parent_title": """Décret présidentiel n° 24-439 du 29 Joumada Ethania 1446"""},
        {"id": 5, "question" : "Quelle est la nouvelle échéance accordée aux groupements d'agriculteurs pour se mettre en règle avec la législation de 1996 qui les régit ?", "answer" : "Le délai de mise en conformité pour les coopératives agricoles et leurs unions (vis-à-vis du décret de 1996) a été repoussé au 31 décembre 2025.", "source" : "F2025009.json", "parent_title" : """Décret exécutif n° 25-73 du 11 Chaâbane 1446 correspondant au 10 février 2025"""},
        {"id": 6, "question" : "Au sein de la commission sectorielle chargée de la tutelle pédagogique sur l'école supérieure de la sécurité sociale, que se passe-t-il si la plupart des membres sont absents lors d'une réunion, et comment les décisions sont-elles tranchées en cas d'égalité des votes ?", "answer" : "Si le quorum exigé des deux tiers (2/3) des membres n'est pas atteint lors de la première réunion, une seconde session doit être convoquée dans les huit (8) jours suivants. Lors de cette seconde réunion, la commission peut valablement délibérer quel que soit le nombre de personnes présentes. Si les votes sont partagés à égalité, c'est la voix du président qui est prépondérante pour trancher.", "source" : "F2025005.json", "parent_title":"""Arrêté interministériel du 7 Joumada Ethania 1446 correspondant au 9 décembre 2024"""},
        {"id": 7, "question" : "La liste des membres du Conseil national économique, social et environnemental (CNESE) nommés en janvier 2025 est-elle complète, et quelle est la durée de leur mandat ?", "answer" : "Non, la liste n'est pas exhaustive et les membres restants seront nommés ultérieurement (selon l'article 3). Pour les membres déjà désignés dans ce texte, la durée de leur mandat est fixée à quatre (4) ans (selon l'article 1er).", "source" : "F2025005.json", "parent_title" : """Décision du 14 Rajab 1446 correspondant au 14 janvier 2025"""},
        {"id": 8, "question" : "De quelles manières exactes le ministre de la jeunesse doit-il intervenir en faveur de la jeunesse algérienne établie hors du pays, selon le décret de février 2025 fixant ses attributions ?", "answer" : "Le ministre a trois responsabilités principales envers la communauté nationale à l'étranger, réparties dans différents domaines d'action :\nIdentité (Art. 2) : Il doit proposer et développer des mesures pour renforcer leur esprit d'appartenance nationale.\nStratégie (Art. 3) : Il est chargé d'élaborer une stratégie d'action spécifique à leur profit, en coordination avec d'autres secteurs ministériels.\nRayonnement et Talents (Art. 7) : Dans le cadre des relations internationales, il doit mettre en œuvre des mesures pour valoriser les compétences et les talents des jeunes issus de cette communauté.", "source" : "F2025010.json","parent_title":"""Décret exécutif n° 25-74 du 12 Chaâbane 1446 correspondant au 11 février 2025"""},
        {"id": 9, "question" : "Monsieur Hamid Benazouz a reçu l'autorisation de valider de nombreuses opérations financières et administratives à la place de la ministre. Cependant, quelle est la limite stricte de cette délégation et quel type de document n'a-t-il absolument pas le droit de signer ?", "answer" : "Bien qu'il puisse signer en son nom les actes, les décisions, les ordonnances de paiement et les pièces justificatives de dépenses, la délégation exclut formellement la signature des arrêtés (conformément à l'Article 1er).", "source" : "F2025010.json", "parent_title": """Arrêté du 20 Joumada Ethania 1446 correspondant au 22 décembre 2024"""},
        {"id": 10, "question" : "Dans le cadre de la convention d'extradition signée en 2021 entre l'Algérie et la Tunisie, un individu recherché pour des actes terroristes ou pour une tentative d'assassinat sur un membre du Gouvernement peut-il bloquer son extradition en affirmant qu'il s'agit d'un crime politique ?", "answer" : "Non. Selon l'Article 4 de la convention, bien que l'extradition soit normalement refusée pour les infractions politiques, les actes terroristes et les attentats à la vie ou à l'intégrité physique d'un Chef d'État, de sa famille ou d'un membre du Gouvernement sont explicitement exclus de la qualification d'infraction politique.", "source" : "F2025008.json", "parent_title":"""Convention relative à l’extradition entre Le Gouvernement de la République algérienne démocratique et populaire et Le Gouvernement de la République tunisienne"""},
        {"id": 11, "question" : "Selon l'accord de décembre 2023 entre l'Algérie et l'Indonésie, un diplomate algérien officiellement affecté pour travailler à l'ambassade d'Algérie à Jakarta a-t-il besoin d'un visa pour sa première entrée sur le territoire indonésien avec son passeport diplomatique ?", "answer" : "Oui. Bien que la règle générale de cet accord exempte les passeports diplomatiques de visa pour des séjours de moins de 30 jours (Article 1), l'Article 4 prévoit une exception stricte : les personnes officiellement affectées à une mission diplomatique ou consulaire doivent impérativement obtenir un visa d'entrée approprié avant leur arrivée. Ce n'est qu'ensuite qu'elles pourront circuler sans visa pendant la durée de leur mission.", "source" : "F2025008.json","parent_title":"""Accord entre le Gouvernement de la République algérienne démocratique et populaire et le Gouvernement de la République d’Indonésie"""},
        {"id": 12, "question" : "D'après le décret présidentiel n° 24-440 du 31 décembre 2024, quels sont les montants exacts en autorisations d'engagement et en crédits de paiement qui ont été transférés à la Présidence de la République, et de quelle rubrique budgétaire spécifique du ministère des finances provenaient ces fonds ?", "answer" : "Selon le décret, 48,3 milliards de dinars (48.300.000.000 DA) en autorisations d'engagement et 20 milliards de dinars (20.000.000.000 DA) en crédits de paiement ont été transférés. Ces fonds proviennent de l'annulation de crédits sur la dotation « Montant non assigné », imputable au titre 7 « Dépenses imprévues » gérée par le ministre des finances.", "source" : "F2025007.json","parent_title":"""Décret présidentiel n° 24-440 du 29 Joumada Ethania 1446 correspondant au 31 décembre 2024"""},
        {"id": 13, "question" : "D'après l'arrêté du ministère de la culture et des arts du 21 janvier 2025, dans quelle ville algérienne le festival culturel international du théâtre du Sahara est-il officiellement institutionnalisé, et à quelle fréquence cet événement doit-il se tenir ?", "answer" : "Selon l'article 1er de cet arrêté, le festival est institutionnalisé dans la ville d'Adrar, et il s'agit d'un événement à périodicité annuelle.", "source" : "F2025007.json", "parent_title":"""Arrêté du 21 Rajab 1446 correspondant au 21 janvier 2025"""},
        {"id": 14, "question": "Dans le cadre du décret présidentiel n° 25-57 du 23 janvier 2025, quel est l'objet précis de l'accord bilatéral ratifié par l'Algérie, avec quel pays a-t-il été conclu, et à quelle date cet accord avait-il été initialement signé ?", "answer": "Le décret ratifie l'accord de coopération culturelle et scientifique conclu entre l'Algérie et la République fédérale d'Allemagne. Ce document avait été initialement signé à Alger le 13 juin 2022, soit plus de deux ans avant sa ratification par ce décret.", "source": "F025006.json","parent_title":"""Décret présidentiel n° 25-57 du 23 Rajab 1446 correspondant au 23 janvier 2025"""},
        {"id": 15, "question": "D'après l'annexe de l'accord de coopération culturelle et scientifique entre l'Algérie et l'Allemagne, à quelles conditions strictes un expert allemand détaché en Algérie peut-il importer son véhicule personnel sans payer de droits de douane, et quand aura-t-il le droit de le revendre sur place ?", "answer": "L'expert peut importer son véhicule en franchise de droits de douane à deux conditions : le véhicule doit avoir été utilisé pendant au moins 6 mois avant le transfert, et il doit être dédouané dans les 12 mois suivant son installation. Pour le revendre (ou le céder gratuitement) en Algérie, il doit obligatoirement attendre un délai de 12 mois, sauf s'il décide de payer les droits de douane au préalable (Annexe, Paragraphe 3).", "source": "F025006.json", "parent_title":"Accord de coopération culturelle et scientifique entre le Gouvernement de la République algérienne démocratique et populaire et le Gouvernement de la République fédérale d’Allemagne"},
        {"id": 16, "question": "Selon le décret exécutif du 26 janvier 2025, quelles sont les sous-directions exactes confiées respectivement à Lynda Ghoul et Farid Chaoui au sein du ministère algérien de la solidarité nationale, et qui a été nommé à la tête des systèmes d'information ?", "answer": "Lynda Ghoul a été nommée sous-directrice de l'enfance et de l'adolescence en difficulté sociale et en danger moral, tandis que Farid Chaoui a pris la sous-direction de la petite enfance et de l'enfance privée de famille. La sous-direction de la communication et des systèmes d'information a quant à elle été confiée à Ali Abderraouf El-Haffaf.", "source": "F025006.json","parent_title":"""Décret exécutif du 26 Rajab 1446 correspondant au 26 janvier 2025"""},
        {"id": 17, "question": "Dans le rectificatif publié au Journal Officiel n° 82 de décembre 2024 concernant l'avis n° 03/A.C.C/I.C/24 de la Cour constitutionnelle sur l'article 122 de la Constitution, quelle précision juridique majeure a été ajoutée concernant la restriction d'accès aux deux chambres du Parlement ?", "answer": "Le rectificatif ajoute la notion de désignation. Au lieu de limiter la restriction au seul fait de « se porter candidat » (qui ne concerne que les élus), le texte corrigé précise désormais « que nul ne peut se porter candidat ou être désigné ». Cela englobe donc formellement à la fois les parlementaires issus d'élections et ceux nommés par décret.", "source": "F025005.json", "parent_title":"""Avis n° 03/A.C.C/I.C/24 du 22 Joumada El Oula 1446 correspondant au 24 novembre 2024"""},
        {"id": 18, "question": "Selon l'arrêté interministériel du 9 décembre 2024 relatif à l'école supérieure de la sécurité sociale, que se passe-t-il très exactement si le quorum des deux tiers (2/3) n'est pas atteint lors d'une réunion de la commission sectorielle, et comment les décisions sont-elles tranchées en cas d'égalité parfaite des voix lors d'un vote ?", "answer": "D'après l'article 7, si le quorum n'est pas atteint, une deuxième réunion doit être organisée dans les huit (8) jours suivants. Lors de cette seconde réunion, la commission peut délibérer valablement quel que soit le nombre de membres présents. En cas de partage égal des voix lors d'un vote, la voix du président de la commission est prépondérante (elle tranche la décision).", "source": "F025005.json", "parent_title":"""Arrêté interministériel du 7 Joumada Ethania 1446 correspondant au 9 décembre 2024"""},
        {"id": 19, "question": "Selon le décret exécutif n° 25-55 du 21 janvier 2025, quel est le taux de l'indemnité de soutien scolaire accordé au personnel d'intendance par rapport au personnel de laboratoire, et ce dernier (le personnel de laboratoire) peut-il également percevoir l'indemnité de qualification ?", "answer": "D'après l'article 10 du décret, le personnel d'intendance et le personnel de laboratoire bénéficient tous deux du même taux, soit 15 % du traitement, pour l'indemnité de soutien scolaire et de remédiation pédagogique. En revanche, le personnel de laboratoire n'a pas droit à l'indemnité de qualification. L'article 7 précise en effet que cette indemnité est exclusivement servie aux personnels cités aux articles 3 et 4 (enseignants, direction, intendance...), excluant de fait les laborantins qui sont régis par l'article 5.", "source": "F025004.json","parent_title":"""Décret exécutif n° 25-55 du 21 Rajab 1446 correspondant au 21 janvier 2025"""},
        {"id": 20, "question": "D'après l'arrêté interministériel du 5 janvier 2025, quelle catégorie spécifique de personnel et quelle administration exacte sont visées par la modification des effectifs et de la durée des contrats ?", "answer": "Cet arrêté cible exclusivement les agents contractuels qui exercent des activités d'entretien, de maintenance ou de service. Il s'applique uniquement au niveau de l'administration centrale du ministère de la santé (historiquement désigné comme ministère de la santé, de la population et de la réforme hospitalière).", "source": "F025004.json", "parent_title":"""Arrêté interministériel du 5 Rajab 1446 correspondant au 5 janvier 2025"""},
        {"id": 21, "question": "D'selon le décret présidentiel n° 25-56 du 22 janvier 2025, à quelle date exacte se tiendra l'élection pour le renouvellement de la moitié des membres élus du Conseil de la Nation, et quelles sont les instances qui composent le collège électoral autorisé à voter ?", "answer": "D'après les articles 1 et 2 du décret, le collège électoral est convoqué pour le dimanche 9 mars 2025. Ce collège n'est pas composé de citoyens ordinaires, mais de l'ensemble des membres de l'assemblée populaire de wilaya (APW) et des membres des assemblées populaires communales (APC) de chaque wilaya.", "source": "F025003.json","parent_title":"""Décret présidentiel n° 25-56 du 22 Rajab 1446 correspondant au 22 janvier 2025"""},
        {"id": 22, "question": "D'après le décret présidentiel du 6 janvier 2025, qui a été nommé à la tête de la garde Républicaine algérienne, avec quel statut précis, et à partir de quelle date exacte cette nomination a-t-elle effectivement pris effet ?", "answer": "Le Général-major Tahar Ayad a été nommé commandant de la garde Républicaine avec le statut précis de commandant « par intérim ». Le point crucial est que, bien que le décret ait été signé le 6 janvier 2025, cette nomination a pris effet de manière rétroactive à compter du 23 décembre 2024.", "source": "F025003.json","parent_title":"""Décret présidentiel du 6 Rajab 1446 correspondant au 6 janvier 2025"""},
        {"id": 23, "question": "D'après l'arrêté du 19 décembre 2024 portant nomination au conseil d'administration du musée national du moudjahid, qui a été désigné comme président de ce conseil et quel ministère représente-t-il ? De plus, quels sont les noms exacts des représentants nommés au titre de l'organisation nationale des enfants de chouhada ?", "answer": "Selon cet arrêté, le président du conseil d'administration est Alallou Abdelhamid, qui siège en tant que représentant du ministre des moudjahidine et des ayants droit. Par ailleurs, l'organisation nationale des enfants de chouhada est exceptionnellement représentée par deux membres : Abidli Mohamed Amine et Bakhouche Mokhtar.", "source": "F025003.json","parent_title":"""Arrêté du 17 Joumada Ethania 1446 correspondant au 19 décembre 2024"""},
        {"id": 24, "question": "Selon le décret présidentiel n° 24-433 du 31 décembre 2024, quels sont les montants exacts transférés respectivement en autorisations d'engagement et en crédits de paiement au profit de la Présidence de la République, et de quelle rubrique budgétaire spécifique ces fonds ont-ils été initialement annulés ?", "answer": "Le décret prévoit le transfert de 4.192.000.000 DA en autorisations d'engagement et de 4.936.300.000 DA en crédits de paiement vers le portefeuille de programmes de la Présidence de la République. Ces fonds ont été prélevés (annulés) sur les crédits gérés par le ministre des finances, plus précisément sur la dotation « Montant non assigné » relevant du titre 7 consacré aux « Dépenses imprévues ».", "source": "F025001.json","parent_title":"""Décret présidentiel n° 24-433 du 29 Joumada Ethania 1446 correspondant au 31 décembre 2024"""},
        {"id": 25, "question": "Selon le décret présidentiel n° 25-03 du 6 janvier 2025, quelle est la durée du mandat d'un membre du Conseil (et est-ce renouvelable ?), comparée à celle d'un membre du bureau ou d'un président de commission ? De plus, à combien de commissions permanentes un membre peut-il appartenir au maximum ?", "answer": "D'après l'article 8 modifié, le mandat d'un membre du Conseil est de quatre (4) ans, renouvelable une seule fois. En revanche, les membres du bureau (article 41) et les présidents des commissions permanentes (article 45) sont élus pour un mandat de deux (2) ans, non renouvelable. Enfin, l'article 45 précise qu'un membre du Conseil ne peut faire partie de plus de deux (2) commissions permanentes.", "source": "F025001.json","parent_title":"""Décret présidentiel n° 25-03 du 6 Rajab 1446 correspondant au 6 janvier 2025"""},
        {"id": 26, "question": "D'après l'arrêté du 25 novembre 2024, quel pouvoir précis a été délégué à M. Brahim Benbouza par le ministre de l'agriculture, du développement rural et de la pêche, et quelle est l'exception stricte à cette délégation ?", "answer": "M. Brahim Benbouza, en sa qualité de directeur de l'administration des moyens, a reçu délégation pour signer tous les actes et décisions au nom du ministre, dans la limite de ses attributions. Cependant, l'article 1er pose une exception stricte : cette délégation s'applique « à l'exclusion des arrêtés ». Il n'a donc pas le pouvoir de signer des arrêtés ministériels.", "source" : "F025001.json","parent_title":"""Arrêté du 23 Joumada El Oula 1446 correspondant au 25 novembre 2024"""}
    ]
}

# --- 2. FONCTION LLM ---
def call_llm(prompt_text, system_instruction=""):
    """
    Envoie le prompt à l'API locale d'Ollama sur tes GPUs.
    """
    payload = {
        "model": LLM_MODEL, 
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt_text}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 8192  # 🚀 Command R gère excellemment bien les grands contextes
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

# --- 3. SYSTÈME RAG (SÉMANTIQUE PURE) ---
def retrieve_and_generate(question, collection, model):
    query_emb = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K
    )
    
    retrieved_docs = results['documents'][0]
    full_context = "\n\n---\n\n".join(retrieved_docs)
    
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
{question}

RÉPONSE :
"""
    
    answer = call_llm(prompt)
    return answer, retrieved_docs

# --- 4. LLM AS A JUDGE ---
def evaluate_with_judge(question, ground_truth, generated_answer):
    judge_prompt = f"""Tu es un juge impitoyable et binaire chargé d'évaluer un système d'IA juridique.
Compare la réponse générée à la réponse de référence.

QUESTION : {question}
RÉFÉRENCE EXACTE : {ground_truth}
RÉPONSE GÉNÉRÉE PAR L'IA : {generated_answer}

Règles d'évaluation :
- Si la réponse générée contient les mêmes faits et numéros d'articles que la référence : donne 1.
- Si elle est incomplète mais ne dit pas de choses fausses : donne 0.5.
- Si elle est fausse, contredit la référence, ou hallucine : donne 0.

Tu DOIS commencer ta réponse exactement par "SCORE: X" (où X est 0, 0.5 ou 1). Explique ensuite brièvement pourquoi.
"""
    
    judge_evaluation = call_llm(judge_prompt, system_instruction="Tu es un juge strict. Tu ne fais aucun sentiment.")
    
    score = 0.0
    print(judge_evaluation)
    match = re.search(r"SCORE\s*:\s*(1|0\.5|0)", judge_evaluation, re.IGNORECASE)
    if match:
        score = float(match.group(1))
    else:
        if " 1" in judge_evaluation or "1.0" in judge_evaluation: score = 1.0
        elif "0.5" in judge_evaluation: score = 0.5
        
    return score, judge_evaluation

# --- 5. EXÉCUTION DE L'ÉVALUATION ---
def run_evaluation_pipeline():
    print("🔄 Initialisation de ChromaDB et BGE-M3...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    embedding_model = SentenceTransformer(MODEL_NAME)
    
    total_score = 0
    total_questions = len(dataset["samples"])
    
    print("\n🚀 Démarrage de la pipeline LLM-as-a-Judge...\n")
    
    for sample in dataset["samples"]:
        q_id = sample["id"]
        question = sample["question"]
        ground_truth = sample["answer"]
        
        print("="*80)
        print(f"📝 Q{q_id} : {question}")
        
        # Étape 1 : RAG
        generated_answer, docs = retrieve_and_generate(question, collection, embedding_model)
        
        # Étape 2 : Juge
        score, judge_feedback = evaluate_with_judge(question, ground_truth, generated_answer)
        total_score += score
        
        print(f"🤖 Rép. Générée : {generated_answer[:150]}...")
        print(f"⚖️ Verdict Juge : {judge_feedback.strip()}")
        print(f"📊 Score : {score}/1.0")
        
    print("\n" + "="*80)
    print("📈 RÉSULTATS FINAUX (LLM AS A JUDGE)")
    print("="*80)
    print(f"Score Global : {total_score} / {total_questions}")
    print(f"Précision    : {(total_score / total_questions) * 100:.2f}%")

if __name__ == "__main__":
    run_evaluation_pipeline()
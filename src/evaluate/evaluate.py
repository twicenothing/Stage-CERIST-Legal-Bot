import chromadb
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
CHROMA_PATH = "../../data/chroma_db"
COLLECTION_NAME = "legal_algeria"
TOP_K_EVAL = 3 # On vérifie si la bonne source est dans le Top 3 (ce qu'on envoie à Mistral)

dataset = {
    "samples": [
        {"id": 1, "question": "Pourquoi la juridiction suprême a-t-elle refusé d'accéder à la requête des parlementaires concernant l'article 158 ?", "source": "F2025009.json"},
        {"id": 2, "question": "Quel est le danger potentiel souligné par les juges s'ils acceptaient d'expliquer une disposition constitutionnelle qui est déjà évidente ?", "source": "F2025009.json"},
        {"id": 3, "question": "Quel est l'impact pratique de l'arrêté de janvier 2025 sur la gestion quotidienne des fonds alloués à l'Inspection générale des finances, et quelle capacité d'action spécifique est transférée à M. Saïd Touakni ?", "source": "F2025009.json"},
        {"id": 4, "question": "À quel programme, sous-programme et titre exacts est applicable le montant ouvert de trente-neuf millions de dinars (39.000.000 DA) pour le portefeuille du ministère des transports ?", "source": "F2025005.json"},
        {"id": 5, "question": "Quelle est la nouvelle échéance accordée aux groupements d'agriculteurs pour se mettre en règle avec la législation de 1996 qui les régit ?", "source": "F2025009.json"},
        {"id": 6, "question": "Au sein de la commission sectorielle chargée de la tutelle pédagogique sur l'école supérieure de la sécurité sociale, que se passe-t-il si la plupart des membres sont absents lors d'une réunion, et comment les décisions sont-elles tranchées en cas d'égalité des votes ?", "source": "F2025005.json"},
        {"id": 7, "question": "La liste des membres du Conseil national économique, social et environnemental (CNESE) nommés en janvier 2025 est-elle complète, et quelle est la durée de leur mandat ?", "source": "F2025005.json"},
        {"id": 8, "question": "De quelles manières exactes le ministre de la jeunesse doit-il intervenir en faveur de la jeunesse algérienne établie hors du pays, selon le décret de février 2025 fixant ses attributions ?", "source": "F2025010.json"},
        {"id": 9, "question": "Monsieur Hamid Benazouz a reçu l'autorisation de valider de nombreuses opérations financières et administratives à la place de la ministre. Cependant, quelle est la limite stricte de cette délégation et quel type de document n'a-t-il absolument pas le droit de signer ?", "source": "F2025010.json"},
        {"id": 10, "question": "Dans le cadre de la convention d'extradition signée en 2021 entre l'Algérie et la Tunisie, un individu recherché pour des actes terroristes ou pour une tentative d'assassinat sur un membre du Gouvernement peut-il bloquer son extradition en affirmant qu'il s'agit d'un crime politique ?", "source": "F2025008.json"},
        {"id": 11, "question": "Selon l'accord de décembre 2023 entre l'Algérie et l'Indonésie, un diplomate algérien officiellement affecté pour travailler à l'ambassade d'Algérie à Jakarta a-t-il besoin d'un visa pour sa première entrée sur le territoire indonésien avec son passeport diplomatique ?", "source": "F2025008.json"},
        {"id": 12, "question": "D'après le décret présidentiel n° 24-440 du 31 décembre 2024, quels sont les montants exacts en autorisations d'engagement et en crédits de paiement qui ont été transférés à la Présidence de la République, et de quelle rubrique budgétaire spécifique du ministère des finances provenaient ces fonds ?", "source": "F2025007.json"},
        {"id": 13, "question": "D'après l'arrêté du ministère de la culture et des arts du 21 janvier 2025, dans quelle ville algérienne le festival culturel international du théâtre du Sahara est-il officiellement institutionnalisé, et à quelle fréquence cet événement doit-il se tenir ?", "source": "F2025007.json"},
        {"id": 14, "question": "Dans le cadre du décret présidentiel n° 25-57 du 23 janvier 2025, quel est l'objet précis de l'accord bilatéral ratifié par l'Algérie, avec quel pays a-t-il été conclu, et à quelle date cet accord avait-il été initialement signé ?", "source": "F2025006.json"},
        {"id": 15, "question": "D'après l'annexe de l'accord de coopération culturelle et scientifique entre l'Algérie et l'Allemagne, à quelles conditions strictes un expert allemand détaché en Algérie peut-il importer son véhicule personnel sans payer de droits de douane, et quand aura-t-il le droit de le revendre sur place ?", "source": "F2025006.json"},
        {"id": 16, "question": "Selon le décret exécutif du 26 janvier 2025, quelles sont les sous-directions exactes confiées respectivement à Lynda Ghoul et Farid Chaoui au sein du ministère algérien de la solidarité nationale, et qui a été nommé à la tête des systèmes d'information ?", "source": "F2025006.json"},
        {"id": 17, "question": "Dans le rectificatif publié au Journal Officiel n° 82 de décembre 2024 concernant l'avis n° 03/A.C.C/I.C/24 de la Cour constitutionnelle sur l'article 122 de la Constitution, quelle précision juridique majeure a été ajoutée concernant la restriction d'accès aux deux chambres du Parlement ?", "source": "F2025005.json"},
        {"id": 18, "question": "Selon l'arrêté interministériel du 9 décembre 2024 relatif à l'école supérieure de la sécurité sociale, que se passe-t-il très exactement si le quorum des deux tiers (2/3) n'est pas atteint lors d'une réunion de la commission sectorielle, et comment les décisions sont-elles tranchées en cas d'égalité parfaite des voix lors d'un vote ?", "source": "F2025005.json"},
        {"id": 19, "question": "Selon le décret exécutif n° 25-55 du 21 janvier 2025, quel est le taux de l'indemnité de soutien scolaire accordé au personnel d'intendance par rapport au personnel de laboratoire, et ce dernier (le personnel de laboratoire) peut-il également percevoir l'indemnité de qualification ?", "source": "F2025004.json"},
        {"id": 20, "question": "D'après l'arrêté interministériel du 5 janvier 2025, quelle catégorie spécifique de personnel et quelle administration exacte sont visées par la modification des effectifs et de la durée des contrats ?", "source": "F2025004.json"},
        {"id": 21, "question": "D'selon le décret présidentiel n° 25-56 du 22 janvier 2025, à quelle date exacte se tiendra l'élection pour le renouvellement de la moitié des membres élus du Conseil de la Nation, et quelles sont les instances qui composent le collège électoral autorisé à voter ?", "source": "F2025003.json"},
        {"id": 22, "question": "D'après le décret présidentiel du 6 janvier 2025, qui a été nommé à la tête de la garde Républicaine algérienne, avec quel statut précis, et à partir de quelle date exacte cette nomination a-t-elle effectivement pris effet ?", "source": "F2025003.json"},
        {"id": 23, "question": "D'après l'arrêté du 19 décembre 2024 portant nomination au conseil d'administration du musée national du moudjahid, qui a été désigné comme président de ce conseil et quel ministère représente-t-il ? De plus, quels sont les noms exacts des représentants nommés au titre de l'organisation nationale des enfants de chouhada ?", "source": "F2025003.json"},
        {"id": 24, "question": "Selon le décret présidentiel n° 24-433 du 31 décembre 2024, quels sont les montants exacts transférés respectivement en autorisations d'engagement et en crédits de paiement au profit de la Présidence de la République, et de quelle rubrique budgétaire spécifique ces fonds ont-ils été initialement annulés ?", "source": "F2025001.json"},
        {"id": 25, "question": "Selon le décret présidentiel n° 25-03 du 6 janvier 2025, quelle est la durée du mandat d'un membre du Conseil (et est-ce renouvelable ?), comparée à celle d'un membre du bureau ou d'un président de commission ? De plus, à combien de commissions permanentes un membre peut-il appartenir au maximum ?", "source": "F2025001.json"},
        {"id": 26, "question": "D'après l'arrêté du 25 novembre 2024, quel pouvoir précis a été délégué à M. Brahim Benbouza par le ministre de l'agriculture, du développement rural et de la pêche, et quelle est l'exception stricte à cette délégation ?", "source": "F2025001.json"}
    ]
}

# --- UTILS ---
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


# --- MAIN EVALUATION ---
def test_retrieval():
    print(f"🔌 Connexion à ChromaDB ({CHROMA_PATH})...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    print("🤖 Chargement du modèle d'embedding BAAI/bge-m3...")
    embedding_model = SentenceTransformer("BAAI/bge-m3", device="cuda") # Retire device="cuda" si tu es sur CPU

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

    wins = 0
    total_questions = len(dataset["samples"])

    print("\n" + "="*70)
    print(f"🚀 DÉBUT DU TEST DE RETRIEVAL ({total_questions} questions)")
    print("="*70 + "\n")

    for sample in dataset["samples"]:
        q_id = sample["id"]
        query = sample["question"]
        expected_source = sample["source"]

        print(f"▶️ [Q{q_id}] QUESTION : {query}")
        print(f"🎯 SOURCE ATTENDUE : {expected_source}")

        # --- A. Vector Search (Top 10) ---
        q_embed = embedding_model.encode([query]).tolist()
        vec_res = collection.query(query_embeddings=q_embed, n_results=10)
        vec_list = []
        if vec_res['ids'] and len(vec_res['ids']) > 0:
            for i in range(len(vec_res['ids'][0])):
                vec_list.append((vec_res['ids'][0][i], vec_res['documents'][0][i], vec_res['metadatas'][0][i]))

        # --- B. Keyword Search BM25 (Top 10) ---
        tokenized_query = normalize_text(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(doc_scores)[::-1][:10]
        kw_list = []
        for idx in top_n_indices:
            if doc_scores[idx] > 0:
                kw_list.append((ids_all[idx], documents_all[idx], metadatas_all[idx]))

        # --- C. Fusion RRF ---
        best_weights = {"keyword": 0.3, "vector": 0.7}
        final_results = reciprocal_rank_fusion(
            {"vector": vec_list, "keyword": kw_list}, 
            weights=best_weights
        )

        # --- D. Vérification et Logs ---
        # On regarde uniquement le Top K défini (ex: Top 3)
        top_results = final_results[:TOP_K_EVAL]
        
        is_win = False
        print("-" * 40)
        print(f"📊 TOP {TOP_K_EVAL} RÉSULTATS RETOURNÉS :")
        
        for rank, (doc_id, data) in enumerate(top_results):
            meta = data['meta']
            score = data['score']
            
            # Gestion de la clé source selon comment tu l'as nommée dans ton JSON
            # Ça peut être 'source_file', 'source', ou 'fichier'
            actual_source = meta.get('source_file') or meta.get('source') or meta.get('file', 'INCONNU')
            doc_title = meta.get('parent_title', meta.get('title', 'Sans Titre'))[:60] + "..."
            
            # Vérification du Win
            match_status = "❌"
            if actual_source == expected_source:
                is_win = True
                match_status = "✅"

            print(f"  [{rank+1}] Score: {score:.4f} | Source: {actual_source} {match_status} | Titre: {doc_title}")

        print("-" * 40)
        if is_win:
            print("🏆 RÉSULTAT : WIN (La bonne source est dans le Top !)")
            wins += 1
        else:
            print("💀 RÉSULTAT : LOSS (Source introuvable dans le Top...)")
        
        print("="*70 + "\n")

    # --- BILAN FINAL ---
    print("📈 BILAN DU TEST DE RETRIEVAL PUREMENT SÉMANTIQUE/MOT-CLÉ")
    print(f"Succès : {wins} / {total_questions} ({(wins/total_questions)*100:.2f}%)")
    print("="*70)

if __name__ == "__main__":
    test_retrieval()
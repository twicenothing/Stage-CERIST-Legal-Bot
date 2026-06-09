import os
import re
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION BASE DE DONNÉES ---
CHROMA_PATH = os.getenv("CHROMA_PATH")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "legal_algeria")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# --- PARAMÈTRES ---
FALLBACK_DISTANCE_THRESHOLD = 1.05


def is_table_query(query: str) -> bool:
    """
    Detects questions that are likely to need table chunks.
    This does not replace vector search. It only decides how many table chunks
    we should retrieve.
    """
    q = query.lower()

    table_keywords = [
    # Explicit table/annex language
    "tableau",
    "annexe",
    "liste",
    "barème",
    "bareme",
    "nomenclature",
    "répartition",
    "repartition",
    "chapitre",
    "rubrique",

    # Numeric question intent
    "combien",
    "nombre",
    "total",
    "effectif",
    "effectifs",
    "montant",
    "taux",
    "tarif",
    "tarifs",
    "quota",
    "plafond",
    "seuil",
    "coefficient",
    "valeur",
    "volume",
    "superficie",
    "surface",
    "capacité",
    "capacite",

    # Money / budget
    "da",
    "dinars",
    "crédits",
    "credits",
    "crédits ouverts",
    "credits ouverts",
    "crédits de paiement",
    "credits de paiement",
    "autorisation d'engagement",
    "autorisations d'engagement",
    "dépenses",
    "depenses",
    "budget",
    "milliers de da",

    # Units
    "tonnes",
    "kg",
    "kilogrammes",
    "m3",
    "m2",
    "hectares",
    "sièges",
    "sieges",

    # Tariff/customs/product tables
    "position tarifaire",
    "sous-position",
    "sous-position tarifaire",
    "tarifaire",
    "marchandise",
    "produit",
    "produits",
    "désignation des produits",
    "designation des produits",
    "dénomination commune internationale",
    "denomination commune internationale",
    "forme dosage",

    # Staffing / employment tables
    "agent",
    "agents",
    "agents contractuels",
    "contrat",
    "contrat à durée",
    "contrat a duree",
    "cdi",
    "cdd",
    "poste",
    "postes",
    "postes de travail",
    "emploi",
    "grade",
    "niveau",
    "catégorie",
    "categorie",
    "point indiciaire",
    "indemnité",
    "indemnite",
    "prime",

    # Names / institutions / locations
    "nom",
    "prénom",
    "prenom",
    "nom et prénom",
    "organisme",
    "établissement",
    "etablissement",
    "dénomination de l'établissement",
    "denomination de l'etablissement",
    "wilaya",
    "commune",

    # Infrastructure / project tables
    "péage",
    "peage",
    "gare de péage",
    "gares de péage",
    "aire de service",
    "aires de service",
    "périmètre de protection",
    "perimetre de protection",
    "zone",
    ]

    # Tariff-like codes: 0201.10.11.00, 87.01, EX 2009.89.91.90, etc.
    tariff_code_pattern = r"\b(?:ex\s*)?\d{2,4}(?:\.\d{2}){1,4}\b"

    return any(k in q for k in table_keywords) or bool(re.search(tariff_code_pattern, q, re.IGNORECASE))


def retrieve_vector_chunks(q_embed, chunk_type, collection, top_k):
    """Effectue une recherche vectorielle filtrée par chunking_method."""
    vec_res = collection.query(
        query_embeddings=q_embed,
        n_results=top_k,
        where={"chunking_method": chunk_type}
    )

    formatted_results = []
    top_distance = 999.0

    if vec_res["ids"] and len(vec_res["ids"][0]) > 0:
        top_distance = vec_res["distances"][0][0]

        for i in range(len(vec_res["ids"][0])):
            formatted_results.append({
                "id": vec_res["ids"][0][i],
                "text": vec_res["documents"][0][i],
                "meta": vec_res["metadatas"][0][i],
                "distance": vec_res["distances"][0][i],
                "retrieval_source": chunk_type
            })

    return formatted_results, top_distance


def merge_and_deduplicate(results):
    """
    Merge results from regex/table/recursive searches.
    If the same ID appears twice, keep the one with the best distance.
    """
    by_id = {}

    for doc in results:
        doc_id = doc["id"]

        if doc_id not in by_id:
            by_id[doc_id] = doc
        else:
            if doc.get("distance", 999.0) < by_id[doc_id].get("distance", 999.0):
                by_id[doc_id] = doc

    return sorted(by_id.values(), key=lambda x: x.get("distance", 999.0))


def get_retrieved_documents(
    query,
    model,
    collection,
    top_k=30,
    threshold=FALLBACK_DISTANCE_THRESHOLD
):
    """
    Main retrieval function.

    Strategy:
    1. Always search regex chunks.
    2. Always search a small number of table chunks.
    3. If the query looks table-related, search more table chunks.
    4. If regex + table results are weak, add recursive chunks as fallback.
    5. Return merged candidates for reranking.
    """

    q_embed = model.encode([query]).tolist()
    table_like = is_table_query(query)

    # -----------------------------
    # PLAN A: regex + table chunks
    # -----------------------------

    regex_results, regex_top_dist = retrieve_vector_chunks(
        q_embed,
        "regex",
        collection,
        top_k
    )

    if table_like:
        table_row_k = top_k
        table_full_k = max(5, top_k // 4)
    else:
        table_row_k = max(5, top_k // 4)
        table_full_k = max(3, top_k // 6)

    table_row_results, table_row_top_dist = retrieve_vector_chunks(
        q_embed,
        "table_row",
        collection,
        table_row_k
    )

    table_full_results, table_full_top_dist = retrieve_vector_chunks(
        q_embed,
        "table_full",
        collection,
        table_full_k
    )

    initial_results = merge_and_deduplicate(
        regex_results + table_row_results + table_full_results
    )

    best_initial_distance = min(
        [doc.get("distance", 999.0) for doc in initial_results],
        default=999.0
    )

    strategy_parts = []

    if regex_results:
        strategy_parts.append("regex")

    if table_row_results or table_full_results:
        strategy_parts.append("table")

    # -----------------------------
    # PLAN B: recursive fallback
    # -----------------------------

    if not initial_results or best_initial_distance > threshold:
        recursive_results, _ = retrieve_vector_chunks(
            q_embed,
            "recursive",
            collection,
            top_k
        )

        final_results = merge_and_deduplicate(initial_results + recursive_results)

        if recursive_results:
            strategy_parts.append("recursive_fallback")
    else:
        final_results = initial_results

    strategy_used = "+".join(strategy_parts) if strategy_parts else "none"

    # For table-like questions, allow more candidates into the reranker.
    # The CrossEncoder will choose the final best ones.
    max_candidates = top_k * 2 if table_like else top_k

    return final_results[:max_candidates], strategy_used
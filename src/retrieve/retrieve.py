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
FALLBACK_DISTANCE_THRESHOLD = float(os.getenv("FALLBACK_DISTANCE_THRESHOLD", "1.05"))

# If a candidate has distance above this, it is considered weak enough
# to justify adding recursive alternatives.
WEAK_CANDIDATE_DISTANCE_THRESHOLD = float(
    os.getenv("WEAK_CANDIDATE_DISTANCE_THRESHOLD", "0.95")
)

# Huge regex chunks that look like table annexes are likely bad swallowed table chunks.
HUGE_REGEX_CHAR_THRESHOLD = int(
    os.getenv("HUGE_REGEX_CHAR_THRESHOLD", "18000")
)

# Avoid retrieving too many recursive chunks.
MAX_RECURSIVE_AUGMENT_K = int(
    os.getenv("MAX_RECURSIVE_AUGMENT_K", "30")
)


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

    tariff_code_pattern = r"\b(?:ex\s*)?\d{2,4}(?:\.\d{2}){1,4}\b"

    return any(k in q for k in table_keywords) or bool(
        re.search(tariff_code_pattern, q, re.IGNORECASE)
    )


def looks_like_tabular_text(text: str) -> bool:
    """
    Detects chunks that look like table/annex data even if metadata says regex.
    Used only to decide whether recursive chunks should be added as alternatives.
    """
    lower = (text or "").lower()

    signals = [
        "tableau annexe",
        "crédits ouverts",
        "credits ouverts",
        "crédits ouverts en da",
        "credits ouverts en da",
        "nos des chapitres",
        "libelles",
        "libellés",
        "répartition par chapitre",
        "repartition par chapitre",
        "effectifs selon la nature",
        "contrat à durée",
        "contrat a duree",
        "total général",
        "total general",
        "|---",
    ]

    return any(s in lower for s in signals)


def is_noisy_candidate(doc) -> bool:
    """
    Marks candidates that should trigger recursive augmentation.

    We do NOT delete these candidates here.
    We only use this to decide whether to add recursive candidates.
    The reranker will decide later.
    """
    meta = doc.get("meta", {}) or {}
    method = meta.get("chunking_method", "")
    text = doc.get("text", "") or ""
    distance = float(doc.get("distance", 999.0) or 999.0)

    weak_by_distance = distance > WEAK_CANDIDATE_DISTANCE_THRESHOLD

    huge_tabular_regex = (
        method == "regex"
        and len(text) > HUGE_REGEX_CHAR_THRESHOLD
        and looks_like_tabular_text(text)
    )

    return weak_by_distance or huge_tabular_regex


def retrieve_vector_chunks(q_embed, chunk_type, collection, top_k):
    """Effectue une recherche vectorielle filtrée par chunking_method."""
    vec_res = collection.query(
        query_embeddings=q_embed,
        n_results=top_k,
        where={"chunking_method": chunk_type},
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
                "retrieval_source": chunk_type,
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
    threshold=FALLBACK_DISTANCE_THRESHOLD,
):
    """
    Main retrieval function.

    Strategy:
    1. Always search regex chunks.
    2. Always search table chunks.
    3. If the query looks table-related, search more table chunks.
    4. Add recursive chunks if:
       - regex/table retrieval is globally weak, OR
       - some initial candidates are weak/noisy.
    5. Return merged candidates for reranking.

    Important:
    - Weak/noisy candidates are not removed here.
    - Recursive chunks are added as alternatives.
    - The reranker decides the final top docs.
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
        top_k,
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
        table_row_k,
    )

    table_full_results, table_full_top_dist = retrieve_vector_chunks(
        q_embed,
        "table_full",
        collection,
        table_full_k,
    )

    initial_results = merge_and_deduplicate(
        regex_results + table_row_results + table_full_results
    )

    best_initial_distance = min(
        [doc.get("distance", 999.0) for doc in initial_results],
        default=999.0,
    )

    strategy_parts = []

    if regex_results:
        strategy_parts.append("regex")

    if table_row_results or table_full_results:
        strategy_parts.append("table")

    # -----------------------------
    # PLAN B: recursive augmentation
    # -----------------------------

    weak_or_noisy_docs = [doc for doc in initial_results if is_noisy_candidate(doc)]

    should_add_recursive = (
        not initial_results
        or best_initial_distance > threshold
        or len(weak_or_noisy_docs) > 0
    )

    if should_add_recursive:
        # If retrieval is globally weak, get a full recursive top_k.
        # If only some candidates are weak/noisy, get enough recursive alternatives
        # without flooding the candidate pool.
        if not initial_results or best_initial_distance > threshold:
            recursive_k = top_k
            recursive_reason = "global_fallback"
        else:
            recursive_k = min(
                MAX_RECURSIVE_AUGMENT_K,
                max(5, len(weak_or_noisy_docs) * 2),
            )
            recursive_reason = f"weak_or_noisy_candidates={len(weak_or_noisy_docs)}"

        recursive_results, _ = retrieve_vector_chunks(
            q_embed,
            "recursive",
            collection,
            recursive_k,
        )

        final_results = merge_and_deduplicate(initial_results + recursive_results)

        if recursive_results:
            strategy_parts.append(f"recursive_augment({recursive_reason})")

        print(
            f"🧩 Recursive augmentation: reason={recursive_reason}, "
            f"recursive_k={recursive_k}, recursive_results={len(recursive_results)}"
        )
    else:
        final_results = initial_results

    strategy_used = "+".join(strategy_parts) if strategy_parts else "none"

    # For table-like questions, allow more candidates into the reranker.
    # Recursive augmentation can increase the pool, but we still cap it.
    max_candidates = top_k * 2 if table_like else top_k

    return final_results[:max_candidates], strategy_used
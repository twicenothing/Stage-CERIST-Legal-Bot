import os
import re
import csv
import json
import math
import argparse
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict


DEFAULT_STOPWORDS = {
    # French common words
    "a", "à", "au", "aux", "avec", "ce", "ces", "cet", "cette", "dans", "de", "des",
    "du", "d", "en", "et", "est", "la", "le", "les", "l", "un", "une", "par", "pour",
    "sur", "sous", "ou", "où", "que", "qui", "dont", "se", "sa", "son", "ses", "leur",
    "leurs", "il", "elle", "ils", "elles", "ne", "pas", "plus", "moins", "ainsi",
    "comme", "entre", "selon", "relatif", "relative", "fixant", "portant",

    # Arabic transliteration / common legal noise
    "correspondant", "rabie", "joumada", "moharram", "safar", "chaoual", "dhou",
    "kaada", "kaâda", "hidja", "rajab", "ramadhan", "ramadan",

    # Boilerplate created by your table chunks
    "source", "page", "tableau", "ligne", "col", "document", "inconnu",
    "journal", "officiel", "republique", "république", "algerienne", "algérienne",

    # Too generic
    "article", "art", "annexe", "suite", "etat", "état", "numero", "numéro",
}


KEEP_SHORT_TERMS = {
    "da", "m2", "m3", "kg", "km", "ht", "ttc",
}


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_key(term: str) -> str:
    term = term.lower()
    term = term.replace("’", "'").replace("`", "'")
    term = strip_accents(term)
    term = re.sub(r"[^a-z0-9'\- ]+", " ", term)
    term = normalize_space(term)
    return term


def tokenize(text: str) -> list[str]:
    text = str(text or "").lower()
    text = text.replace("’", "'").replace("`", "'")
    # Keep accented French letters, digits, apostrophes, hyphens.
    tokens = re.findall(r"[a-zàâäéèêëîïôöùûüç0-9]+(?:['\-][a-zàâäéèêëîïôöùûüç0-9]+)?", text)
    return [t for t in tokens if t.strip()]


def is_bad_single_token(token: str) -> bool:
    key = normalize_key(token)

    if key in KEEP_SHORT_TERMS:
        return False

    if key in DEFAULT_STOPWORDS:
        return True

    if len(key) < 3:
        return True

    if key.isdigit():
        return True

    return False


def valid_ngram(tokens: list[str]) -> bool:
    if not tokens:
        return False

    # Do not allow stopwords at edges.
    if is_bad_single_token(tokens[0]) or is_bad_single_token(tokens[-1]):
        return False

    key = normalize_key(" ".join(tokens))

    if not key:
        return False

    # Avoid pure numeric/code-like ngrams. Tariff codes are better handled by regex.
    if re.fullmatch(r"[0-9.\- ]+", key):
        return False

    # Avoid very long phrases.
    if len(key) > 80:
        return False

    return True


def extract_ngrams(text: str, max_n: int = 3) -> list[tuple[str, str]]:
    """
    Returns list of (normalized_key, display_term).
    Keeps stopwords inside phrases but not at the edges.
    Example: "aire de service" can survive.
    """
    tokens = tokenize(text)
    results = []

    for n in range(1, max_n + 1):
        for i in range(0, len(tokens) - n + 1):
            phrase_tokens = tokens[i:i + n]

            if not valid_ngram(phrase_tokens):
                continue

            display = " ".join(phrase_tokens)
            key = normalize_key(display)

            # Extra boilerplate filtering
            if key in DEFAULT_STOPWORDS:
                continue

            if key in {"source page", "page tableau", "tableau ligne", "source page tableau"}:
                continue

            results.append((key, display))

    return results


def iter_json_files(folder: Path):
    if not folder.exists():
        return

    for path in sorted(folder.rglob("*.json")):
        yield path


def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Could not read {path}: {e}")
        return None


def update_stats(
    stats,
    term_display_counter,
    table_id,
    table_kind,
    text,
    source_type,
    weight=1.0,
    max_n=3,
):
    """
    source_type examples:
    - header_caption
    - cell_value
    - table_full_chunk
    - table_row_chunk
    """
    if not text or not str(text).strip():
        return

    seen_in_this_text = set()

    for key, display in extract_ngrams(text, max_n=max_n):
        term_display_counter[key][display] += 1

        stats[key]["occurrences"] += 1
        stats[key]["weighted_occurrences"] += weight
        stats[key]["source_type_counts"][source_type] += 1

        if table_kind:
            stats[key]["table_kind_counts"][table_kind] += 1

        if table_id:
            stats[key]["table_ids"].add(table_id)

        if key not in seen_in_this_text:
            stats[key]["text_hits"] += 1
            seen_in_this_text.add(key)


def load_structured_tables(tables_dir: Path, stats, term_display_counter, max_n: int, include_cells: bool):
    table_count = 0
    kind_counter = Counter()

    for path in iter_json_files(tables_dir) or []:
        data = safe_load_json(path)
        if not data:
            continue

        for table in data.get("tables", []):
            table_id = table.get("table_id") or f"{path.name}_{table_count}"
            table_kind = table.get("table_kind", "unknown")
            caption = table.get("caption", "")
            headers = table.get("headers", []) or []

            table_count += 1
            kind_counter[table_kind] += 1

            # Headers and captions are the best place to discover table query keywords.
            header_caption_text = " ".join([caption] + headers)
            update_stats(
                stats,
                term_display_counter,
                table_id=table_id,
                table_kind=table_kind,
                text=header_caption_text,
                source_type="header_caption",
                weight=5.0,
                max_n=max_n,
            )

            if include_cells:
                # Lower weight because cell values can contain very specific names/places.
                rows = table.get("rows", []) or []
                for row in rows:
                    cells = row.get("cells", {}) or {}
                    cell_text = " ".join(str(v) for v in cells.values() if str(v).strip())
                    update_stats(
                        stats,
                        term_display_counter,
                        table_id=table_id,
                        table_kind=table_kind,
                        text=cell_text,
                        source_type="cell_value",
                        weight=0.5,
                        max_n=max_n,
                    )

    return table_count, kind_counter


def load_table_chunks(table_chunks_dir: Path, stats, term_display_counter, max_n: int):
    chunk_count = 0
    method_counter = Counter()

    for path in iter_json_files(table_chunks_dir) or []:
        data = safe_load_json(path)
        if not data:
            continue

        for chunk in data.get("chunks", []):
            text = chunk.get("text", "")
            meta = chunk.get("metadata", {}) or {}

            table_id = meta.get("table_id") or chunk.get("id") or path.name
            table_kind = meta.get("table_kind", "unknown")
            method = meta.get("chunking_method", "unknown")

            chunk_count += 1
            method_counter[method] += 1

            if method == "table_full":
                source_type = "table_full_chunk"
                weight = 2.0
            elif method == "table_row":
                source_type = "table_row_chunk"
                weight = 1.0
            else:
                source_type = "table_chunk"
                weight = 1.0

            update_stats(
                stats,
                term_display_counter,
                table_id=table_id,
                table_kind=table_kind,
                text=text,
                source_type=source_type,
                weight=weight,
                max_n=max_n,
            )

    return chunk_count, method_counter


def extract_non_table_texts_from_json(data: dict):
    """
    Optional contrast mode:
    Supports common regex/recursive JSON structures.
    This is only used if you pass --text-json-dirs.
    """
    texts = []

    if "documents" in data:
        for doc in data.get("documents", []):
            title = doc.get("title", "")
            context = doc.get("context", "")
            articles = doc.get("articles", [])

            if articles:
                for article in articles:
                    texts.append(f"{title}\n{article}")
            elif context:
                texts.append(f"{title}\n{context}")

    if "chunks" in data:
        for chunk in data.get("chunks", []):
            meta = chunk.get("metadata", {}) or {}
            method = meta.get("chunking_method", "")

            # Skip table chunks if mixed in.
            if method in {"table_row", "table_full"}:
                continue

            text = chunk.get("text", "")
            if text:
                texts.append(text)

    return texts


def build_contrast_counts(text_json_dirs: list[Path], max_n: int):
    """
    Counts ngram document frequency in non-table chunks.
    Helps find terms that are table-specific rather than generally common.
    """
    text_df = Counter()
    total_docs = 0

    for folder in text_json_dirs:
        if not folder.exists():
            print(f"⚠️ Contrast folder not found: {folder}")
            continue

        for path in iter_json_files(folder) or []:
            data = safe_load_json(path)
            if not data:
                continue

            texts = extract_non_table_texts_from_json(data)

            for text in texts:
                keys = set(key for key, _ in extract_ngrams(text, max_n=max_n))
                if not keys:
                    continue

                total_docs += 1
                for key in keys:
                    text_df[key] += 1

    return text_df, total_docs


def score_terms(stats, term_display_counter, table_count, text_df=None, text_doc_count=0, min_table_df=3):
    rows = []

    eps = 1e-9

    for key, s in stats.items():
        table_df = len(s["table_ids"])

        if table_df < min_table_df:
            continue

        occurrences = s["occurrences"]
        weighted_occurrences = s["weighted_occurrences"]
        header_hits = s["source_type_counts"].get("header_caption", 0)
        row_hits = s["source_type_counts"].get("table_row_chunk", 0)
        full_hits = s["source_type_counts"].get("table_full_chunk", 0)
        cell_hits = s["source_type_counts"].get("cell_value", 0)

        display = term_display_counter[key].most_common(1)[0][0]

        phrase_len = len(key.split())
        phrase_boost = 1.0 + (0.35 * (phrase_len - 1))

        table_ratio = table_df / max(table_count, 1)

        if text_df is not None and text_doc_count > 0:
            non_table_df = text_df.get(key, 0)
            non_table_ratio = non_table_df / max(text_doc_count, 1)
            lift = (table_ratio + eps) / (non_table_ratio + eps)

            # High lift means more table-specific.
            lift_component = math.log1p(lift)
        else:
            non_table_df = None
            non_table_ratio = None
            lift = None
            lift_component = 1.0

        score = (
            (math.log1p(weighted_occurrences) * 2.0)
            + (math.log1p(table_df) * 4.0)
            + (math.log1p(header_hits) * 5.0)
            + (math.log1p(full_hits) * 1.5)
            + (math.log1p(row_hits) * 1.0)
            - (math.log1p(cell_hits) * 0.2)
        ) * phrase_boost * lift_component

        rows.append({
            "term": display,
            "normalized_term": key,
            "score": round(score, 4),
            "table_df": table_df,
            "table_df_percent": round(table_ratio * 100, 3),
            "occurrences": occurrences,
            "weighted_occurrences": round(weighted_occurrences, 2),
            "header_caption_hits": header_hits,
            "table_full_hits": full_hits,
            "table_row_hits": row_hits,
            "cell_value_hits": cell_hits,
            "non_table_df": non_table_df if non_table_df is not None else "",
            "non_table_df_percent": round(non_table_ratio * 100, 3) if non_table_ratio is not None else "",
            "table_lift": round(lift, 4) if lift is not None else "",
            "table_kind_breakdown": json.dumps(dict(s["table_kind_counts"]), ensure_ascii=False),
            "source_type_breakdown": json.dumps(dict(s["source_type_counts"]), ensure_ascii=False),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def write_outputs(rows, output_dir: Path, top: int):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "table_keyword_candidates.csv"
    json_path = output_dir / "table_keyword_candidates.json"
    txt_path = output_dir / "recommended_table_keywords.txt"
    py_path = output_dir / "recommended_table_keywords.py"

    if not rows:
        print("⚠️ No keyword candidates found.")
        return

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    recommended = [r["term"] for r in rows[:top]]

    with open(txt_path, "w", encoding="utf-8") as f:
        for term in recommended:
            f.write(term + "\n")

    with open(py_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated table keyword suggestions.\n")
        f.write("# Review manually before putting into production.\n\n")
        f.write("TABLE_KEYWORDS = [\n")
        for term in recommended:
            escaped = term.replace("\\", "\\\\").replace('"', '\\"')
            f.write(f'    "{escaped}",\n')
        f.write("]\n")

    print(f"\n✅ CSV saved:  {csv_path}")
    print(f"✅ JSON saved: {json_path}")
    print(f"✅ TXT saved:  {txt_path}")
    print(f"✅ PY saved:   {py_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze table JSON/chunk outputs to discover good table-query keywords."
    )

    parser.add_argument(
        "--tables-dir",
        default="../../data/tables",
        help="Folder containing *_tables.json files.",
    )

    parser.add_argument(
        "--table-chunks-dir",
        default="../../data/table_chunks",
        help="Folder containing *_table_chunks.json files.",
    )

    parser.add_argument(
        "--text-json-dirs",
        nargs="*",
        default=[],
        help="Optional non-table JSON folders for contrast, e.g. ../../data/json ../../data/json_recursive",
    )

    parser.add_argument(
        "--output-dir",
        default="../../data/table_keyword_stats",
        help="Output folder.",
    )

    parser.add_argument(
        "--max-n",
        type=int,
        default=3,
        help="Max ngram size.",
    )

    parser.add_argument(
        "--min-table-df",
        type=int,
        default=3,
        help="Minimum number of distinct tables a keyword must appear in.",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=120,
        help="Number of recommended keywords to export.",
    )

    parser.add_argument(
        "--no-cells",
        action="store_true",
        help="Ignore table cell values and use only headers/captions/chunks.",
    )

    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    table_chunks_dir = Path(args.table_chunks_dir)
    output_dir = Path(args.output_dir)
    text_json_dirs = [Path(p) for p in args.text_json_dirs]

    stats = defaultdict(lambda: {
        "occurrences": 0,
        "weighted_occurrences": 0.0,
        "text_hits": 0,
        "table_ids": set(),
        "source_type_counts": Counter(),
        "table_kind_counts": Counter(),
    })

    term_display_counter = defaultdict(Counter)

    print("🔎 Loading structured tables...")
    table_count, kind_counter = load_structured_tables(
        tables_dir=tables_dir,
        stats=stats,
        term_display_counter=term_display_counter,
        max_n=args.max_n,
        include_cells=not args.no_cells,
    )

    print("🔎 Loading table chunks...")
    chunk_count, method_counter = load_table_chunks(
        table_chunks_dir=table_chunks_dir,
        stats=stats,
        term_display_counter=term_display_counter,
        max_n=args.max_n,
    )

    text_df = None
    text_doc_count = 0

    if text_json_dirs:
        print("🔎 Loading non-table JSONs for contrast...")
        text_df, text_doc_count = build_contrast_counts(
            text_json_dirs=text_json_dirs,
            max_n=args.max_n,
        )

    rows = score_terms(
        stats=stats,
        term_display_counter=term_display_counter,
        table_count=max(table_count, 1),
        text_df=text_df,
        text_doc_count=text_doc_count,
        min_table_df=args.min_table_df,
    )

    print("\n" + "=" * 80)
    print("📊 TABLE KEYWORD ANALYSIS")
    print("=" * 80)
    print(f"Tables analyzed:       {table_count}")
    print(f"Table chunks analyzed: {chunk_count}")
    print(f"Table kinds:           {dict(kind_counter)}")
    print(f"Chunk methods:         {dict(method_counter)}")

    if text_json_dirs:
        print(f"Non-table docs:        {text_doc_count}")

    print("\nTop candidates:")
    for i, row in enumerate(rows[:30], start=1):
        print(
            f"{i:02d}. {row['term']:<35} "
            f"score={row['score']:<8} "
            f"table_df={row['table_df']:<5} "
            f"headers={row['header_caption_hits']:<5} "
            f"kind={row['table_kind_breakdown']}"
        )

    write_outputs(rows, output_dir, top=args.top)


if __name__ == "__main__":
    main()
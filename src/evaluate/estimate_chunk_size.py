import os
import json
import csv
import argparse
import statistics
from pathlib import Path
from collections import Counter


DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")


def load_tokenizer(model_name: str):
    """
    Loads the tokenizer of the embedding model.
    This gives much better estimates than character counting.
    Falls back to char-based estimation if transformers/tokenizer is unavailable.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print(f"✅ Loaded tokenizer for: {model_name}")
        return tokenizer

    except Exception as e:
        print(f"⚠️ Could not load tokenizer for {model_name}: {e}")
        print("⚠️ Falling back to approximate token estimate: chars / 3.7")
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    if not text:
        return 0

    if tokenizer is None:
        return int(len(text) / 3.7)

    return len(tokenizer.encode(text, add_special_tokens=False))


def percentile(values, p):
    if not values:
        return 0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def extract_chunks_from_regex_json(data: dict, filename: str):
    """
    Matches your embedding script logic:
    - if document has articles: each article becomes one embedded chunk:
      Source: title + Contenu: article
    - if no articles: context becomes one embedded chunk
    """
    source_file = data.get("source_file", filename)
    chunks = []

    for doc_idx, doc in enumerate(data.get("documents", [])):
        title = doc.get("title", "Sans titre")
        context = doc.get("context", "")
        articles = doc.get("articles", [])
        page = doc.get("page", "Inconnu")

        parent_id = f"{source_file}_regex_doc_{doc_idx}"

        if articles:
            for art_idx, article_text in enumerate(articles):
                chunk_id = f"{parent_id}_art_{art_idx}"
                text = f"Source: {title}\nContenu: {article_text}"

                chunks.append({
                    "id": chunk_id,
                    "source_file": source_file,
                    "page": page,
                    "chunking_method": "regex",
                    "chunk_format": "article",
                    "title": title,
                    "text": text,
                })
        else:
            if context.strip():
                chunk_id = f"{parent_id}_full_context"

                chunks.append({
                    "id": chunk_id,
                    "source_file": source_file,
                    "page": page,
                    "chunking_method": "regex",
                    "chunk_format": "full_context",
                    "title": title,
                    "text": context,
                })

    return chunks


def extract_chunks_from_generic_chunks_json(data: dict, filename: str):
    """
    Supports:
    - recursive JSONs: {"chunks": [{"text": "...", "metadata": {...}}]}
    - table chunk JSONs: {"chunks": [{"text": "...", "metadata": {...}}]}
    """
    source_file = data.get("source_file", filename)
    chunks = []

    for idx, chunk in enumerate(data.get("chunks", [])):
        text = chunk.get("text", "")
        if not text.strip():
            continue

        metadata = chunk.get("metadata", {}) or {}

        chunk_id = chunk.get("id") or f"{source_file}_chunk_{idx}"

        chunks.append({
            "id": str(chunk_id),
            "source_file": metadata.get("source_file", source_file),
            "page": metadata.get("page", "Inconnu"),
            "chunking_method": metadata.get("chunking_method", "unknown"),
            "chunk_format": metadata.get("chunk_format", "unknown"),
            "title": metadata.get("parent_title", ""),
            "text": text,
        })

    return chunks


def extract_chunks_from_json_file(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = path.name

    if "documents" in data:
        return extract_chunks_from_regex_json(data, filename)

    if "chunks" in data:
        return extract_chunks_from_generic_chunks_json(data, filename)

    return []


def get_context_window_report(token_counts, context_windows):
    report = {}

    total = len(token_counts)

    for window in context_windows:
        exceeding = sum(1 for t in token_counts if t > window)
        fitting = total - exceeding

        report[str(window)] = {
            "fit_count": fitting,
            "exceed_count": exceeding,
            "exceed_percent": round((exceeding / total) * 100, 2) if total else 0,
        }

    return report


def recommend_context_window(max_tokens: int, context_windows):
    for window in context_windows:
        if max_tokens <= window:
            return window
    return None


def analyze_json_dirs(input_dirs, output_dir, model_name, embedding_context):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(model_name)

    all_rows = []

    for input_dir in input_dirs:
        input_path = Path(input_dir)

        if not input_path.exists():
            print(f"⚠️ Folder not found: {input_path}")
            continue

        json_files = sorted(input_path.glob("*.json"))

        print(f"\n📁 Scanning {input_path} ({len(json_files)} JSON files)")

        for file_path in json_files:
            try:
                chunks = extract_chunks_from_json_file(file_path)

                for chunk in chunks:
                    text = chunk["text"]
                    char_count = len(text)
                    token_count = count_tokens(text, tokenizer)

                    all_rows.append({
                        "json_file": str(file_path),
                        "chunk_id": chunk["id"],
                        "source_file": chunk["source_file"],
                        "page": chunk["page"],
                        "chunking_method": chunk["chunking_method"],
                        "chunk_format": chunk["chunk_format"],
                        "title": chunk["title"],
                        "char_count": char_count,
                        "estimated_tokens": token_count,
                        "exceeds_embedding_context": token_count > embedding_context,
                        "text_preview": text[:300].replace("\n", " "),
                    })

            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

    if not all_rows:
        print("⚠️ No chunks found.")
        return

    token_counts = [row["estimated_tokens"] for row in all_rows]
    char_counts = [row["char_count"] for row in all_rows]

    largest = max(all_rows, key=lambda r: r["estimated_tokens"])

    context_windows = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    context_report = get_context_window_report(token_counts, context_windows)
    recommended_context = recommend_context_window(largest["estimated_tokens"], context_windows)

    by_method = Counter(row["chunking_method"] for row in all_rows)

    summary = {
        "embedding_model": model_name,
        "embedding_context_tested": embedding_context,
        "tokenizer_loaded": tokenizer is not None,
        "total_chunks": len(all_rows),

        "char_stats": {
            "min": min(char_counts),
            "avg": round(statistics.mean(char_counts), 2),
            "median": statistics.median(char_counts),
            "p90": percentile(char_counts, 90),
            "p95": percentile(char_counts, 95),
            "p99": percentile(char_counts, 99),
            "max": max(char_counts),
        },

        "token_stats": {
            "min": min(token_counts),
            "avg": round(statistics.mean(token_counts), 2),
            "median": statistics.median(token_counts),
            "p90": percentile(token_counts, 90),
            "p95": percentile(token_counts, 95),
            "p99": percentile(token_counts, 99),
            "max": max(token_counts),
        },

        "chunks_by_method": dict(by_method),

        "largest_chunk": {
            "json_file": largest["json_file"],
            "chunk_id": largest["chunk_id"],
            "source_file": largest["source_file"],
            "page": largest["page"],
            "chunking_method": largest["chunking_method"],
            "chunk_format": largest["chunk_format"],
            "char_count": largest["char_count"],
            "estimated_tokens": largest["estimated_tokens"],
            "title": largest["title"],
            "text_preview": largest["text_preview"],
        },

        "context_window_fit_report": context_report,
        "recommended_min_context_for_largest_chunk": recommended_context,
    }

    csv_path = output_dir / "chunk_size_details.csv"
    json_path = output_dir / "chunk_size_summary.json"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("📊 CHUNK SIZE SUMMARY")
    print("=" * 80)
    print(f"Embedding model: {model_name}")
    print(f"Tokenizer loaded: {tokenizer is not None}")
    print(f"Total chunks: {len(all_rows)}")
    print(f"Embedding context tested: {embedding_context} tokens")

    print("\nToken stats:")
    print(f"  Min:    {summary['token_stats']['min']}")
    print(f"  Avg:    {summary['token_stats']['avg']}")
    print(f"  Median: {summary['token_stats']['median']}")
    print(f"  P90:    {summary['token_stats']['p90']}")
    print(f"  P95:    {summary['token_stats']['p95']}")
    print(f"  P99:    {summary['token_stats']['p99']}")
    print(f"  Max:    {summary['token_stats']['max']}")

    print("\nLargest chunk:")
    print(f"  Source file: {largest['source_file']}")
    print(f"  JSON file:   {largest['json_file']}")
    print(f"  Chunk ID:    {largest['chunk_id']}")
    print(f"  Page:        {largest['page']}")
    print(f"  Method:      {largest['chunking_method']}")
    print(f"  Format:      {largest['chunk_format']}")
    print(f"  Chars:       {largest['char_count']}")
    print(f"  Tokens:      {largest['estimated_tokens']}")

    print("\nContext window fit:")
    for window, values in context_report.items():
        print(
            f"  {window:>6} tokens: "
            f"{values['fit_count']} fit, "
            f"{values['exceed_count']} exceed "
            f"({values['exceed_percent']}%)"
        )

    if recommended_context:
        print(f"\n✅ Minimum listed context that fits the largest chunk: {recommended_context}")
    else:
        print("\n⚠️ Largest chunk exceeds 65536 tokens.")

    print(f"\n✅ Details saved to: {csv_path}")
    print(f"✅ Summary saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate JSON chunk sizes in chars/tokens for embedding context planning."
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["../../data/json"],
        help="One or more folders containing JSON chunks.",
    )

    parser.add_argument(
        "--output",
        default="../../data/chunk_size_stats",
        help="Output folder for CSV/JSON reports.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Embedding model tokenizer to use.",
    )

    parser.add_argument(
        "--embedding-context",
        type=int,
        default=8192,
        help="Embedding model context window to test against.",
    )

    args = parser.parse_args()

    analyze_json_dirs(
        input_dirs=args.inputs,
        output_dir=args.output,
        model_name=args.model,
        embedding_context=args.embedding_context,
    )


if __name__ == "__main__":
    main()
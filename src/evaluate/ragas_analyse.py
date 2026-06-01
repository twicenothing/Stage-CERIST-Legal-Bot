import json
import csv
import os
import sys


THRESHOLDS = {
    "faithfulness": 0.70,
    "answer_relevancy": 0.60,
    "context_precision": 0.70,
    "context_recall": 0.80,
}


def safe_float(value):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def load_detailed_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Your current format:
    # {
    #   "config": {...},
    #   "rows": [...]
    # }
    if isinstance(data, dict) and "rows" in data and isinstance(data["rows"], list):
        return data["rows"], data.get("config", {})

    # Fallback: already a list of rows
    if isinstance(data, list):
        return data, {}

    raise ValueError("Unsupported detailed JSON format. Expected {'config': ..., 'rows': [...]} or a list.")


def load_csv_scores(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def get_weak_reasons(row):
    weak_reasons = []

    for metric, threshold in THRESHOLDS.items():
        score = safe_float(row.get(metric))

        if score is not None and score < threshold:
            weak_reasons.append({
                "metric": metric,
                "score": score,
                "threshold": threshold,
            })

    return weak_reasons


def summarize_retrieved_docs(row):
    docs = row.get("retrieved_docs_debug", []) or []

    summary = []
    for doc in docs:
        summary.append({
            "id": doc.get("id", ""),
            "chunking_method": doc.get("chunking_method", ""),
            "chunk_format": doc.get("chunk_format", ""),
            "source_file": doc.get("source_file", ""),
            "page": doc.get("page", ""),
            "parent_title": doc.get("parent_title", ""),
            "table_id": doc.get("table_id", ""),
            "table_kind": doc.get("table_kind", ""),
            "row_index": doc.get("row_index", ""),
            "distance": doc.get("distance", ""),
            "rerank_score": doc.get("rerank_score", ""),
        })

    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("python ragas_analyse.py ragas_results_detailed.json")
        print("python ragas_analyse.py ragas_results_detailed.json ragas_results.csv")
        return

    detailed_json_path = sys.argv[1]

    if len(sys.argv) >= 3:
        csv_path = sys.argv[2]
    else:
        csv_path = os.path.join(os.path.dirname(detailed_json_path), "ragas_results.csv")

    output_path = os.path.join(
        os.path.dirname(detailed_json_path),
        "ragas_weak_results.json"
    )

    rows, config = load_detailed_json(detailed_json_path)

    if os.path.exists(csv_path):
        score_rows = load_csv_scores(csv_path)

        if len(score_rows) != len(rows):
            print(f"⚠️ Warning: CSV rows ({len(score_rows)}) and detailed rows ({len(rows)}) differ.")

        merged_rows = []

        for i, row in enumerate(rows):
            merged = dict(row)

            if i < len(score_rows):
                for metric in THRESHOLDS:
                    merged[metric] = safe_float(score_rows[i].get(metric))

            merged_rows.append(merged)
    else:
        print(f"⚠️ CSV not found: {csv_path}")
        print("Filtering only works if metric scores are already present in the JSON.")
        merged_rows = rows

    weak_results = []

    for i, row in enumerate(merged_rows):
        weak_reasons = get_weak_reasons(row)

        if weak_reasons:
            weak_results.append({
                "index": i,
                "question": row.get("user_input", ""),
                "optimized_query": row.get("optimized_query", ""),
                "response": row.get("response", ""),
                "reference": row.get("reference", ""),
                "scores": {
                    "faithfulness": row.get("faithfulness"),
                    "answer_relevancy": row.get("answer_relevancy"),
                    "context_precision": row.get("context_precision"),
                    "context_recall": row.get("context_recall"),
                },
                "weak_reasons": weak_reasons,
                "retrieved_contexts": row.get("retrieved_contexts", []),
                "retrieved_docs_debug": summarize_retrieved_docs(row),
            })

    output = {
        "config": config,
        "thresholds": THRESHOLDS,
        "total_rows": len(merged_rows),
        "weak_count": len(weak_results),
        "weak_results": weak_results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✅ Weak results saved to: {output_path}")
    print(f"📉 Found {len(weak_results)} weak results out of {len(merged_rows)} total.")


if __name__ == "__main__":
    main()
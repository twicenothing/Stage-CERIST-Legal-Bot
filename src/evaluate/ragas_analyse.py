import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def load_results(csv_path: Path, detailed_path: Path):
    df = pd.read_csv(csv_path)

    with open(detailed_path, "r", encoding="utf-8") as f:
        detailed = json.load(f)

    return df, detailed


def print_metric_summary(df):
    print("\n" + "=" * 100)
    print("📊 METRIC SUMMARY")
    print("=" * 100)

    for col in METRIC_COLS:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")

        print(f"\n{col}")
        print(f"  count:  {s.count()}")
        print(f"  mean:   {s.mean():.4f}")
        print(f"  median: {s.median():.4f}")
        print(f"  min:    {s.min():.4f}")
        print(f"  p25:    {s.quantile(0.25):.4f}")
        print(f"  p75:    {s.quantile(0.75):.4f}")
        print(f"  max:    {s.max():.4f}")


def print_weak_rows(df, top=15):
    print("\n" + "=" * 100)
    print("🚨 WEAKEST ROWS BY METRIC")
    print("=" * 100)

    base_cols = [
        "user_input",
        "optimized_query",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "context_chars",
        "estimated_context_tokens",
        "response",
        "reference",
    ]

    cols = [c for c in base_cols if c in df.columns]

    for metric in METRIC_COLS:
        if metric not in df.columns:
            continue

        print("\n" + "-" * 100)
        print(f"Lowest {metric}")
        print("-" * 100)

        tmp = df.copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")

        print(
            tmp.sort_values(metric, ascending=True)[cols]
               .head(top)
               .to_string(index=False, max_colwidth=180)
        )


def analyze_context_size(df):
    if "context_chars" not in df.columns:
        return

    print("\n" + "=" * 100)
    print("📏 CONTEXT SIZE ANALYSIS")
    print("=" * 100)

    chars = pd.to_numeric(df["context_chars"], errors="coerce")

    print(f"Average context chars: {chars.mean():.2f}")
    print(f"Median context chars:  {chars.median():.2f}")
    print(f"P90 context chars:     {chars.quantile(0.90):.2f}")
    print(f"P95 context chars:     {chars.quantile(0.95):.2f}")
    print(f"Max context chars:     {chars.max():.2f}")

    for threshold in [10000, 20000, 30000, 50000, 100000]:
        count = (chars > threshold).sum()
        print(f"Rows > {threshold:>6} chars: {count}")

    print("\nMetric averages by context size:")

    buckets = [
        ("<=10k", df[chars <= 10000]),
        ("10k-30k", df[(chars > 10000) & (chars <= 30000)]),
        ("30k-50k", df[(chars > 30000) & (chars <= 50000)]),
        (">50k", df[chars > 50000]),
    ]

    for name, part in buckets:
        if len(part) == 0:
            continue

        print(f"\n{name} rows={len(part)}")
        for metric in METRIC_COLS:
            if metric in part.columns:
                s = pd.to_numeric(part[metric], errors="coerce")
                print(f"  {metric}: {s.mean():.4f}")


def parse_docs_debug(value):
    if not isinstance(value, str) or not value.strip():
        return []

    try:
        return json.loads(value)
    except Exception:
        return []


def analyze_retrieved_docs(df, detailed):
    print("\n" + "=" * 100)
    print("📚 RETRIEVED DOCS ANALYSIS")
    print("=" * 100)

    method_counter = Counter()
    source_counter = Counter()
    huge_docs = []

    if "retrieved_docs_debug" in df.columns:
        iterator = []
        for row_idx, row in df.iterrows():
            docs = parse_docs_debug(row.get("retrieved_docs_debug", ""))
            iterator.append((row_idx, row.get("user_input", ""), docs))
    else:
        iterator = []
        for row_idx, row in enumerate(detailed.get("rows", [])):
            iterator.append((row_idx, row.get("user_input", ""), row.get("retrieved_docs_debug", [])))

    for row_idx, question, docs in iterator:
        for doc in docs:
            method = doc.get("chunking_method", "")
            source = doc.get("source_file", "")
            text_chars = doc.get("text_chars", 0) or 0

            method_counter[method] += 1
            source_counter[source] += 1

            if text_chars > 20000:
                huge_docs.append({
                    "row_index": row_idx,
                    "question": question,
                    "source_file": source,
                    "page": doc.get("page", ""),
                    "chunking_method": method,
                    "chunk_format": doc.get("chunk_format", ""),
                    "text_chars": text_chars,
                    "rerank_score": doc.get("rerank_score", ""),
                    "distance": doc.get("distance", ""),
                    "id": doc.get("id", ""),
                })

    print("\nRetrieved chunk methods:")
    for k, v in method_counter.most_common():
        print(f"  {k}: {v}")

    print("\nTop retrieved source files:")
    for k, v in source_counter.most_common(20):
        print(f"  {k}: {v}")

    huge_docs.sort(key=lambda x: x["text_chars"], reverse=True)

    print("\nHuge retrieved docs > 20k chars:")
    print(f"  count: {len(huge_docs)}")

    for item in huge_docs[:20]:
        print("\n" + "-" * 100)
        print(f"row_index:       {item['row_index']}")
        print(f"text_chars:      {item['text_chars']}")
        print(f"method:          {item['chunking_method']}")
        print(f"format:          {item['chunk_format']}")
        print(f"source:          {item['source_file']}")
        print(f"page:            {item['page']}")
        print(f"score:           {item['rerank_score']}")
        print(f"distance:        {item['distance']}")
        print(f"id:              {item['id']}")
        print(f"question:        {item['question']}")

    return huge_docs


def detect_refusals(df):
    print("\n" + "=" * 100)
    print("🚫 REFUSAL ANALYSIS")
    print("=" * 100)

    refusal_phrase = "Je suis désolé, je n'ai pas la réponse"
    if "response" not in df.columns:
        return

    refusals = df[df["response"].astype(str).str.contains(refusal_phrase, case=False, na=False)]

    print(f"Refusal rows: {len(refusals)} / {len(df)}")

    if len(refusals):
        cols = [
            "user_input",
            "optimized_query",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_chars",
            "response",
            "reference",
        ]
        cols = [c for c in cols if c in df.columns]

        print(
            refusals[cols]
            .head(20)
            .to_string(index=False, max_colwidth=180)
        )


def write_action_files(df, huge_docs, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    weak_path = output_dir / "weak_rows.csv"
    huge_path = output_dir / "huge_retrieved_docs.csv"
    refusals_path = output_dir / "refusal_rows.csv"

    weak = df.copy()

    for metric in METRIC_COLS:
        if metric in weak.columns:
            weak[metric] = pd.to_numeric(weak[metric], errors="coerce")

    conditions = []
    if "faithfulness" in weak.columns:
        conditions.append(weak["faithfulness"] < 0.6)
    if "answer_relevancy" in weak.columns:
        conditions.append(weak["answer_relevancy"] < 0.5)
    if "context_precision" in weak.columns:
        conditions.append(weak["context_precision"] < 0.5)
    if "context_recall" in weak.columns:
        conditions.append(weak["context_recall"] < 0.5)

    if conditions:
        mask = conditions[0]
        for cond in conditions[1:]:
            mask = mask | cond

        weak_rows = weak[mask].copy()
        weak_rows.to_csv(weak_path, index=False)
        print(f"\n✅ Weak rows saved: {weak_path}")

    if huge_docs:
        pd.DataFrame(huge_docs).to_csv(huge_path, index=False)
        print(f"✅ Huge retrieved docs saved: {huge_path}")

    if "response" in df.columns:
        refusal_phrase = "Je suis désolé, je n'ai pas la réponse"
        refusals = df[df["response"].astype(str).str.contains(refusal_phrase, case=False, na=False)]
        refusals.to_csv(refusals_path, index=False)
        print(f"✅ Refusal rows saved: {refusals_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="ragas_results.csv",
        help="Path to ragas_results.csv",
    )
    parser.add_argument(
        "--json",
        default="ragas_results_detailed.json",
        help="Path to ragas_results_detailed.json",
    )
    parser.add_argument(
        "--output-dir",
        default="ragas_analysis",
        help="Folder where analysis files will be saved",
    )
    parser.add_argument("--top", type=int, default=15)

    args = parser.parse_args()

    csv_path = Path(args.csv)
    detailed_path = Path(args.json)
    output_dir = Path(args.output_dir)

    df, detailed = load_results(csv_path, detailed_path)

    print_metric_summary(df)
    analyze_context_size(df)
    huge_docs = analyze_retrieved_docs(df, detailed)
    detect_refusals(df)
    print_weak_rows(df, top=args.top)
    write_action_files(df, huge_docs, output_dir)


if __name__ == "__main__":
    main()
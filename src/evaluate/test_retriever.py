import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from dotenv import load_dotenv


# -------------------------------------------------------------------
# Project paths
# -------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

load_dotenv(dotenv_path=Path(project_root) / ".env")


from generate.query_parse import rewrite_query
from generate.llm_generate import init_rag_pipeline
from retrieve.retrieve import get_retrieved_documents
from rerank.rerank import rerank_documents


TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "5"))


def normalize_source_name(value: str) -> str:
    """
    Converts:
      F202009.txt -> f202009
      F202009.pdf -> f202009
    """
    if not value:
        return ""

    name = os.path.basename(str(value).strip())
    stem = os.path.splitext(name)[0]
    return stem.lower()


def load_golden_dataset(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both:
    # [ {...}, {...} ]
    # {"data": [ ... ]}
    # {"questions": [ ... ]}
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ["data", "questions", "items", "rows"]:
            if key in data and isinstance(data[key], list):
                return data[key]

    raise ValueError("Unsupported golden dataset format. Expected a list of objects.")


def doc_debug(doc, rank: int, expected_stem: str):
    meta = doc.get("meta", {}) or {}
    text = doc.get("text", "") or ""

    source_file = meta.get("source_file", "")
    source_stem = normalize_source_name(source_file)

    return {
        "rank": rank,
        "hit_expected_source": source_stem == expected_stem,
        "doc_id": doc.get("id", ""),
        "source_file": source_file,
        "source_stem": source_stem,
        "page": meta.get("page", ""),
        "chunking_method": meta.get("chunking_method", ""),
        "chunk_format": meta.get("chunk_format", ""),
        "table_id": meta.get("table_id", ""),
        "table_kind": meta.get("table_kind", ""),
        "row_index": meta.get("row_index", ""),
        "distance": doc.get("distance", None),
        "rerank_score": doc.get("rerank_score", None),
        "adjusted_rerank_score": doc.get("adjusted_rerank_score", None),
        "rerank_penalty_reason": doc.get("rerank_penalty_reason", ""),
        "text_chars": len(text),
        "text_preview": " ".join(text.split())[:500],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test retriever + reranker against golden dataset expected source files."
    )

    parser.add_argument(
        "--dataset",
        default=os.path.join(project_root, "data", "golden_dataset", "golden_dataset.json"),
        help="Path to golden dataset JSON.",
    )

    parser.add_argument(
        "--output-dir",
        default=os.path.join(current_dir, "retriever_test_results"),
        help="Output directory.",
    )

    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=TOP_K_RETRIEVE,
        help="Number of chunks to retrieve before reranking.",
    )

    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=TOP_K_RERANK,
        help="Number of final reranked docs.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Use original question for retrieval instead of rewritten query.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_golden_dataset(dataset_path)

    if args.limit:
        rows = rows[:args.limit]

    print("=" * 100)
    print("🔎 GOLDEN DATASET RETRIEVER TEST")
    print("=" * 100)
    print(f"Dataset:        {dataset_path}")
    print(f"Questions:      {len(rows)}")
    print(f"top_k_retrieve: {args.top_k_retrieve}")
    print(f"top_k_rerank:   {args.top_k_rerank}")
    print(f"Rewrite:        {not args.no_rewrite}")
    print("=" * 100)

    print("\n🚀 Initializing RAG pipeline...")
    collection, bi_encoder, reranker = init_rag_pipeline()

    summary_rows = []
    detailed_rows = []

    for idx, item in enumerate(rows, start=1):
        question = item.get("question", "")
        expected_source = item.get("source", "")
        expected_stem = normalize_source_name(expected_source)
        reference_answer = item.get("reponse") or item.get("réponse") or item.get("answer") or item.get("ground_truth", "")

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(rows)}] {question}")
        print(f"Expected source: {expected_source} -> {expected_stem}")

        if not question:
            print("⚠️ Empty question, skipping.")
            continue

        if args.no_rewrite:
            retrieval_query = question
        else:
            retrieval_query = rewrite_query(question)

        print(f"Retrieval query: {retrieval_query}")

        if not retrieval_query or retrieval_query.strip().upper() == "SKIP_OPTIMIZATION":
            summary_rows.append({
                "index": idx,
                "question": question,
                "expected_source": expected_source,
                "expected_stem": expected_stem,
                "optimized_query": retrieval_query,
                "strategy_used": "SKIP_OPTIMIZATION",
                "hit_at_1": False,
                "hit_at_k": False,
                "hit_rank": "",
                "final_docs_count": 0,
                "recursive_in_final": False,
                "recursive_count_final": 0,
                "final_sources": "",
                "final_methods": "",
                "reference_answer": reference_answer,
            })
            continue

        initial_docs, strategy_used = get_retrieved_documents(
            retrieval_query,
            bi_encoder,
            collection,
            top_k=args.top_k_retrieve,
        )

        final_docs = rerank_documents(
            question,  # original question for reranking
            initial_docs,
            reranker,
            top_k=args.top_k_rerank,
        )

        final_debug = [
            doc_debug(doc, rank, expected_stem)
            for rank, doc in enumerate(final_docs, start=1)
        ]

        hit_ranks = [d["rank"] for d in final_debug if d["hit_expected_source"]]
        hit_at_k = len(hit_ranks) > 0
        hit_at_1 = bool(final_debug and final_debug[0]["hit_expected_source"])
        hit_rank = hit_ranks[0] if hit_ranks else ""

        final_sources = [d["source_file"] for d in final_debug]
        final_methods = [d["chunking_method"] for d in final_debug]

        recursive_count = sum(1 for d in final_debug if d["chunking_method"] == "recursive")
        recursive_in_final = recursive_count > 0

        print(f"Strategy used:      {strategy_used}")
        print(f"Hit@1:              {hit_at_1}")
        print(f"Hit@{args.top_k_rerank}:              {hit_at_k}")
        print(f"Hit rank:           {hit_rank}")
        print(f"Recursive in final: {recursive_in_final} ({recursive_count})")

        for d in final_debug:
            print(
                f"  #{d['rank']} "
                f"| hit={d['hit_expected_source']} "
                f"| source={d['source_file']} "
                f"| page={d['page']} "
                f"| method={d['chunking_method']} "
                f"| raw={d['rerank_score']} "
                f"| adjusted={d['adjusted_rerank_score']} "
                f"| distance={d['distance']} "
                f"| chars={d['text_chars']}"
            )

        summary_rows.append({
            "index": idx,
            "question": question,
            "expected_source": expected_source,
            "expected_stem": expected_stem,
            "optimized_query": retrieval_query,
            "strategy_used": strategy_used,
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "hit_rank": hit_rank,
            "final_docs_count": len(final_docs),
            "recursive_in_final": recursive_in_final,
            "recursive_count_final": recursive_count,
            "final_sources": json.dumps(final_sources, ensure_ascii=False),
            "final_methods": json.dumps(final_methods, ensure_ascii=False),
            "reference_answer": reference_answer,
        })

        detailed_rows.append({
            "index": idx,
            "question": question,
            "expected_source": expected_source,
            "expected_stem": expected_stem,
            "reference_answer": reference_answer,
            "optimized_query": retrieval_query,
            "strategy_used": strategy_used,
            "hit_at_1": hit_at_1,
            "hit_at_k": hit_at_k,
            "hit_rank": hit_rank,
            "recursive_in_final": recursive_in_final,
            "recursive_count_final": recursive_count,
            "final_docs": final_debug,
        })

    summary_df = pd.DataFrame(summary_rows)

    csv_path = output_dir / "retriever_test_summary.csv"
    json_path = output_dir / "retriever_test_detailed.json"

    summary_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "dataset": str(dataset_path),
                    "top_k_retrieve": args.top_k_retrieve,
                    "top_k_rerank": args.top_k_rerank,
                    "rewrite_enabled": not args.no_rewrite,
                },
                "summary": {
                    "total": len(summary_rows),
                    "hit_at_1": int(summary_df["hit_at_1"].sum()) if len(summary_df) else 0,
                    "hit_at_k": int(summary_df["hit_at_k"].sum()) if len(summary_df) else 0,
                    "recursive_in_final": int(summary_df["recursive_in_final"].sum()) if len(summary_df) else 0,
                },
                "rows": detailed_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 100)
    print("📊 RETRIEVER TEST SUMMARY")
    print("=" * 100)

    total = len(summary_df)
    if total:
        hit1 = summary_df["hit_at_1"].sum()
        hitk = summary_df["hit_at_k"].sum()
        recursive_used = summary_df["recursive_in_final"].sum()

        print(f"Total questions:       {total}")
        print(f"Hit@1:                 {hit1}/{total} = {hit1 / total:.3f}")
        print(f"Hit@{args.top_k_rerank}:                 {hitk}/{total} = {hitk / total:.3f}")
        print(f"Recursive in final:    {recursive_used}/{total} = {recursive_used / total:.3f}")

        method_counter = Counter()
        for methods_json in summary_df["final_methods"]:
            try:
                for m in json.loads(methods_json):
                    method_counter[m] += 1
            except Exception:
                pass

        print("\nFinal chunk methods:")
        for method, count in method_counter.most_common():
            print(f"  {method}: {count}")

    print(f"\n✅ CSV saved:  {csv_path}")
    print(f"✅ JSON saved: {json_path}")


if __name__ == "__main__":
    main()
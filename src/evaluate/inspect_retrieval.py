import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if src_dir not in sys.path:
    sys.path.append(src_dir)

load_dotenv(dotenv_path=Path(project_root) / ".env")

from generate.query_parse import rewrite_query
from generate.llm_generate import init_rag_pipeline
from rerank.rerank import get_best_documents_for_llm


TOP_K_RETRIEVE = int(os.getenv("RAG_TOP_K_RETRIEVE", "30"))
TOP_K_RERANK = int(os.getenv("RAG_TOP_K_RERANK", "5"))


def looks_tabular(text: str) -> bool:
    """
    Heuristic only.
    Helps us detect if a chunk looks like table/annex data even when metadata says regex.
    """
    if not text:
        return False

    low = text.lower()

    table_signals = [
        "tableau annexe",
        "tableau",
        "crédits ouverts",
        "credits ouverts",
        "en da",
        "nos des chapitres",
        "libelles",
        "libellés",
        "répartition par chapitre",
        "repartition par chapitre",
        "total de la",
        "total du titre",
        "|---",
        "| ",
    ]

    return any(s in low for s in table_signals)


def compact_preview(text: str, max_chars: int = 700) -> str:
    text = str(text or "")
    text = " ".join(text.split())
    return text[:max_chars]


def doc_to_debug_row(question_id, question, optimized_query, rank, doc):
    meta = doc.get("meta", {}) or {}
    text = doc.get("text", "") or ""

    return {
        "question_id": question_id,
        "question": question,
        "optimized_query": optimized_query,

        "rank": rank,
        "doc_id": doc.get("id", ""),
        "rerank_score": doc.get("rerank_score", None),
        "distance": doc.get("distance", None),

        "chunking_method": meta.get("chunking_method", ""),
        "chunk_format": meta.get("chunk_format", ""),
        "source_file": meta.get("source_file", ""),
        "page": meta.get("page", ""),
        "parent_title": meta.get("parent_title", ""),
        "document_type": meta.get("document_type", ""),
        "table_id": meta.get("table_id", ""),
        "table_kind": meta.get("table_kind", ""),
        "row_index": meta.get("row_index", ""),

        "text_chars": len(text),
        "looks_tabular": looks_tabular(text),
        "text_preview": compact_preview(text),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect retrieval/reranking results for questions that produced huge retrieved docs."
    )

    parser.add_argument(
        "--huge-csv",
        default="ragas_analysis/huge_retrieved_docs.csv",
        help="Path to huge_retrieved_docs.csv produced by analyse_ragas_results.py",
    )

    parser.add_argument(
        "--output-dir",
        default="retrieval_inspection",
        help="Folder where inspection files will be saved.",
    )

    parser.add_argument(
        "--top-k-retrieve",
        type=int,
        default=TOP_K_RETRIEVE,
        help="Number of candidates retrieved before reranking.",
    )

    parser.add_argument(
        "--top-k-rerank",
        type=int,
        default=TOP_K_RERANK,
        help="Number of final reranked docs to inspect.",
    )

    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewriting and retrieve with original questions.",
    )

    args = parser.parse_args()

    huge_csv = Path(args.huge_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not huge_csv.exists():
        raise FileNotFoundError(f"Could not find: {huge_csv}")

    df = pd.read_csv(huge_csv)

    if "question" not in df.columns:
        raise ValueError(
            "Expected a 'question' column in huge_retrieved_docs.csv. "
            f"Columns found: {list(df.columns)}"
        )

    questions = (
        df["question"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    print("=" * 100)
    print("🔎 RETRIEVAL INSPECTION FROM HUGE DOC QUESTIONS")
    print("=" * 100)
    print(f"Huge CSV:        {huge_csv}")
    print(f"Unique questions:{len(questions)}")
    print(f"top_k_retrieve:  {args.top_k_retrieve}")
    print(f"top_k_rerank:    {args.top_k_rerank}")
    print(f"Rewrite enabled: {not args.no_rewrite}")
    print("=" * 100)

    print("\n🚀 Initializing RAG pipeline...")
    collection, bi_encoder, reranker = init_rag_pipeline()

    all_rows = []
    detailed = []

    for qid, question in enumerate(questions, start=1):
        print("\n" + "=" * 100)
        print(f"[{qid}/{len(questions)}] QUESTION")
        print(question)

        if args.no_rewrite:
            optimized_query = question
        else:
            optimized_query = rewrite_query(question)

        print("\nOptimized query:")
        print(optimized_query)

        if not optimized_query or optimized_query.strip().upper() == "SKIP_OPTIMIZATION":
            print("⛔ SKIP_OPTIMIZATION. Skipping retrieval.")
            detailed.append({
                "question_id": qid,
                "question": question,
                "optimized_query": optimized_query,
                "docs": [],
            })
            continue

        best_docs = get_best_documents_for_llm(
            optimized_query,
            collection,
            bi_encoder,
            reranker,
            top_k_retrieve=args.top_k_retrieve,
            top_k_rerank=args.top_k_rerank,
            rerank_query=question,
        )

        question_detail = {
            "question_id": qid,
            "question": question,
            "optimized_query": optimized_query,
            "docs": [],
        }

        print("\nTOP RERANKED DOCS RETURNED TO MODEL:")
        for rank, doc in enumerate(best_docs, start=1):
            row = doc_to_debug_row(
                question_id=qid,
                question=question,
                optimized_query=optimized_query,
                rank=rank,
                doc=doc,
            )

            all_rows.append(row)

            question_detail["docs"].append({
                **row,
                "full_text": doc.get("text", "") or "",
            })

            print("-" * 100)
            print(f"Rank:            {rank}")
            print(f"Score:           {row['rerank_score']}")
            print(f"Distance:        {row['distance']}")
            print(f"Method:          {row['chunking_method']}")
            print(f"Format:          {row['chunk_format']}")
            print(f"Source:          {row['source_file']}")
            print(f"Page:            {row['page']}")
            print(f"Table ID:        {row['table_id']}")
            print(f"Table kind:      {row['table_kind']}")
            print(f"Text chars:      {row['text_chars']}")
            print(f"Looks tabular:   {row['looks_tabular']}")
            print(f"Preview:         {row['text_preview']}")

        detailed.append(question_detail)

    csv_path = output_dir / "top_reranked_docs.csv"
    json_path = output_dir / "top_reranked_docs_detailed.json"

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(csv_path, index=False)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "huge_csv": str(huge_csv),
                    "top_k_retrieve": args.top_k_retrieve,
                    "top_k_rerank": args.top_k_rerank,
                    "rewrite_enabled": not args.no_rewrite,
                },
                "questions": detailed,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 100)
    print("✅ INSPECTION SAVED")
    print("=" * 100)
    print(f"CSV summary:   {csv_path}")
    print(f"JSON detailed: {json_path}")

    if len(out_df):
        print("\nChunking method counts:")
        print(out_df["chunking_method"].value_counts().to_string())

        print("\nLooks tabular counts:")
        print(out_df["looks_tabular"].value_counts().to_string())

        print("\nHuge regex returned to model:")
        huge_regex = out_df[
            (out_df["chunking_method"] == "regex")
            & (out_df["text_chars"] > 20000)
        ]
        print(f"{len(huge_regex)} / {len(out_df)} final docs")


if __name__ == "__main__":
    main()
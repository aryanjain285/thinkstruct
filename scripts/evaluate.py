"""Build a test collection and compare every retrieval system on it.

    python scripts/evaluate.py --generate            # build data/evaluation/eval_set.jsonl
    python scripts/evaluate.py --systems bm25 dense hybrid
    python scripts/evaluate.py --systems hybrid hybrid_reranked --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import DATA_DIR, EMBEDDER, INDEX_NAME, OPENSEARCH_HOST, RAW_DIR, REPORTS_DIR
from patsearch.embeddings.service import create_embedder
from patsearch.evaluation.evaluator import (
    EvalSet,
    build_eval_set,
    comparison_table,
    evaluate_system,
)
from patsearch.evaluation.query_gen import (
    LLMQueryParaphraser,
    overlap_report,
    paraphrase_eval_set,
)
from patsearch.evaluation.significance import compare_systems, significance_table
from patsearch.ingestion.loader import load_all
from patsearch.pipeline import search
from patsearch.processing.reconstruct import reconstruct_claims
from patsearch.processing.records import build_records
from patsearch.reranking.service import create_reranker
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.query import Filters

EVAL_PATH = DATA_DIR / "evaluation" / "eval_set.jsonl"
REPORT_METRICS = (
    "success@10", "success@10_strict", "recall@10", "recall@50",
    "ndcg@10", "mrr@10", "precision@5",
)


def generate(path: Path, n_queries: int) -> EvalSet:
    patents, _ = load_all(RAW_DIR)
    by_patent = {}
    for p in patents:
        by_patent[p.patent_id] = build_records(p, reconstruct_claims(p.patent_id, p.claims_raw))
    es = build_eval_set(patents, by_patent, n_queries=n_queries)
    es.save(path)
    print(f"wrote {path}")
    print(f"  queries        : {len(es.queries)}")
    print(f"  source patents : {es.notes['source_patents']}")
    types = {}
    for q in es.queries:
        types[q.query_type] = types.get(q.query_type, 0) + 1
    print(f"  by type        : {types}")
    print(f"  mean relevant  : {sum(len(q.qrels) for q in es.queries)/len(es.queries):.1f}")
    return es


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--n-queries", type=int, default=40)
    ap.add_argument("--systems", nargs="+",
                    default=["bm25", "dense", "hybrid", "hybrid_reranked"])
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--reranker", default="llm-mini")
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--limit", type=int, help="evaluate only the first N queries")
    ap.add_argument("--eval-path", type=Path, default=EVAL_PATH)
    ap.add_argument(
        "--paraphrase", action="store_true",
        help="rewrite queries with an LLM to remove verbatim overlap with their targets",
    )
    ap.add_argument("--paraphrase-model", default="gpt-5.4-mini")
    ap.add_argument("--baseline", help="system to test others against (default: first)")
    ap.add_argument("--sig-test", choices=["bootstrap", "t"], default="bootstrap")
    args = ap.parse_args()

    if args.generate or not args.eval_path.exists():
        generate(args.eval_path, args.n_queries)
        if args.generate and not args.paraphrase:
            return 0

    if args.paraphrase:
        base = EvalSet.load(args.eval_path)
        print(f"paraphrasing {len(base.queries)} queries with {args.paraphrase_model} ...")
        para = paraphrase_eval_set(base, LLMQueryParaphraser(args.paraphrase_model))
        args.eval_path = args.eval_path.with_name(
            args.eval_path.stem + "_paraphrased" + args.eval_path.suffix
        )
        para.save(args.eval_path)
        print(f"wrote {args.eval_path}  ({len(para.queries)} queries, "
              f"{para.notes['dropped']} dropped)")

    es = EvalSet.load(args.eval_path)
    print(f"\nloaded {len(es.queries)} queries from {args.eval_path}")
    print(f"NOTE: {es.notes.get('measures')}")
    print(f"      does NOT measure: {es.notes.get('does_not_measure')}")

    client = get_client(args.host)
    wait_for_health(client)

    # Lexical overlap is reported up front: a high value means the benchmark rewards
    # exact term matching and any lexical-vs-dense comparison must be read with that
    # in mind.
    def _doc(rid: str) -> str:
        try:
            return client.get(index=args.index, id=rid, _source=["text"])["_source"]["text"]
        except Exception:
            return ""

    ov = overlap_report(es, _doc)
    if ov:
        print(f"      query/target token overlap: mean {ov['mean_overlap']:.1%}, "
              f"median {ov['median_overlap']:.1%}")
    print()

    needs_vec = any(s != "bm25" for s in args.systems)
    embedder = create_embedder(args.embedder) if needs_vec else None
    reranker = create_reranker(args.reranker) if "hybrid_reranked" in args.systems else None

    results = []
    for system in args.systems:
        print(f"evaluating {system} ...", flush=True)

        def run_query(text: str, _s=system):
            out = search(
                client, args.index, text, method=_s, filters=Filters(),
                embedder=embedder, reranker=reranker,
                candidates=args.candidates, top_k=args.candidates,
            )
            return [h.record_id for h in out.hits], out.timings_ms

        results.append(evaluate_system(system, es, run_query, limit=args.limit))

    print("\n" + comparison_table(results, REPORT_METRICS))

    # A raw score difference on 80 queries is meaningless without a paired test.
    sig: dict[str, list] = {}
    if len(results) > 1:
        print()
        for metric in ("ndcg@10", "recall@50"):
            tests = compare_systems(
                results, metric, baseline=args.baseline or results[0].name,
                test=args.sig_test,
            )
            sig[metric] = [asdict(t) for t in tests]
            print(significance_table(tests))
            print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "evaluation_results.json"
    out.write_text(
        json.dumps(
            {
                "notes": es.notes,
                "lexical_overlap": ov,
                "n_queries": len(es.queries[: args.limit] if args.limit else es.queries),
                "candidates": args.candidates,
                "systems": {
                    r.name: {"metrics": r.metrics, "latency_ms": r.latency_ms} for r in results
                },
                "significance": sig,
                "per_query": {r.name: r.per_query for r in results},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

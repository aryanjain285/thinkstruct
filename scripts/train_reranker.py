"""Train a learning-to-rank reranker on the pooled relevance judgements.

    python scripts/train_reranker.py                 # train + evaluate
    python scripts/train_reranker.py --no-eval       # train only

Splits by patent, so no patent contributes rows to both train and test.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import DATA_DIR, EMBEDDER, INDEX_NAME, OPENSEARCH_HOST, REPORTS_DIR, ROOT
from patsearch.embeddings.service import create_embedder
from patsearch.evaluation.evaluator import EvalSet, comparison_table, evaluate_system
from patsearch.evaluation.significance import compare_systems, significance_table
from patsearch.pipeline import search
from patsearch.reranking.service import rerank
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.query import Filters
from patsearch.training.features import build_rows
from patsearch.training.ltr import LTRModel, LTRReranker, train, write_report

MODEL_PATH = ROOT / "models" / "ltr_reranker.pkl"
REPORT_METRICS = ("success@10", "recall@10", "recall@50", "ndcg@10", "mrr@10", "precision@5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", type=Path,
                    default=DATA_DIR / "evaluation" / "eval_set_pooled.jsonl")
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--test-fraction", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    if not args.eval_path.exists():
        print(f"missing {args.eval_path}. Run scripts/build_qrels.py first.", file=sys.stderr)
        return 1

    es = EvalSet.load(args.eval_path)
    print(f"loaded {len(es.queries)} queries with pooled judgements")

    client = get_client(args.host)
    wait_for_health(client)
    embedder = create_embedder(args.embedder)

    # ---------------------------------------------------------- training data
    print(f"\ngenerating training data (hybrid top-{args.candidates} per query) ...")
    rows = []
    hits_cache: dict[str, list] = {}
    for i, q in enumerate(es.queries, 1):
        out = search(client, args.index, q.text, method="hybrid", filters=Filters(),
                     embedder=embedder, candidates=args.candidates,
                     top_k=args.candidates)
        hits_cache[q.query_id] = out.hits
        rows.extend(build_rows(q.query_id, q.text, out.hits, q.qrels))
        if i % 20 == 0:
            print(f"  {i}/{len(es.queries)} queries, {len(rows)} rows", flush=True)

    print(f"  {len(rows)} training rows from {len(es.queries)} queries")

    # ------------------------------------------------------------------ train
    print("\ntraining ...")
    model, report, test_rows = train(
        rows, test_fraction=args.test_fraction, seed=args.seed
    )
    model.save(args.model_path)
    write_report(report, REPORTS_DIR / "training_report.json")

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nsaved model to {args.model_path}")

    if args.no_eval:
        return 0

    # ------------------------------------------------------ held-out evaluation
    # Only queries whose patents are in the test split, so we never score the model
    # on patents it trained on.
    test_patents = {r.patent_id for r in test_rows}
    held_out = [q for q in es.queries if q.source_patent_id in test_patents]
    if not held_out:
        print("\nno held-out queries; skipping evaluation", file=sys.stderr)
        return 0

    print(f"\nevaluating on {len(held_out)} held-out queries "
          f"({len(test_patents)} unseen patents)")
    held = EvalSet(held_out, es.notes)
    reranker = LTRReranker(LTRModel.load(args.model_path))

    def run_baseline(text: str, _qid=None):
        out = search(client, args.index, text, method="hybrid", filters=Filters(),
                     embedder=embedder, candidates=args.candidates, top_k=args.candidates)
        return [h.record_id for h in out.hits], out.timings_ms

    def run_ltr(text: str):
        out = search(client, args.index, text, method="hybrid", filters=Filters(),
                     embedder=embedder, candidates=args.candidates, top_k=args.candidates)
        import time
        t0 = time.perf_counter()
        ranked = rerank(reranker, text, out.hits)
        timings = dict(out.timings_ms)
        timings["rerank"] = (time.perf_counter() - t0) * 1000
        return [h.record_id for h in ranked], timings

    results = [
        evaluate_system("hybrid", held, run_baseline),
        evaluate_system("hybrid+ltr", held, run_ltr),
    ]
    print("\n" + comparison_table(results, REPORT_METRICS))

    print()
    sig: dict[str, list] = {}
    for metric in ("ndcg@10", "recall@10", "recall@50"):
        tests = compare_systems(results, metric, baseline="hybrid")
        sig[metric] = [asdict(t) for t in tests]
        print(significance_table(tests))
        print()

    out_path = REPORTS_DIR / "training_evaluation.json"
    out_path.write_text(json.dumps({
        "held_out_queries": len(held_out),
        "held_out_patents": len(test_patents),
        "systems": {r.name: {"metrics": r.metrics, "latency_ms": r.latency_ms}
                    for r in results},
        "significance": sig,
    }, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

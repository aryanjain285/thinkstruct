"""Build cross-patent relevance judgements by TREC-style pooling.

    python scripts/build_qrels.py                    # pool + judge, writes a new eval set
    python scripts/build_qrels.py --depth 20         # deeper pool, more judgements
    python scripts/build_qrels.py --dry-run          # show pool size and cost, judge nothing

Judgements are cached in data/evaluation/judgements.jsonl, so re-running is free and
an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import DATA_DIR, EMBEDDER, INDEX_NAME, OPENSEARCH_HOST
from patsearch.embeddings.service import create_embedder
from patsearch.evaluation.evaluator import EvalSet
from patsearch.evaluation.pooling import (
    JudgementCache,
    LLMAssessor,
    apply_to_eval_set,
    build_pool,
    judge_pool,
    pool_stats,
)
from patsearch.pipeline import search
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.query import Filters

EVAL_DIR = DATA_DIR / "evaluation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-path", type=Path,
                    default=EVAL_DIR / "eval_set_paraphrased.jsonl")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "eval_set_pooled.jsonl")
    ap.add_argument("--cache", type=Path, default=EVAL_DIR / "judgements.jsonl")
    ap.add_argument("--systems", nargs="+", default=["bm25", "dense", "hybrid"])
    ap.add_argument("--depth", type=int, default=15, help="top-D per system into the pool")
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--assessor", default="gpt-5.4-mini")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--limit", type=int, help="only pool the first N queries")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    es = EvalSet.load(args.eval_path)
    queries = es.queries[: args.limit] if args.limit else es.queries
    print(f"loaded {len(queries)} queries from {args.eval_path}")

    client = get_client(args.host)
    wait_for_health(client)
    needs_vec = any(s != "bm25" for s in args.systems)
    embedder = create_embedder(args.embedder) if needs_vec else None

    def run_system(system: str, text: str):
        out = search(client, args.index, text, method=system, filters=Filters(),
                     embedder=embedder, candidates=args.depth, top_k=args.depth)
        return [(h.record_id, h.patent_id, h.text) for h in out.hits]

    print(f"pooling top-{args.depth} from {args.systems} ...")
    pool = build_pool(queries, run_system, args.systems, depth=args.depth)
    stats = pool_stats(pool)
    print(json.dumps(stats, indent=2))

    cache = JudgementCache(args.cache)
    need = sum(
        1 for qid, entries in pool.items()
        for rid in entries if cache.get(qid, rid) is None
    )
    calls = -(-need // args.batch_size)
    print(f"\ncached judgements : {len(cache)}")
    print(f"still to judge    : {need}")
    print(f"estimated calls   : {calls}")

    if args.dry_run:
        print("\n--dry-run: nothing judged")
        return 0
    if need == 0:
        print("everything already judged (cache hit)")

    assessor = LLMAssessor(args.assessor, batch_size=args.batch_size)
    print(f"\njudging with {args.assessor} ...")
    judged = judge_pool(queries, pool, assessor, cache)

    pooled = apply_to_eval_set(EvalSet(queries, es.notes), judged)
    pooled.save(args.out)

    # How much did pooling actually add over the structural labels?
    before = sum(len(q.qrels) for q in queries) / max(len(queries), 1)
    after = sum(len(q.qrels) for q in pooled.queries) / max(len(pooled.queries), 1)
    cross = 0
    for q in pooled.queries:
        cross += sum(1 for rid in q.qrels if not rid.startswith(q.source_patent_id))
    grades = {}
    for q in pooled.queries:
        for v in q.qrels.values():
            grades[v] = grades.get(v, 0) + 1

    print(f"\nwrote {args.out}")
    print(f"  queries                       : {len(pooled.queries)}")
    print(f"  mean relevant before pooling  : {before:.1f}")
    print(f"  mean relevant after pooling   : {after:.1f}")
    print(f"  cross-patent relevant records : {cross}")
    print(f"  grade distribution            : {dict(sorted(grades.items()))}")
    print(f"\nEvaluate against it:\n"
          f"  python scripts/evaluate.py --eval-path {args.out} "
          f"--systems bm25 dense hybrid --baseline bm25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Query the index.

    python scripts/search.py --query "flexible fibre spoke between hub and rim"
    python scripts/search.py --query "..." --method hybrid_reranked --classification-prefix B60B
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import EMBEDDER, INDEX_NAME, OPENSEARCH_HOST
from patsearch.embeddings.service import PRESETS, create_embedder
from patsearch.pipeline import search
from patsearch.reranking.service import RERANKER_PRESETS, create_reranker
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.query import Filters

METHODS = ("bm25", "dense", "hybrid", "hybrid_reranked")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--method", choices=METHODS, default="hybrid")
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument(
        "--embedder", default=EMBEDDER,
        help=f"preset or provider spec. Presets: {', '.join(sorted(PRESETS))}",
    )
    ap.add_argument("--dimensions", type=int)
    ap.add_argument(
        "--reranker", default="llm-mini",
        help=f"reranker preset or spec. Presets: {', '.join(sorted(RERANKER_PRESETS))}",
    )
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--classification-prefix")
    ap.add_argument("--title-keyword")
    ap.add_argument("--abstract-keyword")
    ap.add_argument("--exact-title")
    ap.add_argument("--record-type", action="append", default=[])
    ap.add_argument("--independent-only", action="store_true")
    ap.add_argument("--exclude-patent")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    filters = Filters(
        classification_prefix=args.classification_prefix,
        title_keyword=args.title_keyword,
        abstract_keyword=args.abstract_keyword,
        exact_title=args.exact_title,
        record_types=args.record_type,
        independent_only=args.independent_only,
        exclude_patent_id=args.exclude_patent,
    )

    client = get_client(args.host)
    wait_for_health(client)

    embedder = None
    if args.method in ("dense", "hybrid", "hybrid_reranked"):
        kw = {"dimensions": args.dimensions} if args.dimensions else {}
        embedder = create_embedder(args.embedder, **kw)
    reranker = create_reranker(args.reranker) if args.method == "hybrid_reranked" else None

    outcome = search(
        client, args.index, args.query,
        method=args.method, filters=filters, embedder=embedder, reranker=reranker,
        candidates=args.candidates, top_k=args.top_k,
    )

    if args.json:
        print(json.dumps(outcome.to_dict(), indent=2))
        return 0

    print(f"\nquery   : {outcome.query}")
    print(f"method  : {outcome.method}")
    if not filters.is_empty():
        print(f"filters : {filters}")
    print("timings : " + "  ".join(f"{k}={v:.0f}ms" for k, v in outcome.timings_ms.items())
          + f"  TOTAL={sum(outcome.timings_ms.values()):.0f}ms")
    print(f"{len(outcome.hits)} records -> {len(outcome.patents)} patents\n")

    for i, r in enumerate(outcome.patents, 1):
        b = r.best
        loc = f"claim {b.claim_number}" if b.claim_number else b.record_type
        ranks = []
        if b.bm25_rank:
            ranks.append(f"bm25#{b.bm25_rank}")
        if b.vector_rank:
            ranks.append(f"vec#{b.vector_rank}")
        rank_s = f"  [{', '.join(ranks)}]" if ranks else ""
        print(f"{i:>2}. {r.patent_id}  {r.classification_raw:<11} score={r.score:.4f}{rank_s}")
        print(f"    {r.title[:88]}")
        print(f"    best match: {loc} - {b.text[:150].strip()}")
        if r.supporting:
            print(f"    +{len(r.supporting)} supporting record(s)")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

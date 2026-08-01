"""Build search records and load them into OpenSearch.

    python scripts/build_index.py                 # BM25 only (no embeddings)
    python scripts/build_index.py --embeddings    # BM25 + dense vectors
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import EMBEDDER, INDEX_NAME, OPENSEARCH_HOST
from patsearch.embeddings.service import PRESETS, create_embedder
from patsearch.pipeline import build_corpus, build_index
from patsearch.search.client import get_client, wait_for_health


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embeddings", action="store_true", help="compute and index dense vectors")
    ap.add_argument(
        "--embedder", default=EMBEDDER,
        help=f"preset or provider spec. Presets: {', '.join(sorted(PRESETS))}",
    )
    ap.add_argument("--dimensions", type=int, help="truncate OpenAI vectors to N dims")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--keep", action="store_true", help="do not recreate the index")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("building corpus from raw JSON ...")
    records = build_corpus()
    print(f"  {len(records)} search records in {(time.perf_counter()-t0)*1000:.0f} ms")

    client = get_client(args.host)
    wait_for_health(client)

    embedder = None
    if args.embeddings:
        kw = {"dimensions": args.dimensions} if args.dimensions else {}
        embedder = create_embedder(args.embedder, **kw)
        print(f"embedder: {args.embedder} -> {embedder.model_name}")
        print(f"  dimension {embedder.dimension}")

    print(f"indexing into '{args.index}' ...")
    result = build_index(
        client, args.index, records,
        embedder=embedder, recreate=not args.keep, batch_size=args.batch_size,
    )

    print(json.dumps(result, indent=2, default=str))
    if result["errors"]:
        print(f"FAILED: {result['errors']} documents rejected", file=sys.stderr)
        return 1
    print(f"\ntotal wall time {(time.perf_counter()-t0):.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

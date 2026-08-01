"""Queue-driven ingestion — the Part 2 proof-of-concept.

    python scripts/ingest.py --enqueue          # register the corpus
    python scripts/ingest.py --run              # process the queue
    python scripts/ingest.py --status           # live state breakdown
    python scripts/ingest.py --enqueue --run    # both

Re-running --enqueue after a completed pass reports everything skipped: the pipeline
is idempotent on (source_hash, parser_version, embedding_model).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import EMBEDDER, INDEX_NAME, OPENSEARCH_HOST, RAW_DIR, ROOT
from patsearch.embeddings.service import create_embedder
from patsearch.ingestion.status_store import StatusStore
from patsearch.ingestion.worker import PARSER_VERSION, IngestionWorker, enqueue_corpus
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.index import create_index

DB_PATH = ROOT / "data" / "ingestion.db"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset-stale", action="store_true", help="reclaim crashed workers' jobs")
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embeddings", action="store_true")
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--parser-version", default=PARSER_VERSION)
    ap.add_argument("--worker-id", default="worker-1")
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--max-batches", type=int)
    args = ap.parse_args()

    if not any([args.enqueue, args.run, args.status, args.reset_stale]):
        ap.error("pick at least one of --enqueue / --run / --status / --reset-stale")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    embedder = create_embedder(args.embedder) if args.embeddings else None
    model_id = embedder.model_name if embedder else None

    with StatusStore(DB_PATH) as store:
        if args.reset_stale:
            print(f"reclaimed {store.reclaim_stale(lease_seconds=0)} stale job(s)")

        if args.enqueue:
            res = enqueue_corpus(
                RAW_DIR, store,
                parser_version=args.parser_version, embedding_model=model_id,
            )
            print(f"enqueue: {res['queued']} queued, {res['skipped_unchanged']} skipped (unchanged)")

        if args.run:
            client = get_client(args.host)
            wait_for_health(client)
            create_index(
                client, args.index,
                dimension=embedder.dimension if embedder else None, recreate=False,
            )
            worker = IngestionWorker(
                store, client, args.index,
                worker_id=args.worker_id, embedder=embedder, raw_dir=RAW_DIR,
            )
            r = worker.run(batch_size=args.batch_size, max_batches=args.max_batches)
            print(
                f"worker {args.worker_id}: processed={r.processed} completed={r.completed} "
                f"failed={r.failed} records={r.records_indexed}"
            )

        if args.status or args.enqueue or args.run:
            print("\nstatus:")
            print(json.dumps(store.summary(), indent=2))
            fails = store.failures(limit=5)
            if fails:
                print("\nrecent failures:")
                for j in fails:
                    first = (j.last_error or "").splitlines()[0]
                    print(f"  {j.patent_id}  retries={j.retry_count}  {first[:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Scaling to 10M patents

Part 2 of the task. Everything here extrapolates from measurements taken on the
640-patent sample in this repo, not from guesswork. Measured basis:

| Measured on the sample | Value | Per patent |
|---|---|---|
| Patents | 640 | — |
| Search records produced | 18,743 | **29.3** |
| Raw text | 15.4 M chars | 24 K chars |
| Embedding tokens | ~4.4 M | **~6,900** |
| Record build (single core) | 2.5 s | 3.9 ms |
| Bulk index into OpenSearch | 2.2 s | 3.4 ms |

Scaled to 10^7 patents: **293 M records, ~240 GB raw text, ~69 B embedding tokens.**

---

## 1. Components

```
        USPTO bulk XML/JSON
                 │
                 ▼
      ┌──────────────────────┐
      │  Object store (S3)   │   immutable raw archive, versioned by release date
      └──────────┬───────────┘
                 │  manifest of new files
                 ▼
      ┌──────────────────────┐
      │   Ingestion queue    │   SQS / Kafka — one message per patent
      └──────────┬───────────┘
                 │
     ┌───────────┼───────────┬──────────────┐
     ▼           ▼           ▼              ▼
 ┌────────┐ ┌────────┐ ┌──────────┐  ┌────────────┐
 │ Parser │ │ Claim  │ │ Embedding│  │  Indexer   │
 │ workers│ │ recon. │ │  workers │  │  workers   │
 └───┬────┘ └───┬────┘ └────┬─────┘  └─────┬──────┘
     └──────────┴───────────┴──────────────┘
                 │ status writes
                 ▼
      ┌──────────────────────┐      ┌──────────────────────┐
      │  Job status DB       │      │  OpenSearch cluster  │
      │  (Postgres)          │      │  lexical + k-NN      │
      └──────────────────────┘      └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │  Query API (FastAPI) │
                                    └──────────────────────┘
```

| Component | Choice | Why |
|---|---|---|
| Raw archive | S3, versioned | Reprocessing must never require re-downloading from USPTO |
| Queue | SQS (or Kafka if replay matters) | Decouples parse rate from index rate; natural retry semantics |
| Status DB | Postgres | Transactional job claim; the POC in this repo is the same schema on SQLite |
| Search | OpenSearch, sharded | Lexical + vector + metadata filters in one query path |
| Model serving | Batch GPU workers | Embedding is the cost driver; batching is what makes it affordable |

## 2. Pipelines

### 2.1 Ingestion (batch, weekly)

USPTO publishes weekly. Each release triggers:

1. **Fetch** → land raw files in S3, record SHA-256 per patent.
2. **Enqueue** → one row per patent in `ingestion_jobs`, keyed by `patent_id`.
   If `(source_hash, parser_version, embedding_model)` are unchanged and the row is
   `COMPLETED`, it is skipped. This makes the whole pipeline idempotent — re-running a
   release is free.
3. **Parse + validate** → schema check, then claim reconstruction.
4. **Embed** → batched GPU inference.
5. **Index** → bulk write to OpenSearch, then mark `COMPLETED` with a record count.

Stages advance the job status, so at any moment `SELECT status, COUNT(*)` tells you
exactly where the corpus is.

### 2.2 Query (online)

```
query ──► filters resolved ──┬──► BM25 top-N ────┐
                             │                   ├──► RRF fusion ──► cross-encoder ──► group by patent
                             └──► k-NN top-N ────┘        (top 50)      (top 10)
```

Filters are applied as **pre-filters** in both branches. For k-NN this matters: the
Lucene engine filters during graph traversal, whereas post-filtering would return far
fewer than N results under a selective constraint like `classification_prefix=B60B`.

## 3. Status tracking and error handling

The `ingestion_jobs` table is the single source of truth. Implemented for real in
`src/patsearch/ingestion/status_store.py` and demonstrated by `scripts/ingest.py`.

```sql
patent_id TEXT PRIMARY KEY, source_file TEXT, source_hash TEXT,
status TEXT, parser_version TEXT, embedding_model TEXT,
record_count INT, retry_count INT, last_error TEXT,
worker_id TEXT, leased_at REAL, updated_at REAL
```

States: `PENDING → VALIDATING → RECONSTRUCTING → EMBEDDING → INDEXING → COMPLETED`,
with `FAILED` terminal after N retries.

| Failure | Handling |
|---|---|
| Malformed source record | Validation error → `FAILED` immediately, no retry (deterministic) |
| Transient API/network error | `retry_count++`, back to `PENDING`, exponential backoff |
| Worker crashes mid-job | Lease expires → `reclaim_stale()` returns it to `PENDING` |
| Poison message | `retry_count >= max` → `FAILED`, excluded from claims, surfaced in a dashboard |
| Bad parser release | Bump `parser_version` → affected patents re-queue automatically |
| Partial bulk-index failure | Per-document errors returned; job fails and retries whole patent (idempotent by `record_id`) |

Because `record_id` is deterministic (`{patent_id}:claim:{n}`), re-indexing overwrites
rather than duplicating. Retries are therefore safe at any point.

## 4. Cost

Assumptions: AWS us-east-1 on-demand, 10M patents, one full build plus ~350K new
patents/year. Figures are order-of-magnitude, not quotes.

### One-time backfill

| Item | Basis | Cost |
|---|---|---|
| Embeddings (hosted) | 69 B tokens × $0.02/1M (`text-embedding-3-small`) | **~$1,400** |
| Embeddings (self-hosted alt.) | MiniLM on 8× g5.xlarge, ~40 h | ~$350 |
| Parsing/compute | 293 M records ÷ 32 vCPU-hours | ~$50 |
| S3 storage | 240 GB raw + 100 GB processed | ~$8/mo |

Self-hosting a small open model is ~4× cheaper than the hosted API at this volume and
is the right call for a full backfill; the hosted API is the right call for the
incremental weekly delta, where operational simplicity dominates.

### Steady state (monthly)

| Item | Sizing | Cost |
|---|---|---|
| OpenSearch data nodes | 6 × `r6g.2xlarge` (vector-heavy) | ~$2,200 |
| EBS gp3 | 3 TB | ~$240 |
| Query/API tier | 3 × `c6g.large` | ~$110 |
| Weekly embeddings | ~7K patents/wk × 6.9K tokens | ~$40 |
| **Total** | | **~$2,600/mo** |

**Vector storage dominates.** 293 M × 1536 dims × 4 bytes = **1.8 TB** of raw float32,
and HNSW wants it resident. Three levers, in order of value:

1. **Reduce dimensions** — `text-embedding-3-small` supports Matryoshka truncation to
   512d at minor quality cost → 600 GB (already wired: `--dimensions 512`).
2. **Quantize** — int8 → 150 GB, ~4× cheaper nodes.
3. **Do not embed everything** — description passages are 38% of records and the least
   queried. Embedding only claims and abstracts drops vector count by ~40%.

Applying all three brings the cluster to roughly $700/mo.

## 5. Challenges at scale

| Challenge | Impact | Mitigation |
|---|---|---|
| **Vector index memory** | Dominates cost; naive float32 needs ~1.8 TB | Dimension reduction + int8 quantization + selective embedding |
| **Reindexing on model change** | Swapping embedding models invalidates 293 M vectors | Version the index; build the new one alongside and alias-swap. Never reindex in place |
| **Claim reconstruction drift** | Upstream XML format changes silently corrupt claims | The invariant test (every source entry consumed exactly once) runs per release; `number_inferred` rate is an alarm |
| **Filter selectivity** | `B60B` matches 47% here but a narrow CPC code may match 0.001% of 10M | Pre-filter in k-NN traversal; fall back to lexical-only below a cardinality threshold |
| **Long-tail patent sizes** | Claim counts range 1–46 here; some patents have 500+ | Per-patent record cap with overflow paging; bound worker memory |
| **Hot shards** | Recent filings queried far more than 1990s ones | Time-based index tiering — hot (2y) on fast nodes, cold on cheap storage |
| **Cross-encoder latency** | ~90 ms per 50 candidates, and it is serial | Cap candidates at 50; batch on GPU; make reranking opt-in per query |
| **Backfill duration** | 69 B tokens is days of wall-clock | Parallelise by weekly file; the queue makes this trivially horizontal |

## 6. What the proof-of-concept demonstrates

`scripts/ingest.py` runs the real pipeline against the real corpus with the real
status store. It shows, on actual data:

- **Idempotency** — re-running ingests nothing; every patent is reported skipped
- **Resumability** — kill it mid-run and stale leases are reclaimed on restart
- **Retry accounting** — induced failures increment `retry_count` and land in `FAILED`
- **Versioned reprocessing** — bumping `parser_version` re-queues the whole corpus
- **Visibility** — `--status` prints the live state breakdown at any time

The only change needed for production is the DSN: SQLite → Postgres. The claim
semantics (`BEGIN IMMEDIATE` + conditional update) are the same in both.

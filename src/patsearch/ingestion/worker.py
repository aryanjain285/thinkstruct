"""Queue-driven ingestion worker.

Each patent moves through PENDING -> VALIDATING -> RECONSTRUCTING -> EMBEDDING ->
INDEXING -> COMPLETED, with the status store as the single source of truth. Multiple
workers can run against one store: claims are transactional, so no patent is processed
twice, and a worker that dies mid-job has its lease reclaimed.
"""
from __future__ import annotations

import json
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opensearchpy import OpenSearch

from patsearch.embeddings.service import EmbeddingService
from patsearch.ingestion.loader import to_patent, validate_record
from patsearch.ingestion.status_store import JobStatus, StatusStore, content_hash
from patsearch.models import Patent
from patsearch.processing.reconstruct import reconstruct_claims
from patsearch.processing.records import build_records
from patsearch.search.index import index_records

PARSER_VERSION = "v1"

#: Parsed source files held in memory at once. Weekly files cluster in the queue,
#: so a handful is enough for a high hit rate without unbounded growth.
SOURCE_CACHE_SIZE = 4


def enqueue_corpus(
    raw_dir: Path,
    store: StatusStore,
    *,
    parser_version: str = PARSER_VERSION,
    embedding_model: str | None = None,
) -> dict[str, int]:
    """Register every patent. Unchanged, already-completed patents are skipped."""
    queued = skipped = 0
    for path in sorted(raw_dir.glob("patents_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for rec in payload:
            if not isinstance(rec, dict) or not rec.get("doc_number"):
                continue
            pid = str(rec["doc_number"]).strip()
            h = content_hash(json.dumps(rec, sort_keys=True, ensure_ascii=False))
            if store.enqueue(
                pid, path.name, h,
                parser_version=parser_version, embedding_model=embedding_model,
            ):
                queued += 1
            else:
                skipped += 1
    return {"queued": queued, "skipped_unchanged": skipped}


@dataclass(slots=True)
class WorkerResult:
    processed: int
    completed: int
    failed: int
    records_indexed: int


class IngestionWorker:
    def __init__(
        self,
        store: StatusStore,
        client: OpenSearch,
        index: str,
        *,
        worker_id: str = "worker-1",
        embedder: EmbeddingService | None = None,
        raw_dir: Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.store = store
        self.client = client
        self.index = index
        self.worker_id = worker_id
        self.embedder = embedder
        self.raw_dir = raw_dir
        self.max_retries = max_retries
        self._source_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _load_source(self, source_file: str, patent_id: str) -> dict[str, Any] | None:
        """Parsed source files are cached, bounded to SOURCE_CACHE_SIZE.

        Jobs are claimed in insertion order so patents from one file arrive together;
        a small cache gets nearly every hit. An unbounded one would hold the entire
        corpus in memory for the worker's lifetime.
        """
        cached = self._source_cache.get(source_file)
        if cached is None:
            path = (self.raw_dir or Path()) / source_file
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached = {
                str(r["doc_number"]).strip(): r
                for r in payload
                if isinstance(r, dict) and r.get("doc_number")
            }
            if len(self._source_cache) >= SOURCE_CACHE_SIZE:
                self._source_cache.pop(next(iter(self._source_cache)))
            self._source_cache[source_file] = cached
        else:
            self._source_cache.move_to_end(source_file)
        return cached.get(patent_id)

    def _process_one(self, job) -> int:
        """Run one patent through every stage. Returns records indexed."""
        rec = self._load_source(job.source_file, job.patent_id)
        if rec is None:
            raise ValueError(f"{job.patent_id} not found in {job.source_file}")

        issues = validate_record(rec, job.source_file)
        errors = [i for i in issues if i.severity == "error"]
        if errors:
            raise ValueError(f"validation failed: {[i.issue for i in errors]}")
        patent: Patent = to_patent(rec, job.source_file)

        self.store.advance(job.patent_id, JobStatus.RECONSTRUCTING)
        claims = reconstruct_claims(patent.patent_id, patent.claims_raw)
        records = build_records(patent, claims)
        if not records:
            raise ValueError("produced no search records")

        vectors = None
        if self.embedder is not None:
            self.store.advance(job.patent_id, JobStatus.EMBEDDING)
            vecs = self.embedder.embed_documents([r.text for r in records])
            vectors = {r.record_id: v for r, v in zip(records, vecs, strict=True)}

        self.store.advance(job.patent_id, JobStatus.INDEXING)
        ok, errs = index_records(
            self.client, self.index, records, vectors=vectors, refresh=False
        )
        if errs:
            raise RuntimeError(f"{len(errs)} documents rejected: {errs[0]}")
        return ok

    def run(self, *, batch_size: int = 25, max_batches: int | None = None) -> WorkerResult:
        processed = completed = failed = indexed = 0
        batches = 0
        while max_batches is None or batches < max_batches:
            jobs = self.store.claim(
                self.worker_id, limit=batch_size, max_retries=self.max_retries
            )
            if not jobs:
                break
            batches += 1
            for job in jobs:
                processed += 1
                try:
                    n = self._process_one(job)
                    self.store.complete(job.patent_id, record_count=n)
                    completed += 1
                    indexed += n
                except Exception as exc:
                    status = self.store.fail(
                        job.patent_id,
                        f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                        max_retries=self.max_retries,
                    )
                    if status is JobStatus.FAILED:
                        failed += 1

        self.client.indices.refresh(index=self.index)
        return WorkerResult(processed, completed, failed, indexed)

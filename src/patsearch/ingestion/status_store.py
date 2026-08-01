"""Ingestion job tracking — the Part 2 proof-of-concept.

SQLite deliberately: it is stdlib, needs no daemon or container, and has the same
semantics we would want from Postgres in production (transactional claim, unique
primary key, atomic status transitions). Swapping the DSN for Postgres is the only
change needed to run this across many machines.

Properties this demonstrates:
  idempotency          re-ingesting an unchanged file is a no-op (source_hash)
  versioned reprocess  bumping parser/embedding version re-queues affected patents
  visibility           every patent's stage is queryable at any time
  retry tracking       failures record the error and increment retry_count
  crash recovery       jobs stuck in-flight past a lease are returned to PENDING
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    PENDING = "PENDING"
    VALIDATING = "VALIDATING"
    RECONSTRUCTING = "RECONSTRUCTING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# Statuses that mean a worker holds the job right now.
IN_FLIGHT = (
    JobStatus.VALIDATING,
    JobStatus.RECONSTRUCTING,
    JobStatus.EMBEDDING,
    JobStatus.INDEXING,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    patent_id         TEXT PRIMARY KEY,
    source_file       TEXT NOT NULL,
    source_hash       TEXT NOT NULL,
    status            TEXT NOT NULL,
    parser_version    TEXT,
    embedding_model   TEXT,
    record_count      INTEGER NOT NULL DEFAULT 0,
    retry_count       INTEGER NOT NULL DEFAULT 0,
    last_error        TEXT,
    worker_id         TEXT,
    leased_at         REAL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON ingestion_jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_lease  ON ingestion_jobs(status, leased_at);
"""


def content_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class Job:
    patent_id: str
    source_file: str
    source_hash: str
    status: JobStatus
    parser_version: str | None
    embedding_model: str | None
    record_count: int
    retry_count: int
    last_error: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Job":
        return cls(
            patent_id=row["patent_id"],
            source_file=row["source_file"],
            source_hash=row["source_hash"],
            status=JobStatus(row["status"]),
            parser_version=row["parser_version"],
            embedding_model=row["embedding_model"],
            record_count=row["record_count"],
            retry_count=row["retry_count"],
            last_error=row["last_error"],
        )


class StatusStore:
    def __init__(self, path: Path | str = ":memory:") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StatusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ enqueue

    def enqueue(
        self,
        patent_id: str,
        source_file: str,
        source_hash: str,
        *,
        parser_version: str | None = None,
        embedding_model: str | None = None,
    ) -> bool:
        """Register a patent for ingestion.

        Returns True if work was queued, False if the existing row is already
        COMPLETED with the same content hash and versions (the idempotent path).
        """
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                "SELECT source_hash, status, parser_version, embedding_model "
                "FROM ingestion_jobs WHERE patent_id = ?",
                (patent_id,),
            ).fetchone()

            if row is not None:
                unchanged = (
                    row["source_hash"] == source_hash
                    and row["status"] == JobStatus.COMPLETED.value
                    and row["parser_version"] == parser_version
                    and row["embedding_model"] == embedding_model
                )
                if unchanged:
                    return False

            c.execute(
                """
                INSERT INTO ingestion_jobs
                    (patent_id, source_file, source_hash, status, parser_version,
                     embedding_model, record_count, retry_count, last_error,
                     worker_id, leased_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, 0, NULL, NULL, NULL, ?)
                ON CONFLICT(patent_id) DO UPDATE SET
                    source_file     = excluded.source_file,
                    source_hash     = excluded.source_hash,
                    status          = excluded.status,
                    parser_version  = excluded.parser_version,
                    embedding_model = excluded.embedding_model,
                    last_error      = NULL,
                    worker_id       = NULL,
                    leased_at       = NULL,
                    updated_at      = excluded.updated_at
                """,
                (patent_id, source_file, source_hash, JobStatus.PENDING.value,
                 parser_version, embedding_model, now),
            )
        return True

    # ------------------------------------------------------------------- claim

    def claim(self, worker_id: str, *, limit: int = 1, max_retries: int = 3) -> list[Job]:
        """Atomically take up to `limit` PENDING jobs for this worker."""
        now = time.time()
        with self._tx() as c:
            rows = c.execute(
                "SELECT * FROM ingestion_jobs WHERE status = ? AND retry_count < ? "
                "ORDER BY updated_at LIMIT ?",
                (JobStatus.PENDING.value, max_retries, limit),
            ).fetchall()
            if not rows:
                return []
            ids = [r["patent_id"] for r in rows]
            c.executemany(
                "UPDATE ingestion_jobs SET status = ?, worker_id = ?, leased_at = ?, "
                "updated_at = ? WHERE patent_id = ?",
                [(JobStatus.VALIDATING.value, worker_id, now, now, pid) for pid in ids],
            )
        return [Job.from_row(r) for r in rows]

    # ------------------------------------------------------------- transitions

    def advance(self, patent_id: str, status: JobStatus) -> None:
        now = time.time()
        with self._tx() as c:
            c.execute(
                "UPDATE ingestion_jobs SET status = ?, leased_at = ?, updated_at = ? "
                "WHERE patent_id = ?",
                (status.value, now, now, patent_id),
            )

    def complete(self, patent_id: str, *, record_count: int) -> None:
        now = time.time()
        with self._tx() as c:
            c.execute(
                "UPDATE ingestion_jobs SET status = ?, record_count = ?, last_error = NULL, "
                "worker_id = NULL, leased_at = NULL, updated_at = ? WHERE patent_id = ?",
                (JobStatus.COMPLETED.value, record_count, now, patent_id),
            )

    def fail(self, patent_id: str, error: str, *, max_retries: int = 3) -> JobStatus:
        """Record a failure. Returns PENDING if it will be retried, else FAILED."""
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                "SELECT retry_count FROM ingestion_jobs WHERE patent_id = ?", (patent_id,)
            ).fetchone()
            retries = (row["retry_count"] if row else 0) + 1
            status = JobStatus.PENDING if retries < max_retries else JobStatus.FAILED
            c.execute(
                "UPDATE ingestion_jobs SET status = ?, retry_count = ?, last_error = ?, "
                "worker_id = NULL, leased_at = NULL, updated_at = ? WHERE patent_id = ?",
                (status.value, retries, error[:2000], now, patent_id),
            )
        return status

    def reclaim_stale(self, *, lease_seconds: float = 300.0) -> int:
        """Return jobs whose worker died (held past the lease) to PENDING."""
        now = time.time()
        cutoff = now - lease_seconds
        with self._tx() as c:
            # '<=' not '<': the system clock has ~15ms granularity on Windows, so a job
            # leased and checked inside one tick compares equal and a strict '<' would
            # never reclaim it.
            cur = c.execute(
                "UPDATE ingestion_jobs SET status = ?, worker_id = NULL, leased_at = NULL, "
                "updated_at = ? WHERE status IN ({}) AND leased_at IS NOT NULL "
                "AND leased_at <= ?".format(",".join("?" * len(IN_FLIGHT))),
                (JobStatus.PENDING.value, now, *[s.value for s in IN_FLIGHT], cutoff),
            )
            return cur.rowcount

    # ------------------------------------------------------------------ query

    def get(self, patent_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT * FROM ingestion_jobs WHERE patent_id = ?", (patent_id,)
        ).fetchone()
        return Job.from_row(row) if row else None

    def counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) n FROM ingestion_jobs GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def failures(self, limit: int = 20) -> list[Job]:
        rows = self._conn.execute(
            "SELECT * FROM ingestion_jobs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            (JobStatus.FAILED.value, limit),
        ).fetchall()
        return [Job.from_row(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        c = self.counts()
        total = sum(c.values())
        done = c.get(JobStatus.COMPLETED.value, 0)
        return {
            "total": total,
            "by_status": c,
            "completed_pct": round(100 * done / total, 1) if total else 0.0,
            "records_indexed": self._conn.execute(
                "SELECT COALESCE(SUM(record_count), 0) s FROM ingestion_jobs"
            ).fetchone()["s"],
            "retries": self._conn.execute(
                "SELECT COALESCE(SUM(retry_count), 0) s FROM ingestion_jobs"
            ).fetchone()["s"],
        }

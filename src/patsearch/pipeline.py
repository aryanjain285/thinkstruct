"""End-to-end wiring: raw JSON -> records -> index, and query -> ranked patents."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from opensearchpy import OpenSearch

from patsearch.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR
from patsearch.embeddings.service import EmbeddingService
from patsearch.ingestion.loader import load_all, quality_report
from patsearch.models import SearchRecord
from patsearch.processing.reconstruct import reconstruct_claims, reconstruction_stats
from patsearch.processing.records import build_records
from patsearch.reranking.service import Reranker, rerank
from patsearch.search.index import create_index, index_records, index_stats
from patsearch.search.query import (
    Filters,
    Hit,
    PatentResult,
    Timer,
    aggregate_by_patent,
    bm25_search,
    dense_search,
    hybrid_search,
)

Method = Literal["bm25", "dense", "hybrid", "hybrid_reranked"]


def build_corpus(raw_dir: Path = RAW_DIR, *, write_reports: bool = True) -> list[SearchRecord]:
    """Load, validate, reconstruct claims, and emit search records."""
    patents, issues = load_all(raw_dir)
    all_claims, records = [], []
    for p in patents:
        claims = reconstruct_claims(p.patent_id, p.claims_raw)
        all_claims.extend(claims)
        records.extend(build_records(p, claims))

    if write_reports:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "data_quality.json").write_text(
            json.dumps(quality_report(patents, issues), indent=2), encoding="utf-8"
        )
        (REPORTS_DIR / "extraction_quality.json").write_text(
            json.dumps(reconstruction_stats(all_claims), indent=2), encoding="utf-8"
        )
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        with (PROCESSED_DIR / "search_records.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    return records


def build_index(
    client: OpenSearch,
    index: str,
    records: list[SearchRecord],
    *,
    embedder: EmbeddingService | None = None,
    recreate: bool = True,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Create the index and load records, embedding them if an embedder is given."""
    timer = Timer()
    dim = embedder.dimension if embedder else None

    with timer("create_index"):
        create_index(client, index, dimension=dim, recreate=recreate)

    vectors = None
    if embedder is not None:
        with timer("embed"):
            texts = [r.text for r in records]
            vecs = embedder.embed_documents(texts)
            vectors = {r.record_id: v for r, v in zip(records, vecs, strict=True)}

    with timer("index"):
        ok, errors = index_records(client, index, records, vectors=vectors, batch_size=batch_size)

    return {
        "indexed": ok,
        "errors": len(errors),
        "error_sample": errors[:3],
        "timings_ms": timer.stages,
        "stats": index_stats(client, index),
    }


@dataclass(slots=True)
class SearchOutcome:
    method: str
    query: str
    patents: list[PatentResult]
    hits: list[Hit]
    timings_ms: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "query": self.query,
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "total_ms": round(sum(self.timings_ms.values()), 2),
            "results": [
                {
                    "patent_id": r.patent_id,
                    "title": r.title,
                    "classification": r.classification_raw,
                    "score": round(r.score, 4),
                    "best_match": {
                        "record_type": r.best.record_type,
                        "claim_number": r.best.claim_number,
                        "bm25_rank": r.best.bm25_rank,
                        "vector_rank": r.best.vector_rank,
                        "rerank_score": (
                            round(r.best.rerank_score, 4) if r.best.rerank_score is not None else None
                        ),
                        "text": r.best.text[:400],
                    },
                    "supporting_records": len(r.supporting),
                }
                for r in self.patents
            ],
        }


def search(
    client: OpenSearch,
    index: str,
    query: str,
    *,
    method: Method = "hybrid",
    filters: Filters | None = None,
    embedder: EmbeddingService | None = None,
    reranker: Reranker | None = None,
    candidates: int = 50,
    top_k: int = 10,
) -> SearchOutcome:
    """Run one search end to end, timing every stage."""
    filters = filters or Filters()
    timer = Timer()

    if method in ("dense", "hybrid", "hybrid_reranked"):
        if embedder is None:
            raise ValueError(f"method '{method}' requires an embedder")
        with timer("embed_query"):
            vector = embedder.embed_query(query)

    if method == "bm25":
        with timer("retrieve"):
            hits = bm25_search(client, index, query, filters=filters, top_k=candidates)
    elif method == "dense":
        with timer("retrieve"):
            hits = dense_search(client, index, vector, filters=filters, top_k=candidates)
    else:
        with timer("retrieve"):
            hits = hybrid_search(
                client, index, query, vector, filters=filters,
                top_k=candidates, candidates=candidates,
            )

    if method == "hybrid_reranked":
        if reranker is None:
            raise ValueError("method 'hybrid_reranked' requires a reranker")
        with timer("rerank"):
            hits = rerank(reranker, query, hits)

    with timer("aggregate"):
        patents = aggregate_by_patent(hits, top_n=top_k)

    return SearchOutcome(
        method=method, query=query, patents=patents, hits=hits, timings_ms=timer.stages
    )

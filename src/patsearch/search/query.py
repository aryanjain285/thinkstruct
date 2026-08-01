"""Retrieval: BM25, dense vector, and hybrid fusion, all sharing one filter set."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import OpenSearch

from patsearch.models import RecordType

RRF_K = 60          # standard reciprocal-rank-fusion constant
PATENT_ALPHA = 0.3  # weight of supporting records in the patent-level score


@dataclass(slots=True)
class Filters:
    """Metadata constraints. All are optional and combine with AND."""

    classification_prefix: str | None = None
    title_keyword: str | None = None
    abstract_keyword: str | None = None
    exact_title: str | None = None
    record_types: list[RecordType | str] = field(default_factory=list)
    independent_only: bool = False
    exclude_patent_id: str | None = None

    def is_empty(self) -> bool:
        return not any(
            [
                self.classification_prefix, self.title_keyword, self.abstract_keyword,
                self.exact_title, self.record_types, self.independent_only,
                self.exclude_patent_id,
            ]
        )

    def to_clauses(self) -> tuple[list[dict], list[dict]]:
        """Return (filter_clauses, must_not_clauses)."""
        must: list[dict] = []
        must_not: list[dict] = []

        if self.classification_prefix:
            # Prefix on the raw code handles any depth: 'B', 'B60', 'B60B', 'B60B11'.
            must.append({"prefix": {"classification_raw": self.classification_prefix}})
        if self.title_keyword:
            must.append({"match": {"title": self.title_keyword}})
        if self.abstract_keyword:
            must.append({"match": {"abstract": self.abstract_keyword}})
        if self.exact_title:
            must.append({"term": {"title.exact": self.exact_title}})
        if self.record_types:
            vals = [rt.value if isinstance(rt, RecordType) else str(rt) for rt in self.record_types]
            must.append({"terms": {"record_type": vals}})
        if self.independent_only:
            must.append({"term": {"is_independent": True}})
        if self.exclude_patent_id:
            must_not.append({"term": {"patent_id": self.exclude_patent_id}})

        return must, must_not


@dataclass(slots=True)
class Hit:
    record_id: str
    patent_id: str
    record_type: str
    title: str
    text: str
    classification_raw: str
    score: float
    claim_number: int | None = None
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


def _to_hit(raw: dict) -> Hit:
    s = raw["_source"]
    return Hit(
        record_id=s.get("record_id", raw["_id"]),
        patent_id=s.get("patent_id", ""),
        record_type=s.get("record_type", ""),
        title=s.get("title", ""),
        text=s.get("text", ""),
        classification_raw=s.get("classification_raw", ""),
        score=float(raw.get("_score") or 0.0),
        claim_number=s.get("claim_number"),
    )


_SOURCE_FIELDS = [
    "record_id", "patent_id", "record_type", "title", "text",
    "classification_raw", "claim_number", "is_independent",
]


def bm25_search(
    client: OpenSearch, index: str, query: str, *, filters: Filters | None = None, top_k: int = 50
) -> list[Hit]:
    must, must_not = (filters or Filters()).to_clauses()
    body = {
        "size": top_k,
        "_source": _SOURCE_FIELDS,
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {
                        "query": query,
                        "fields": ["text^1.0", "title^2.0"],
                        "type": "best_fields",
                    }}
                ],
                "filter": must,
                "must_not": must_not,
            }
        },
    }
    res = client.search(index=index, body=body)
    return [_to_hit(h) for h in res["hits"]["hits"]]


def dense_search(
    client: OpenSearch, index: str, vector: list[float], *,
    filters: Filters | None = None, top_k: int = 50,
) -> list[Hit]:
    must, must_not = (filters or Filters()).to_clauses()
    knn: dict[str, Any] = {"vector": vector, "k": top_k}
    if must or must_not:
        # Pre-filter during graph traversal rather than post-filtering results,
        # which would return fewer than top_k under selective filters.
        knn["filter"] = {"bool": {"filter": must, "must_not": must_not}}
    body = {
        "size": top_k,
        "_source": _SOURCE_FIELDS,
        "query": {"knn": {"embedding": knn}},
    }
    res = client.search(index=index, body=body)
    return [_to_hit(h) for h in res["hits"]["hits"]]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], *, k: int = RRF_K, top_k: int = 50
) -> list[Hit]:
    """Fuse ranked lists by reciprocal rank. Score-scale independent, so BM25 and
    cosine can be combined without normalisation."""
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, Hit] = {}
    for list_idx, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.record_id] += 1.0 / (k + rank)
            if hit.record_id not in best:
                best[hit.record_id] = hit
            if list_idx == 0:
                best[hit.record_id].bm25_rank = rank
            else:
                best[hit.record_id].vector_rank = rank

    out = []
    for rid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]:
        h = best[rid]
        h.score = score
        out.append(h)
    return out


def hybrid_search(
    client: OpenSearch, index: str, query: str, vector: list[float], *,
    filters: Filters | None = None, top_k: int = 50, candidates: int = 50,
) -> list[Hit]:
    lexical = bm25_search(client, index, query, filters=filters, top_k=candidates)
    dense = dense_search(client, index, vector, filters=filters, top_k=candidates)
    return reciprocal_rank_fusion([lexical, dense], top_k=top_k)


@dataclass(slots=True)
class PatentResult:
    patent_id: str
    title: str
    classification_raw: str
    score: float
    best: Hit
    supporting: list[Hit]


def aggregate_by_patent(
    hits: list[Hit], *, alpha: float = PATENT_ALPHA, top_n: int = 10
) -> list[PatentResult]:
    """Group record hits into patent-level results.

    score = best record score + alpha * mean(next best up to three)
    """
    grouped: dict[str, list[Hit]] = defaultdict(list)
    for h in hits:
        grouped[h.patent_id].append(h)

    results = []
    for pid, group in grouped.items():
        group.sort(key=lambda h: h.score, reverse=True)
        best = group[0]
        rest = group[1:4]
        bonus = (sum(h.score for h in rest) / len(rest)) if rest else 0.0
        results.append(
            PatentResult(
                patent_id=pid,
                title=best.title,
                classification_raw=best.classification_raw,
                score=best.score + alpha * bonus,
                best=best,
                supporting=group[1:],
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


class Timer:
    """Accumulates named stage timings in milliseconds."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._start: float | None = None
        self._label: str | None = None

    def __call__(self, label: str) -> "Timer":
        self._label = label
        return self

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if self._start is not None and self._label:
            self.stages[self._label] = (time.perf_counter() - self._start) * 1000.0
        self._start = None

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

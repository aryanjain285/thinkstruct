"""Feature extraction for learning-to-rank.

Features come from the retrieval stage itself, so scoring costs microseconds — no
corpus reads, no model calls.

Ranks are reciprocal (1/(k+rank)) rather than raw, because rank 1 vs 2 matters far
more than 41 vs 42. A missing rank means that retriever never returned the document,
which is informative, so it maps to 0 rather than being imputed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from patsearch.search.query import Hit

RRF_K = 60

FEATURE_NAMES: tuple[str, ...] = (
    "bm25_rr",              # reciprocal BM25 rank, 0 if not retrieved lexically
    "vec_rr",               # reciprocal vector rank, 0 if not retrieved semantically
    "found_by_both",        # agreement between the two retrievers
    "bm25_only",
    "vec_only",
    "fused_score",          # the RRF score itself
    "is_claim",
    "is_abstract",
    "is_summary",
    "is_description",
    "is_independent",
    "claim_number_norm",    # early claims are broader and usually more relevant
    "text_len_norm",
    "query_len_norm",
    "term_overlap",         # lexical overlap, complements the semantic signal
    "title_overlap",
)


@dataclass(slots=True)
class FeatureRow:
    query_id: str
    record_id: str
    patent_id: str
    features: list[float]
    label: int


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def extract_features(query: str, hit: Hit) -> list[float]:
    """Feature vector for one (query, candidate) pair."""
    bm25_rr = 1.0 / (RRF_K + hit.bm25_rank) if hit.bm25_rank else 0.0
    vec_rr = 1.0 / (RRF_K + hit.vector_rank) if hit.vector_rank else 0.0
    has_bm25 = hit.bm25_rank is not None
    has_vec = hit.vector_rank is not None

    q_tok = _tokens(query)
    t_tok = _tokens(hit.text)
    overlap = len(q_tok & t_tok) / len(q_tok) if q_tok else 0.0
    title_overlap = len(q_tok & _tokens(hit.title)) / len(q_tok) if q_tok else 0.0

    rt = hit.record_type
    return [
        bm25_rr,
        vec_rr,
        1.0 if (has_bm25 and has_vec) else 0.0,
        1.0 if (has_bm25 and not has_vec) else 0.0,
        1.0 if (has_vec and not has_bm25) else 0.0,
        float(hit.score),
        1.0 if rt == "claim" else 0.0,
        1.0 if rt == "abstract" else 0.0,
        1.0 if rt == "summary" else 0.0,
        1.0 if rt == "description" else 0.0,
        1.0 if hit.claim_number is not None and hit.claim_number == 1 else 0.0,
        # log-scaled: the gap between claims 1 and 5 matters more than 40 and 45.
        1.0 / math.log2(2 + (hit.claim_number or 0)),
        min(len(hit.text) / 2000.0, 1.0),
        min(len(query) / 2000.0, 1.0),
        overlap,
        title_overlap,
    ]


def build_rows(
    query_id: str, query_text: str, hits: list[Hit], qrels: dict[str, int]
) -> list[FeatureRow]:
    """Feature rows for every candidate. Unjudged candidates are label 0, matching
    the TREC convention used when the qrels were built."""
    return [
        FeatureRow(
            query_id=query_id,
            record_id=h.record_id,
            patent_id=h.patent_id,
            features=extract_features(query_text, h),
            label=int(qrels.get(h.record_id, 0)),
        )
        for h in hits
    ]

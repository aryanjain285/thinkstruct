"""Standard IR metrics.

Pure functions over (ranked_ids, relevance_map) so they can be unit-tested against
hand-computed values. `qrels` maps doc_id -> graded relevance; anything absent is 0.

Graded scale used throughout:
    0 irrelevant | 1 broadly related | 2 materially relevant | 3 strong overlap
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def _rel(qrels: Mapping[str, int], doc_id: str) -> int:
    return int(qrels.get(doc_id, 0))


def recall_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """Fraction of all relevant documents retrieved in the top k."""
    relevant = {d for d, r in qrels.items() if r >= threshold}
    if not relevant:
        return 0.0
    found = sum(1 for d in ranked[:k] if d in relevant)
    return found / len(relevant)


def success_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """1.0 if at least one relevant document appears in the top k, else 0.0.

    Also called hit rate. This is the metric people usually *mean* when they read
    recall@k informally ("did the query find anything useful?"). Unlike recall it is
    not capped by how many relevant documents exist, so it is the honest headline for
    a corpus where queries have many relevant items.
    """
    return 1.0 if any(_rel(qrels, d) >= threshold for d in ranked[:k]) else 0.0


def recall_ceiling(qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """Maximum achievable recall@k given how many relevant documents exist.

    With 24 relevant documents, recall@10 cannot exceed 10/24 = 0.42. Reporting raw
    recall@k against 1.0 without this makes a correct system look broken.
    """
    n = sum(1 for r in qrels.values() if r >= threshold)
    return min(k, n) / n if n else 0.0


def precision_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for d in ranked[:k] if _rel(qrels, d) >= threshold)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """1/rank of the first relevant hit within k, else 0."""
    for i, d in enumerate(ranked[:k], start=1):
        if _rel(qrels, d) >= threshold:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain with exponential gain, as used by TREC."""
    return sum(
        (2 ** _rel(qrels, d) - 1) / math.log2(i + 1)
        for i, d in enumerate(ranked[:k], start=1)
    )


def ndcg_at_k(ranked: Sequence[str], qrels: Mapping[str, int], k: int) -> float:
    """DCG normalised by the best achievable ordering. 0 when nothing is relevant."""
    ideal = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 1) for i, r in enumerate(ideal, start=1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, qrels, k) / idcg


def average_precision(ranked: Sequence[str], qrels: Mapping[str, int], k: int, *, threshold: int = 1) -> float:
    """Mean of precision@i over the positions where a relevant doc appears."""
    relevant = {d for d, r in qrels.items() if r >= threshold}
    if not relevant:
        return 0.0
    hits = 0
    acc = 0.0
    for i, d in enumerate(ranked[:k], start=1):
        if d in relevant:
            hits += 1
            acc += hits / i
    return acc / min(len(relevant), k)


METRIC_FNS = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "mrr": reciprocal_rank,
    "ndcg": ndcg_at_k,
    "map": average_precision,
}


def evaluate_one(
    ranked: Sequence[str],
    qrels: Mapping[str, int],
    *,
    ks: Sequence[int] = (5, 10, 20, 50),
) -> dict[str, float]:
    """All metrics at all cutoffs for a single query."""
    out: dict[str, float] = {}
    for k in ks:
        out[f"success@{k}"] = success_at_k(ranked, qrels, k)
        out[f"recall@{k}"] = recall_at_k(ranked, qrels, k)
        out[f"recall_ceiling@{k}"] = recall_ceiling(qrels, k)
        out[f"precision@{k}"] = precision_at_k(ranked, qrels, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, qrels, k)
        out[f"mrr@{k}"] = reciprocal_rank(ranked, qrels, k)
        out[f"map@{k}"] = average_precision(ranked, qrels, k)
        # Strict view: only materially-relevant documents (grade >= 2) count. The
        # permissive view counts description passages graded 1, which inflates the
        # relevant-set size and depresses recall.
        out[f"success@{k}_strict"] = success_at_k(ranked, qrels, k, threshold=2)
        out[f"precision@{k}_strict"] = precision_at_k(ranked, qrels, k, threshold=2)
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Macro-average across queries — every query weighted equally."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(q.get(k, 0.0) for q in per_query) / len(per_query) for k in keys}

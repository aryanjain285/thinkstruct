# What every metric means

Plain-language reference for the numbers in the README and `reports/`. Implemented in
`src/patsearch/evaluation/metrics.py`, each tested against hand-computed values.

Throughout: **k** is how many results you look at, **relevant** means a graded label
of 1 or higher, and **strict** variants count only grade ≥ 2.

## The grading scale

Every `(query, candidate)` pair carries a grade from the examiner rubric:

| grade | meaning |
|---|---|
| 3 | strong overlap — same mechanism, structure or method |
| 2 | materially relevant — shares key technical elements |
| 1 | broadly related — same field, different technical solution |
| 0 | irrelevant |

---

## success@k — "did the search find anything useful?"

**Fraction of queries with at least one relevant result in the top k.**

The most intuitive metric and the one people usually *mean* when they say recall.
`success@10 = 0.64` means 64% of searches surface something relevant on page one.

Unlike recall it is **not capped** by how many relevant documents exist, so it is
readable on its own. Its weakness is saturation: on a narrow corpus where everything
is vaguely related, it hits 1.000 and stops discriminating between systems — which is
exactly what happened here.

## recall@k — "what share of everything relevant did we get?"

**Relevant documents retrieved in the top k, divided by all relevant documents.**

> ⚠️ **Read recall against its ceiling, not against 1.0.**
> If a query has 24 relevant documents, the top 10 can contain at most 10 of them, so
> `recall@10` cannot exceed 10/24 = **0.42**. A score of 0.21 is half of what is
> achievable, not 21% of perfect. `reports/evaluation_results.json` records
> `recall_ceiling@k` next to every recall figure for this reason.

Recall@50 is the metric that matters most for a **first-stage** retriever: a reranker
can reorder what retrieval found, but it can never recover a claim that was never
retrieved.

## precision@k — "how much of what I read was worth reading?"

**Fraction of the top k that is relevant.** `P@5 = 0.885` means roughly 4.4 of the
top 5 results are useful.

This is the attorney's-time metric. At $400–700/hour, precision is what stops search
from wasting money. It saturates when qrels are dense, so read it alongside nDCG.

## MRR@k — "how far down before the first good hit?"

**Reciprocal rank of the first relevant result**, averaged over queries. First result
relevant → 1.0; second → 0.5; third → 0.33; nothing in top k → 0.

`MRR@10 = 0.97` means the first useful result is essentially always at position 1.
It ignores everything after the first hit, so it says nothing about coverage.

## nDCG@k — "is the ordering right, weighted by how relevant each hit is?"

**Discounted Cumulative Gain, normalised by the best possible ordering.**

The only metric here that uses the *graded* scale rather than treating relevance as
yes/no. Two properties:

- **Gain**: a grade-3 hit is worth much more than a grade-1 (`2^grade − 1`).
- **Discount**: a hit at rank 8 counts less than the same hit at rank 1
  (`÷ log₂(rank+1)`).

Normalising by the ideal ordering puts it on 0–1, where 1.0 means perfect ranking.

**This is the headline metric for a reranker**, because reranking changes order
without changing membership. It is also the most robust of these when qrels are dense
and precision/MRR saturate.

## MAP@k — "precision, averaged over every relevant hit"

Mean of precision@i taken at each position where a relevant document appears. It
rewards packing relevant results early across the whole list rather than just at the
top. Reported for completeness; nDCG is generally preferred for graded relevance.

---

## Statistical significance

A raw difference between two systems on 80 queries means nothing on its own.
`0.590` vs `0.527` could easily be noise.

We use a **paired bootstrap** (`src/patsearch/evaluation/significance.py`):

1. Both systems run on the *same* queries, so per-query scores are paired.
2. Resample query indices with replacement 10,000 times.
3. Count how often the resampled difference contradicts the observed one.

That proportion is the **p-value**. `p < 0.05` conventionally means "unlikely to be
chance". Distribution-free, so it assumes nothing about the shape of the scores —
which matters, because IR metrics are bounded and skewed, not normal.

A paired t-test is also provided because reviewers expect to see one.

| result | how to read it |
|---|---|
| `+0.0627  p=0.0003 *` | real improvement, very unlikely to be chance |
| `+0.0129  p=0.1839` | directionally better, **cannot be claimed** |
| `+0.0000  p=1.0000` | genuinely identical |

---

## Which metric to quote when

| question | metric |
|---|---|
| Does search find anything useful? | success@10 |
| Is the first-stage retriever good enough to feed a reranker? | recall@50 |
| Is the reranker improving the order? | nDCG@10 |
| How much of an attorney's reading time is wasted? | precision@5 |
| How fast do they see the first useful result? | MRR@10 |
| Is a difference between two systems real? | the p-value, always |

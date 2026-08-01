"""Statistical significance testing for system comparisons.

Reporting that system A scores 0.295 and system B scores 0.272 says nothing without
a significance test — on 80 queries that difference is well within noise. TREC-style
evaluation requires a paired test, because the same queries are run through both
systems and the per-query scores are correlated.

Two tests are provided:
  paired_bootstrap  distribution-free, the safer default for IR metrics which are
                    bounded, skewed, and definitely not normal
  paired_t_test     the classic; included because reviewers expect to see it

Both are two-sided and operate on per-query score vectors.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(slots=True)
class TestResult:
    metric: str
    system_a: str
    system_b: str
    mean_a: float
    mean_b: float
    delta: float
    p_value: float
    n: int
    test: str

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        star = "*" if self.significant else " "
        return (
            f"{self.metric:<12} {self.system_a} {self.mean_a:.4f} vs "
            f"{self.system_b} {self.mean_b:.4f}  "
            f"delta={self.delta:+.4f}  p={self.p_value:.4f}{star}"
        )


def paired_bootstrap(
    a: list[float],
    b: list[float],
    *,
    iterations: int = 10_000,
    seed: int = 7,
) -> float:
    """Two-sided p-value for mean(a) - mean(b) under the paired bootstrap.

    Resamples query indices with replacement, recomputes the paired difference, and
    counts how often the resampled difference reverses sign relative to the observed
    one. Distribution-free, so it makes no normality assumption.
    """
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length score vectors")
    n = len(a)
    if n == 0:
        return 1.0

    diffs = [x - y for x, y in zip(a, b)]
    observed = sum(diffs) / n
    if observed == 0:
        return 1.0

    rng = random.Random(seed)
    # Centre the differences so the resampling distribution matches the null.
    centred = [d - observed for d in diffs]
    extreme = 0
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += centred[rng.randrange(n)]
        if abs(total / n) >= abs(observed):
            extreme += 1
    return (extreme + 1) / (iterations + 1)   # add-one keeps p strictly positive


def paired_t_test(a: list[float], b: list[float]) -> float:
    """Two-sided p-value from a paired t-test, via a normal approximation to the
    t distribution (adequate for n >= 30, which any usable eval set exceeds)."""
    if len(a) != len(b):
        raise ValueError("paired test needs equal-length score vectors")
    n = len(a)
    if n < 2:
        return 1.0

    diffs = [x - y for x, y in zip(a, b)]
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    if var == 0:
        return 1.0 if mean == 0 else 0.0

    t = mean / math.sqrt(var / n)
    # Two-sided normal tail.
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def compare_systems(
    results: list,
    metric: str,
    *,
    baseline: str | None = None,
    test: str = "bootstrap",
    iterations: int = 10_000,
) -> list[TestResult]:
    """Compare every system against a baseline on one metric.

    `results` are SystemResult objects; per-query scores are pulled from `per_query`.
    """
    by_name = {r.name: r for r in results}
    if not by_name:
        return []
    base_name = baseline or results[0].name
    if base_name not in by_name:
        raise ValueError(f"baseline {base_name!r} not among {sorted(by_name)}")

    def scores(r) -> tuple[list[str], list[float]]:
        ids = [q["query_id"] for q in r.per_query]
        return ids, [float(q.get(metric, 0.0)) for q in r.per_query]

    base_ids, base_scores = scores(by_name[base_name])
    out: list[TestResult] = []

    for r in results:
        if r.name == base_name:
            continue
        ids, sc = scores(r)
        # Align on query_id: a system that failed some queries must not shift the pairing.
        common = [q for q in base_ids if q in set(ids)]
        bi = {q: s for q, s in zip(base_ids, base_scores)}
        ri = {q: s for q, s in zip(ids, sc)}
        av = [ri[q] for q in common]
        bv = [bi[q] for q in common]

        p = (
            paired_bootstrap(av, bv, iterations=iterations)
            if test == "bootstrap"
            else paired_t_test(av, bv)
        )
        out.append(
            TestResult(
                metric=metric, system_a=r.name, system_b=base_name,
                mean_a=sum(av) / len(av) if av else 0.0,
                mean_b=sum(bv) / len(bv) if bv else 0.0,
                delta=(sum(av) - sum(bv)) / len(av) if av else 0.0,
                p_value=p, n=len(common), test=test,
            )
        )
    return out


def significance_table(tests: list[TestResult]) -> str:
    if not tests:
        return "(no comparisons)"
    lines = [
        f"paired {tests[0].test} test, two-sided, n={tests[0].n} queries "
        f"(* = p < 0.05)",
        "-" * 78,
    ]
    lines.extend(t.summary() for t in tests)
    return "\n".join(lines)


def to_trec_run(
    query_ids: list[str], rankings: list[list[str]], run_name: str
) -> str:
    """Export as a TREC run file so results can be scored with trec_eval."""
    lines = []
    for qid, ranked in zip(query_ids, rankings):
        for rank, doc_id in enumerate(ranked, start=1):
            lines.append(f"{qid} Q0 {doc_id} {rank} {1.0 / rank:.6f} {run_name}")
    return "\n".join(lines)

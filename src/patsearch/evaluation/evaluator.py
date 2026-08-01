"""Evaluation harness: build a test collection, run systems against it, compare.

The qrels here are generated from corpus structure, not human judgement. That measures
"can the system retrieve content from the patent this query came from" — a real and
useful signal for retrieval quality, but NOT the same as "can it find prior art in
other patents", which requires human labelling. The distinction is stated in every
report this module writes so results are never overclaimed.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from patsearch.evaluation.metrics import aggregate, evaluate_one


@dataclass(slots=True)
class EvalQuery:
    query_id: str
    text: str
    source_patent_id: str
    query_type: str
    qrels: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class EvalSet:
    queries: list[EvalQuery]
    notes: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"_notes": self.notes}, ensure_ascii=False) + "\n")
            for q in self.queries:
                fh.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalSet":
        notes: dict[str, Any] = {}
        queries: list[EvalQuery] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                if "_notes" in obj:
                    notes = obj["_notes"]
                    continue
                queries.append(EvalQuery(**obj))
        return cls(queries, notes)


def build_eval_set(
    patents: list,
    records_by_patent: dict[str, list],
    *,
    n_queries: int = 60,
    seed: int = 13,
    min_claims: int = 2,
) -> EvalSet:
    """Generate a known-item test collection from corpus structure.

    Query types:
      abstract_to_claim  abstract as query, own independent claims are relevant
      claim_to_patent    an independent claim as query, own other records relevant

    Grading: independent claim of the source patent = 3, other claims = 2,
    abstract/summary = 2, description passages = 1.
    """
    rng = random.Random(seed)
    eligible = [
        p for p in patents
        if p.abstract and len(records_by_patent.get(p.patent_id, [])) >= min_claims
    ]
    rng.shuffle(eligible)
    chosen = eligible[:n_queries]

    queries: list[EvalQuery] = []
    for i, p in enumerate(chosen):
        recs = records_by_patent[p.patent_id]
        qrels: dict[str, int] = {}
        for r in recs:
            rt = r.record_type.value if hasattr(r.record_type, "value") else str(r.record_type)
            if rt == "claim":
                qrels[r.record_id] = 3 if r.is_independent else 2
            elif rt in ("abstract", "summary"):
                qrels[r.record_id] = 2
            else:
                qrels[r.record_id] = 1

        # Query 1: the abstract. Exclude abstract/summary records of the same patent
        # from the qrels — retrieving the text you searched with is not a real result.
        a_qrels = {k: v for k, v in qrels.items() if not k.endswith((":abstract", ":summary"))}
        if a_qrels:
            queries.append(EvalQuery(
                query_id=f"q{i:03d}a", text=p.abstract, source_patent_id=p.patent_id,
                query_type="abstract_to_claim", qrels=a_qrels,
            ))

        # Query 2: an independent claim, if there is one.
        indep = [r for r in recs if getattr(r, "is_independent", False)]
        if indep:
            src = rng.choice(indep)
            c_qrels = {k: v for k, v in qrels.items() if k != src.record_id}
            if c_qrels:
                queries.append(EvalQuery(
                    query_id=f"q{i:03d}c", text=src.text, source_patent_id=p.patent_id,
                    query_type="claim_to_patent", qrels=c_qrels,
                ))

    return EvalSet(
        queries=queries,
        notes={
            "generation": "structural known-item, not human-judged",
            "measures": "retrieval of content from the query's own patent",
            "does_not_measure": "cross-patent prior-art relevance (needs human labels)",
            "seed": seed,
            "source_patents": len(chosen),
        },
    )


@dataclass(slots=True)
class SystemResult:
    name: str
    metrics: dict[str, float]
    per_query: list[dict[str, Any]]
    latency_ms: dict[str, float]


def evaluate_system(
    name: str,
    eval_set: EvalSet,
    run_query: Callable[[str], tuple[list[str], dict[str, float]]],
    *,
    ks: tuple[int, ...] = (5, 10, 20, 50),
    limit: int | None = None,
) -> SystemResult:
    """Run one system over the eval set.

    `run_query` takes query text and returns (ranked_record_ids, stage_timings_ms).
    """
    queries = eval_set.queries[:limit] if limit else eval_set.queries
    per_query: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    latencies: list[float] = []

    for q in queries:
        ranked, timings = run_query(q.text)
        m = evaluate_one(ranked, q.qrels, ks=ks)
        scores.append(m)
        total = sum(timings.values())
        latencies.append(total)
        per_query.append({
            "query_id": q.query_id,
            "query_type": q.query_type,
            "source_patent_id": q.source_patent_id,
            "n_relevant": len(q.qrels),
            "total_ms": round(total, 1),
            **{k: round(v, 4) for k, v in m.items()},
        })

    latencies.sort()
    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(int(round(p / 100 * (len(latencies) - 1))), len(latencies) - 1)]

    return SystemResult(
        name=name,
        metrics={k: round(v, 4) for k, v in aggregate(scores).items()},
        per_query=per_query,
        latency_ms={"p50": round(pct(50), 1), "p95": round(pct(95), 1),
                    "mean": round(sum(latencies) / len(latencies), 1) if latencies else 0.0},
    )


def comparison_table(results: list[SystemResult], metrics: tuple[str, ...]) -> str:
    """Fixed-width table for the console and the README."""
    head = f"{'system':<20}" + "".join(f"{m:>18}" for m in metrics) + f"{'P50 ms':>10}"
    lines = [head, "-" * len(head)]
    for r in results:
        row = f"{r.name:<20}"
        row += "".join(f"{r.metrics.get(m, 0.0):>18.4f}" for m in metrics)
        row += f"{r.latency_ms['p50']:>10.1f}"
        lines.append(row)

    # recall@k is bounded by min(k, |relevant|)/|relevant|. Without this line the
    # numbers read as if they were scored against 1.0, which they are not.
    ceil10 = results[0].metrics.get("recall_ceiling@10") if results else None
    ceil50 = results[0].metrics.get("recall_ceiling@50") if results else None
    if ceil10:
        lines.append("")
        lines.append(
            f"recall ceilings for this qrel density: recall@10 max {ceil10:.3f}, "
            f"recall@50 max {ceil50:.3f}"
        )
        lines.append("success@k = fraction of queries with >=1 relevant hit in top k "
                     "(_strict = grade >=2 only)")
    return "\n".join(lines)

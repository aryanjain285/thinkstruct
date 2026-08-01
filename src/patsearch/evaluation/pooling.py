"""TREC-style pooling with an LLM assessor.

The structural eval set only marks a query's own patent as relevant, so a system that
correctly surfaces a *different* patent covering the same mechanism is scored as
wrong. That measures known-item retrieval, not prior-art search.

Pooling fixes it the way TREC has since 1992:

  1. run every system on every query and take the top-D results
  2. union and dedupe the candidates into a pool
  3. judge each (query, candidate) pair on a graded relevance scale
  4. treat anything unjudged as non-relevant

Judgements are cached on disk by (query_id, record_id), so re-running costs nothing
and an interrupted run resumes where it stopped. Human assessors are the gold
standard; an LLM assessor is the affordable approximation and is adequate for
*relative* comparison between systems, which is what we need.
"""
from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from patsearch.evaluation.evaluator import EvalQuery, EvalSet

RUBRIC = (
    "You are a patent examiner assessing prior-art relevance.\n\n"
    "Score how technically relevant each passage is to the search query:\n"
    "  3 = strong overlap - describes the same mechanism, structure or method\n"
    "  2 = materially relevant - shares key technical elements a searcher would want\n"
    "  1 = broadly related - same general field, but different technical solution\n"
    "  0 = irrelevant\n\n"
    "Judge technical substance only. Wording similarity is not relevance: a passage "
    "using different vocabulary for the same mechanism scores high, and a passage "
    "sharing jargon but solving a different problem scores low.\n"
    "Do not make novelty, validity or infringement determinations."
)


@dataclass(slots=True)
class PoolEntry:
    query_id: str
    record_id: str
    patent_id: str
    text: str
    found_by: list[str] = field(default_factory=list)
    best_rank: int = 10**6


def build_pool(
    queries: list[EvalQuery],
    run_system: Callable[[str, str], list[tuple[str, str, str]]],
    systems: Iterable[str],
    *,
    depth: int = 15,
) -> dict[str, dict[str, PoolEntry]]:
    """Union the top-`depth` results of every system, per query.

    `run_system(system, query_text)` returns [(record_id, patent_id, text), ...].
    Returns {query_id: {record_id: PoolEntry}}.
    """
    pool: dict[str, dict[str, PoolEntry]] = {}
    for q in queries:
        entries: dict[str, PoolEntry] = {}
        for sysname in systems:
            for rank, (rid, pid, text) in enumerate(run_system(sysname, q.text)[:depth], 1):
                e = entries.get(rid)
                if e is None:
                    e = PoolEntry(q.query_id, rid, pid, text)
                    entries[rid] = e
                e.found_by.append(sysname)
                e.best_rank = min(e.best_rank, rank)
        pool[q.query_id] = entries
    return pool


class JudgementCache:
    """Disk-backed cache keyed by (query_id, record_id). Append-only JSONL."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[tuple[str, str], int] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                self._data[(r["query_id"], r["record_id"])] = int(r["relevance"])

    def get(self, qid: str, rid: str) -> int | None:
        return self._data.get((qid, rid))

    def put(self, qid: str, rid: str, rel: int) -> None:
        if (qid, rid) in self._data:
            return
        self._data[(qid, rid)] = rel
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"query_id": qid, "record_id": rid, "relevance": rel}) + "\n")

    def __len__(self) -> int:
        return len(self._data)


class LLMAssessor:
    """Grades (query, passage) pairs. Batched to keep the call count down."""

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        *,
        batch_size: int = 10,
        max_chars: int = 1100,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(max_retries=4, timeout=120.0)
        return self._client

    def judge(self, query: str, passages: list[str]) -> list[int]:
        if not passages:
            return []
        listing = "\n\n".join(f"[{i}] {p[: self.max_chars]}" for i, p in enumerate(passages))
        resp = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": RUBRIC},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nPassages:\n{listing}\n\n"
                        'Respond with JSON only: {"judgements":[{"id":0,"relevance":2},...]} '
                        "covering every passage id."
                    ),
                },
            ],
        )
        # Default 0: an unjudged document is non-relevant, per the TREC convention.
        out = [0] * len(passages)
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            for item in data.get("judgements", []):
                i = int(item["id"])
                if 0 <= i < len(passages):
                    out[i] = max(0, min(3, int(item["relevance"])))
        except (ValueError, KeyError, TypeError):
            pass
        return out


def judge_pool(
    queries: list[EvalQuery],
    pool: dict[str, dict[str, PoolEntry]],
    assessor: LLMAssessor,
    cache: JudgementCache,
    *,
    progress: bool = True,
) -> dict[str, dict[str, int]]:
    """Judge every pooled candidate. Returns {query_id: {record_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    by_id = {q.query_id: q for q in queries}
    total_calls = 0

    for qi, (qid, entries) in enumerate(pool.items(), 1):
        q = by_id[qid]
        judged: dict[str, int] = {}
        pending: list[PoolEntry] = []

        for e in entries.values():
            cached = cache.get(qid, e.record_id)
            if cached is None:
                pending.append(e)
            else:
                judged[e.record_id] = cached

        for i in range(0, len(pending), assessor.batch_size):
            batch = pending[i : i + assessor.batch_size]
            scores = assessor.judge(q.text, [e.text for e in batch])
            total_calls += 1
            for e, s in zip(batch, scores, strict=True):
                cache.put(qid, e.record_id, s)
                judged[e.record_id] = s

        qrels[qid] = judged
        if progress and qi % 10 == 0:
            print(f"  judged {qi}/{len(pool)} queries ({total_calls} API calls)", flush=True)

    return qrels


def merge_qrels(
    structural: dict[str, int], judged: dict[str, int], *, prefer_judged: bool = True
) -> dict[str, int]:
    """Combine same-patent structural labels with pooled judgements.

    Structural labels are reliable for a query's own patent — those records genuinely
    belong to it. Judged labels cover everything else. Where both exist, the assessor
    wins by default, because it saw the actual text.
    """
    merged = dict(structural)
    for rid, rel in judged.items():
        if prefer_judged or rid not in merged:
            merged[rid] = rel
    return {k: v for k, v in merged.items() if v > 0}


def apply_to_eval_set(
    eval_set: EvalSet, judged: dict[str, dict[str, int]], *, keep_structural: bool = True
) -> EvalSet:
    """Return a new EvalSet whose qrels include cross-patent judgements."""
    from dataclasses import replace

    out = []
    for q in eval_set.queries:
        j = judged.get(q.query_id, {})
        new = merge_qrels(q.qrels if keep_structural else {}, j)
        if new:
            out.append(replace(q, qrels=new))

    notes = dict(eval_set.notes)
    notes.update({
        "generation": "TREC-style pooling with an LLM assessor",
        "measures": "cross-patent prior-art relevance, not only known-item retrieval",
        "does_not_measure": "absolute quality — an LLM assessor approximates, but does "
                            "not replace, a human examiner's judgement",
        "assessor": "LLM (approximation of a human assessor; adequate for relative "
                    "system comparison, not an absolute quality ceiling)",
        "unjudged_policy": "documents outside the pool are treated as non-relevant",
        "queries_with_judgements": len(judged),
    })
    return EvalSet(queries=out, notes=notes)


def pool_stats(pool: dict[str, dict[str, PoolEntry]]) -> dict[str, Any]:
    sizes = [len(v) for v in pool.values()]
    overlap = [
        sum(1 for e in v.values() if len(set(e.found_by)) > 1) for v in pool.values()
    ]
    return {
        "queries": len(pool),
        "total_candidates": sum(sizes),
        "mean_pool_size": round(sum(sizes) / len(sizes), 1) if sizes else 0,
        "max_pool_size": max(sizes) if sizes else 0,
        "mean_found_by_multiple_systems": round(sum(overlap) / len(overlap), 1) if overlap else 0,
    }

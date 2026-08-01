"""Realistic query generation for evaluation.

Using a patent's own abstract as the query produces ~55% verbatim token overlap with
its claims, because both were drafted by the same person in the same vocabulary. That
inflates lexical retrieval and makes any dense/hybrid comparison meaningless.

These helpers rewrite queries the way an examiner would actually phrase them, and
measure the resulting overlap so the bias is visible in every report rather than
hidden.
"""
from __future__ import annotations

import os
import re
from dataclasses import replace

from patsearch.evaluation.evaluator import EvalQuery, EvalSet

_WORD = re.compile(r"[a-z0-9]+")

PARAPHRASE_PROMPT = (
    "You rewrite patent abstracts as short natural-language prior-art search queries.\n"
    "Rules:\n"
    "- One or two sentences, under 40 words.\n"
    "- Describe the technical idea in PLAIN language.\n"
    "- Deliberately AVOID reusing distinctive nouns and phrases from the source. "
    "Use ordinary synonyms (e.g. 'tyre' -> 'wheel covering', 'spoke' -> 'radial "
    "support', 'elastomeric composition' -> 'rubber blend').\n"
    "- Keep the underlying technical meaning exactly.\n"
    "- Output only the query text."
)


def tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def lexical_overlap(query: str, doc: str) -> float:
    """Fraction of query tokens that appear verbatim in the document."""
    q, d = tokens(query), tokens(doc)
    return len(q & d) / len(q) if q else 0.0


class LLMQueryParaphraser:
    def __init__(self, model_name: str = "gpt-5.4-mini", *, max_chars: int = 3000) -> None:
        self.model_name = model_name
        self.max_chars = max_chars
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(max_retries=4, timeout=90.0)
        return self._client

    def paraphrase(self, text: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": PARAPHRASE_PROMPT},
                {"role": "user", "content": text[: self.max_chars]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


def paraphrase_eval_set(
    eval_set: EvalSet,
    paraphraser: LLMQueryParaphraser,
    *,
    limit: int | None = None,
    progress: bool = True,
) -> EvalSet:
    """Rewrite every query, keeping qrels untouched so the two sets are comparable."""
    queries = eval_set.queries[:limit] if limit else eval_set.queries
    out: list[EvalQuery] = []
    for i, q in enumerate(queries, 1):
        try:
            new_text = paraphraser.paraphrase(q.text)
        except Exception:
            new_text = ""
        if not new_text:
            continue  # drop rather than silently keep the un-paraphrased original
        out.append(replace(q, text=new_text, query_type=f"{q.query_type}_paraphrased"))
        if progress and i % 10 == 0:
            print(f"  paraphrased {i}/{len(queries)}", flush=True)

    notes = dict(eval_set.notes)
    notes.update({
        "paraphrased_by": paraphraser.model_name,
        "generation": "structural known-item with LLM-paraphrased queries",
        "why": "raw abstracts share ~55% verbatim tokens with their own claims, "
               "which biases evaluation toward lexical retrieval",
        "dropped": len(queries) - len(out),
    })
    return EvalSet(queries=out, notes=notes)


def overlap_report(eval_set: EvalSet, doc_lookup) -> dict[str, float]:
    """Mean/median query-to-relevant-doc token overlap for an eval set."""
    import statistics as st

    vals = []
    for q in eval_set.queries:
        for rid in list(q.qrels)[:1]:
            doc = doc_lookup(rid)
            if doc:
                vals.append(lexical_overlap(q.text, doc))
    if not vals:
        return {}
    return {
        "mean_overlap": round(st.mean(vals), 4),
        "median_overlap": round(st.median(vals), 4),
        "n": len(vals),
    }

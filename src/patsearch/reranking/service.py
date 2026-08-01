"""Cross-encoder reranking of the candidate set returned by first-stage retrieval.

A cross-encoder scores (query, document) jointly, so it is far more accurate than
bi-encoder cosine but too slow to run over the whole corpus. It only ever sees the
top-N candidates.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from patsearch.search.query import Hit

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_CHARS = 2000  # claims run long; truncate to keep within the model's window


@runtime_checkable
class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, *, batch_size: int = 32,
                 device: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [(query, t[:MAX_CHARS]) for t in texts]
        return [float(s) for s in self.model.predict(pairs, batch_size=self.batch_size,
                                                     show_progress_bar=False)]


class LLMReranker:
    """Pointwise LLM relevance scoring — the task's "ask a language model to output
    rankings" option.

    Candidates are scored in groups so one call covers several documents. Scores are
    the same 0-3 graded scale the evaluation harness uses. Any candidate the model
    fails to score keeps a neutral value rather than being dropped, so the candidate
    set is never silently truncated.
    """

    RUBRIC = (
        "You score how relevant each patent passage is to a search query, for a patent "
        "examiner doing prior-art search.\n"
        "3 = strong technical overlap: same mechanism or structure\n"
        "2 = materially relevant: shares key technical elements\n"
        "1 = broadly related: same general field only\n"
        "0 = irrelevant\n"
        "Judge only technical content. Do not make novelty or infringement conclusions."
    )

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        *,
        group_size: int = 10,
        max_chars: int = 1200,
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.group_size = group_size
        self.max_chars = max_chars
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import os

            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(max_retries=4, timeout=90.0)
        return self._client

    def _score_group(self, query: str, texts: list[str]) -> list[float]:
        import json

        listing = "\n\n".join(
            f"[{i}] {t[: self.max_chars]}" for i, t in enumerate(texts)
        )
        # GPT-5-family models only accept the default temperature; sending one is a
        # 400. Older models benefit from temperature=0 for scoring stability.
        extra = {} if self.model_name.startswith("gpt-5") else {"temperature": self.temperature}
        resp = self.client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            **extra,
            messages=[
                {"role": "system", "content": self.RUBRIC},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nPassages:\n{listing}\n\n"
                        'Respond with JSON only: {"scores": [{"id": 0, "score": 2}, ...]} '
                        "covering every passage id."
                    ),
                },
            ],
        )
        scores = [1.0] * len(texts)  # neutral default if the model omits an id
        try:
            data = json.loads(resp.choices[0].message.content or "{}")
            for item in data.get("scores", []):
                i = int(item["id"])
                if 0 <= i < len(texts):
                    scores[i] = float(item["score"])
        except (ValueError, KeyError, TypeError):
            pass
        return scores

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        out: list[float] = []
        for i in range(0, len(texts), self.group_size):
            out.extend(self._score_group(query, texts[i : i + self.group_size]))
        return out


class IdentityReranker:
    """No-op used for the 'none' mode and in tests — preserves incoming order."""

    model_name = "identity"

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(len(texts) - i) for i in range(len(texts))]


RERANKERS = {
    "none": IdentityReranker,
    "cross-encoder": CrossEncoderReranker,
    "llm": LLMReranker,
}

#: Shorthands. HF cross-encoders need huggingface.co reachable; the llm:* variants
#: only need OPENAI_API_KEY, which is why they are the default in this environment.
RERANKER_PRESETS: dict[str, str] = {
    # HuggingFace cross-encoders (local inference, no per-query cost)
    "ce-minilm": "cross-encoder:cross-encoder/ms-marco-MiniLM-L-6-v2",   # fastest, 22M
    "ce-minilm-12": "cross-encoder:cross-encoder/ms-marco-MiniLM-L-12-v2",  # stronger
    "ce-electra": "cross-encoder:cross-encoder/ms-marco-electra-base",
    "bge-reranker": "cross-encoder:BAAI/bge-reranker-base",              # strong, 278M
    "bge-reranker-large": "cross-encoder:BAAI/bge-reranker-large",
    # Hosted LLM rerankers
    "llm-mini": "llm:gpt-5.4-mini",
    "llm-nano": "llm:gpt-5.4-nano",
    "llm": "llm:gpt-5.4-mini",
    "off": "none",
}


def create_reranker(spec: str = "llm-mini", **kwargs) -> Reranker:
    """Build a reranker from a preset or an explicit spec.

        create_reranker("llm-mini")                                  # preset
        create_reranker("ce-minilm")                                 # HF cross-encoder
        create_reranker("cross-encoder:BAAI/bge-reranker-base")      # any HF model id
        create_reranker("cross-encoder:D:/models/my-reranker")       # sideloaded dir
        create_reranker("llm:gpt-5.4-nano", group_size=20)
        create_reranker("none")                                      # no-op baseline

    All kinds satisfy the same protocol, so swapping one never changes calling code.
    """
    spec = RERANKER_PRESETS.get(spec, spec)
    kind, _, name = spec.partition(":")
    if kind not in RERANKERS:
        raise ValueError(
            f"unknown reranker {kind!r}; use one of {sorted(RERANKERS)} "
            f"or a preset: {sorted(RERANKER_PRESETS)}"
        )
    cls = RERANKERS[kind]
    if kind == "none":
        return cls()
    return cls(name, **kwargs) if name else cls(**kwargs)


def rerank(reranker: Reranker, query: str, hits: list[Hit], *, top_k: int | None = None) -> list[Hit]:
    """Rescore hits. The candidate set is never changed, so comparisons between
    rerankers stay controlled."""
    if not hits:
        return []
    scores = reranker.score(query, [h.text for h in hits])
    for h, s in zip(hits, scores, strict=True):
        h.rerank_score = s
        h.score = s
    # Coarse rerankers (an LLM emitting 0-3) produce many ties; the positional index
    # keeps first-stage order for equals rather than shuffling them arbitrarily.
    order = sorted(range(len(hits)), key=lambda i: (-hits[i].score, i))
    ordered = [hits[i] for i in order]
    return ordered[:top_k] if top_k else ordered

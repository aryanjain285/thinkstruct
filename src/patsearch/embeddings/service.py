"""Embedding services behind a narrow protocol.

The protocol lets tests inject a deterministic fake instead of downloading a model,
and lets the model be swapped without touching indexing or query code.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# bge/e5 models are trained with an asymmetric query prefix; omitting it costs recall.
# all-MiniLM is symmetric and needs none.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class EmbeddingService(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddings:
    """Production embedder. Loads lazily so importing the module stays cheap."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 64,
        query_prefix: str | None = None,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        # Pin a commit sha to make the artifact immutable; without it a repo owner
        # (or anyone who compromises the account) can change what 'main' resolves to.
        self.revision = revision
        self.query_prefix = (
            QUERY_PREFIX if query_prefix is None and "bge" in model_name else (query_prefix or "")
        )
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
                revision=self.revision,
                # Never execute custom modelling code shipped in a model repo.
                trust_remote_code=False,
            )
        return self._model

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # cosine == dot product downstream
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([self.query_prefix + text])[0]

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "dimension": self.dimension,
            "normalized": True,
            "query_prefix": self.query_prefix,
        }


class OpenAIEmbeddings:
    """OpenAI embedding provider.

    The key is read from OPENAI_API_KEY and never stored on the instance or logged.
    `dimensions` uses OpenAI's Matryoshka truncation to shrink vectors (and the index)
    at a small quality cost; None keeps the model default.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        batch_size: int = 128,
        dimensions: int | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimensions = dimensions
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import os

            from openai import OpenAI

            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Export it in your shell; do not hardcode it."
                )
            self._client = OpenAI(max_retries=self.max_retries, timeout=self.timeout)
        return self._client

    @property
    def dimension(self) -> int:
        if self._dimensions:
            return self._dimensions
        return len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i : i + self.batch_size]]
            kwargs: dict[str, object] = {"model": self.model_name, "input": batch}
            if self._dimensions:
                kwargs["dimensions"] = self._dimensions
            resp = self.client.embeddings.create(**kwargs)
            # The API may return items out of order; index is authoritative.
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def metadata(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "dimension": self._dimensions,
            "provider": "openai",
        }


#: Shorthand names so common models can be selected without typing the full id.
PRESETS: dict[str, str] = {
    "minilm": "st:sentence-transformers/all-MiniLM-L6-v2",   # 384d, Apache-2.0, fastest
    "mpnet": "st:sentence-transformers/all-mpnet-base-v2",   # 768d, stronger, ~3x slower
    "bge-small": "st:BAAI/bge-small-en-v1.5",                # 384d, top MTEB for size
    "bge-base": "st:BAAI/bge-base-en-v1.5",                  # 768d
    "e5-small": "st:intfloat/e5-small-v2",                   # 384d
    "gte-small": "st:thenlper/gte-small",                    # 384d
    "openai-small": "openai:text-embedding-3-small",         # 1536d, hosted
    "openai-large": "openai:text-embedding-3-large",         # 3072d, hosted
}


def create_embedder(spec: str = "minilm", **kwargs) -> EmbeddingService:
    """Build an embedder from a spec string.

        create_embedder("minilm")                                  # preset
        create_embedder("st:BAAI/bge-small-en-v1.5")               # any HF model id
        create_embedder("st:D:/models/all-MiniLM-L6-v2")           # sideloaded local dir
        create_embedder("openai:text-embedding-3-small", dimensions=512)
        create_embedder("hashing")                                 # tests, no network

    Every provider satisfies the same protocol, so indexing and query code is unchanged.
    """
    spec = PRESETS.get(spec, spec)
    provider, sep, name = spec.partition(":")
    if not sep:
        provider, name = provider, ""

    match provider:
        case "st" | "sentence-transformers":
            return SentenceTransformerEmbeddings(name or DEFAULT_MODEL, **kwargs)
        case "openai":
            return OpenAIEmbeddings(name or "text-embedding-3-small", **kwargs)
        case "hashing":
            return HashingEmbeddings(**kwargs)
        case _:
            raise ValueError(
                f"unknown embedder provider {provider!r}. "
                f"Use one of st/openai/hashing, or a preset: {sorted(PRESETS)}"
            )


class HashingEmbeddings:
    """Deterministic stand-in for tests. Same text always yields the same vector."""

    def __init__(self, dimension: int = 32) -> None:
        self.model_name = "hashing-test"
        self.dimension = dimension

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]
        norm = sum(v * v for v in raw) ** 0.5 or 1.0
        return [v / norm for v in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)

    def metadata(self) -> dict[str, object]:
        return {"model_name": self.model_name, "dimension": self.dimension}

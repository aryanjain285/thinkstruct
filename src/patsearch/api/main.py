"""FastAPI service exposing the search engine.

Models are loaded once at startup and reused; loading per-request would dominate
latency. If no embedder is available the service still starts and serves BM25 — vector
methods then return 503 rather than the whole API failing.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from patsearch.api.schemas import (
    CapabilitiesResponse,
    ClaimOut,
    HealthResponse,
    HitOut,
    PatentDetail,
    PatentOut,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from patsearch.config import EMBEDDER, INDEX_NAME, OPENSEARCH_HOST, RAW_DIR
from patsearch.embeddings.service import PRESETS, create_embedder
from patsearch.ingestion.loader import load_all
from patsearch.pipeline import search as run_search
from patsearch.processing.reconstruct import reconstruct_claims
from patsearch.reranking.service import RERANKER_PRESETS, create_reranker
from patsearch.search.client import get_client
from patsearch.search.index import index_stats
from patsearch.search.query import Filters

log = logging.getLogger("patsearch.api")

RERANKER_SPEC = os.environ.get("PATSEARCH_RERANKER", "llm-mini")
VECTOR_METHODS = {"dense", "hybrid", "hybrid_reranked"}

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["client"] = get_client(OPENSEARCH_HOST)

    try:
        emb = create_embedder(EMBEDDER)
        _ = emb.dimension  # force the load/probe now, not on first query
        state["embedder"] = emb
        log.info("embedder ready: %s", emb.model_name)
    except Exception as exc:
        state["embedder"] = None
        state["embedder_error"] = str(exc)
        log.warning("embedder unavailable, BM25 only: %s", exc)

    try:
        state["reranker"] = create_reranker(RERANKER_SPEC)
    except Exception as exc:
        state["reranker"] = None
        log.warning("reranker unavailable: %s", exc)

    yield
    state.clear()


app = FastAPI(
    title="Patent claim search",
    version="0.1.0",
    description="Hybrid BM25 + dense retrieval over USPTO vehicle patent applications.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _patent_index() -> dict[str, object]:
    """Lazily load raw patents for the detail endpoint. Cached for the process life."""
    patents, _ = load_all(RAW_DIR)
    return {p.patent_id: p for p in patents}


def _hit_out(h) -> HitOut:
    return HitOut(
        record_id=h.record_id, record_type=h.record_type, claim_number=h.claim_number,
        text=h.text, score=round(h.score, 6), bm25_rank=h.bm25_rank,
        vector_rank=h.vector_rank,
        rerank_score=round(h.rerank_score, 4) if h.rerank_score is not None else None,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    client = state["client"]
    try:
        cluster = client.cluster.health(request_timeout=5)["status"]
        docs = client.count(index=INDEX_NAME)["count"]
    except Exception as exc:
        raise HTTPException(503, f"OpenSearch unavailable: {exc}") from exc
    emb = state.get("embedder")
    return HealthResponse(
        status="ok", opensearch=cluster, index=INDEX_NAME, documents=docs,
        embedder=getattr(emb, "model_name", None), vector_search_available=emb is not None,
    )


@app.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    emb = state.get("embedder")
    methods = ["bm25"] + ([m for m in ("dense", "hybrid")] if emb else [])
    if emb and state.get("reranker"):
        methods.append("hybrid_reranked")
    return CapabilitiesResponse(
        methods=methods,
        embedder_presets=sorted(PRESETS),
        reranker_presets=sorted(RERANKER_PRESETS),
        active_embedder=getattr(emb, "model_name", None),
        active_reranker=getattr(state.get("reranker"), "model_name", None),
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    client = state["client"]
    try:
        base = index_stats(client, INDEX_NAME)
        agg = client.search(
            index=INDEX_NAME,
            body={
                "size": 0,
                "aggs": {
                    "patents": {"cardinality": {"field": "patent_id"}},
                    "cls": {"terms": {"field": "classification_subclass", "size": 20}},
                },
            },
        )["aggregations"]
    except Exception as exc:
        raise HTTPException(503, f"OpenSearch unavailable: {exc}") from exc
    return StatsResponse(
        documents=base["documents"],
        by_record_type=base["by_record_type"],
        patents=agg["patents"]["value"],
        classifications={b["key"]: b["doc_count"] for b in agg["cls"]["buckets"]},
    )


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    embedder = state.get("embedder")
    if req.method in VECTOR_METHODS and embedder is None:
        raise HTTPException(
            503,
            f"'{req.method}' needs an embedder, which failed to load "
            f"({state.get('embedder_error', 'unknown')}). Use method='bm25'.",
        )
    reranker = state.get("reranker")
    if req.method == "hybrid_reranked" and reranker is None:
        raise HTTPException(503, "reranker unavailable")

    filters = Filters(
        classification_prefix=req.classification_prefix,
        title_keyword=req.title_keyword,
        abstract_keyword=req.abstract_keyword,
        exact_title=req.exact_title,
        record_types=list(req.record_types),
        independent_only=req.independent_only,
        exclude_patent_id=req.exclude_patent_id,
    )

    try:
        outcome = run_search(
            state["client"], INDEX_NAME, req.query,
            method=req.method, filters=filters, embedder=embedder, reranker=reranker,
            candidates=req.candidates, top_k=req.top_k,
        )
    except Exception as exc:
        log.exception("search failed")
        raise HTTPException(500, f"search failed: {type(exc).__name__}: {exc}") from exc

    return SearchResponse(
        query=outcome.query,
        method=outcome.method,
        total_ms=round(sum(outcome.timings_ms.values()), 2),
        timings_ms={k: round(v, 2) for k, v in outcome.timings_ms.items()},
        candidates_retrieved=len(outcome.hits),
        results=[
            PatentOut(
                patent_id=r.patent_id, title=r.title, classification=r.classification_raw,
                score=round(r.score, 6), best_match=_hit_out(r.best),
                supporting=[_hit_out(h) for h in r.supporting[:5]],
            )
            for r in outcome.patents
        ],
    )


@app.get("/patents/{patent_id}", response_model=PatentDetail)
def patent_detail(
    patent_id: str,
    max_paragraphs: int = Query(40, ge=0, le=1000),
) -> PatentDetail:
    p = _patent_index().get(patent_id)
    if p is None:
        raise HTTPException(404, f"patent {patent_id} not found")
    claims = reconstruct_claims(p.patent_id, p.claims_raw)
    return PatentDetail(
        patent_id=p.patent_id, title=p.title, abstract=p.abstract,
        classification_raw=p.classification_raw,
        classification_subclass=p.classification_subclass,
        has_description=p.has_description,
        claims=[
            ClaimOut(
                claim_number=c.claim_number, text=c.text, is_independent=c.is_independent,
                depends_on=c.depends_on, status=c.status.value,
                number_inferred=c.number_inferred,
            )
            for c in claims
        ],
        description_paragraphs=[x for x in p.description_paragraphs if x.strip()][:max_paragraphs],
    )

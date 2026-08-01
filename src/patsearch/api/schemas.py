"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Method = Literal["bm25", "dense", "hybrid", "hybrid_reranked"]
RecordType = Literal["summary", "abstract", "claim", "description"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    method: Method = "hybrid"
    top_k: int = Field(10, ge=1, le=100)
    candidates: int = Field(50, ge=1, le=200)

    classification_prefix: str | None = Field(None, max_length=16)
    title_keyword: str | None = Field(None, max_length=200)
    abstract_keyword: str | None = Field(None, max_length=200)
    exact_title: str | None = Field(None, max_length=400)
    record_types: list[RecordType] = Field(default_factory=list)
    independent_only: bool = False
    exclude_patent_id: str | None = Field(None, max_length=32)


class HitOut(BaseModel):
    record_id: str
    record_type: str
    claim_number: int | None = None
    text: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    rerank_score: float | None = None


class PatentOut(BaseModel):
    patent_id: str
    title: str
    classification: str
    score: float
    best_match: HitOut
    supporting: list[HitOut] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    method: str
    total_ms: float
    timings_ms: dict[str, float]
    candidates_retrieved: int
    results: list[PatentOut]


class ClaimOut(BaseModel):
    claim_number: int | None
    text: str
    is_independent: bool
    depends_on: list[int]
    status: str
    number_inferred: bool


class PatentDetail(BaseModel):
    patent_id: str
    title: str
    abstract: str
    classification_raw: str
    classification_subclass: str
    has_description: bool
    claims: list[ClaimOut]
    description_paragraphs: list[str]


class HealthResponse(BaseModel):
    status: str
    opensearch: str
    index: str
    documents: int
    embedder: str | None
    vector_search_available: bool


class StatsResponse(BaseModel):
    documents: int
    by_record_type: dict[str, int]
    patents: int
    classifications: dict[str, int]


class CapabilitiesResponse(BaseModel):
    methods: list[str]
    embedder_presets: list[str]
    reranker_presets: list[str]
    active_embedder: str | None
    active_reranker: str | None

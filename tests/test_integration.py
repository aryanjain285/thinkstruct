"""End-to-end integration tests against a live OpenSearch index.

Skipped automatically when OpenSearch is unreachable or the index is missing, so the
unit suite still runs anywhere. Build the index first:

    python scripts/build_index.py --embeddings
"""
from __future__ import annotations

import re

import pytest

from patsearch.config import INDEX_NAME
from patsearch.pipeline import search
from patsearch.search.client import get_client
from patsearch.search.query import (
    Filters,
    Hit,
    aggregate_by_patent,
    bm25_search,
    dense_search,
    hybrid_search,
)


def _client_or_skip():
    try:
        c = get_client()
        c.cluster.health(request_timeout=3)
        if not c.indices.exists(index=INDEX_NAME):
            pytest.skip(f"index '{INDEX_NAME}' does not exist — run scripts/build_index.py")
        return c
    except pytest.skip.Exception:
        raise
    except Exception as exc:
        pytest.skip(f"OpenSearch unreachable: {exc}")


@pytest.fixture(scope="module")
def client():
    return _client_or_skip()


@pytest.fixture(scope="module")
def has_vectors(client):
    n = client.count(
        index=INDEX_NAME,
        body={"query": {"bool": {"must_not": [{"exists": {"field": "embedding"}}]}}},
    )["count"]
    return n == 0


@pytest.fixture(scope="module")
def embedder(has_vectors):
    if not has_vectors:
        pytest.skip("index has no embeddings")
    from patsearch.config import EMBEDDER
    from patsearch.embeddings.service import create_embedder

    try:
        e = create_embedder(EMBEDDER)
        e.embed_query("warmup")
        return e
    except Exception as exc:
        pytest.skip(f"embedder unavailable: {exc}")


# --------------------------------------------------------------- business cases

class TestBusinessUseCases:
    def test_plain_language_query_returns_on_topic_results(self, client, embedder):
        out = search(client, INDEX_NAME, "a wheel spoke made of carbon fibre bonded to the rim",
                     method="hybrid", embedder=embedder,
                     filters=Filters(classification_prefix="B60B"), top_k=5)
        assert out.patents
        titles = " ".join(r.title.lower() for r in out.patents)
        assert any(w in titles for w in ("spoke", "wheel", "rim", "composite"))

    def test_claim_as_query_excludes_source_patent(self, client, embedder):
        """Prior-art search on your own claim must not return your own patent."""
        claim = client.get(index=INDEX_NAME, id="20240051333:claim:1")["_source"]["text"]
        out = search(client, INDEX_NAME, claim, method="hybrid", embedder=embedder,
                     filters=Filters(exclude_patent_id="20240051333"), top_k=5)
        assert out.patents
        assert all(r.patent_id != "20240051333" for r in out.patents)

    def test_independent_claims_only(self, client):
        out = search(client, INDEX_NAME, "pneumatic tyre tread pattern", method="bm25",
                     filters=Filters(record_types=["claim"], independent_only=True), top_k=5)
        assert all(h.record_type == "claim" for h in out.hits)
        for h in out.hits[:5]:
            assert client.get(index=INDEX_NAME, id=h.record_id)["_source"]["is_independent"]

    def test_exact_title_lookup(self, client):
        out = search(client, INDEX_NAME, "wheel", method="bm25",
                     filters=Filters(exact_title="SPOKE"), top_k=10)
        assert out.hits
        assert all(h.title == "SPOKE" for h in out.hits)

    def test_abstract_keyword_filter(self, client):
        hits = bm25_search(client, INDEX_NAME, "tire",
                           filters=Filters(abstract_keyword="silica"), top_k=20)
        assert hits
        for h in hits[:5]:
            src = client.get(index=INDEX_NAME, id=h.record_id)["_source"]
            assert "silica" in src.get("abstract", "").lower()


class TestSpellingSynonyms:
    """British/US variants must be equivalent — European filings write 'tyre'."""

    @pytest.mark.parametrize("gb,us", [("tyre", "tire"), ("fibre", "fiber"),
                                       ("aluminium", "aluminum")])
    def test_variants_retrieve_equivalently(self, client, gb, us):
        a = bm25_search(client, INDEX_NAME, gb, top_k=20)
        b = bm25_search(client, INDEX_NAME, us, top_k=20)
        assert a and b
        overlap = len({h.record_id for h in a} & {h.record_id for h in b})
        assert overlap >= 15, f"{gb}/{us} overlap only {overlap}/20"

    def test_gb_spelling_works_with_filters(self, client):
        """Regression: 'tyre' + abstract filter returned zero before synonyms."""
        hits = bm25_search(client, INDEX_NAME, "tyre",
                           filters=Filters(abstract_keyword="silica"), top_k=20)
        assert hits


# ------------------------------------------------------------------- filtering

class TestFilterCorrectness:
    @pytest.mark.parametrize("prefix", ["B", "B60", "B60B", "B60C"])
    def test_classification_prefix_at_any_depth(self, client, prefix):
        hits = bm25_search(client, INDEX_NAME, "wheel tyre",
                           filters=Filters(classification_prefix=prefix), top_k=20)
        assert hits
        assert all(h.classification_raw.startswith(prefix) for h in hits)

    def test_dense_prefilters_and_still_fills_k(self, client, embedder):
        """Pre-filtering during traversal; post-filtering would return fewer than k."""
        v = embedder.embed_query("wheel spoke")
        hits = dense_search(client, INDEX_NAME, v,
                            filters=Filters(classification_prefix="B60B"), top_k=50)
        assert len(hits) >= 40
        assert all(h.classification_raw.startswith("B60B") for h in hits)

    def test_hybrid_prefilters_and_still_fills_k(self, client, embedder):
        v = embedder.embed_query("wheel spoke")
        hits = hybrid_search(client, INDEX_NAME, "wheel spoke", v,
                             filters=Filters(classification_prefix="B60B"), top_k=50)
        assert len(hits) >= 40
        assert all(h.classification_raw.startswith("B60B") for h in hits)

    def test_impossible_filter_returns_empty_not_error(self, client):
        assert bm25_search(client, INDEX_NAME, "wheel",
                           filters=Filters(classification_prefix="ZZZZ"), top_k=10) == []
        out = search(client, INDEX_NAME, "wheel", method="bm25",
                     filters=Filters(classification_prefix="ZZZZ"))
        assert out.patents == []


# ------------------------------------------------------------------ robustness

ADVERSARIAL = [
    ("single_char", "a"),
    ("very_long", "wheel spoke rim hub tyre " * 200),
    ("unicode", "roue à rayons en fibre de carbone 車輪 스포크"),
    ("lucene_metachars", 'wheel AND spoke OR NOT "rim" && || ! ( ) { } [ ] ^ ~ * ? : \\ /'),
    ("punctuation_only", "!!! ??? ***"),
    ("numbers_only", "12345 67890"),
    ("whitespace", "wheel\n\nspoke\t\trim"),
    ("emoji", "wheel 🛞 spoke"),
    ("json_injection", '{"query": {"match_all": {}}}'),
    ("script_tag", "<script>alert(1)</script> wheel"),
]


class TestAdversarialInput:
    @pytest.mark.parametrize("name,query", ADVERSARIAL, ids=[n for n, _ in ADVERSARIAL])
    def test_bm25_never_raises(self, client, name, query):
        hits = bm25_search(client, INDEX_NAME, query, top_k=5)
        assert isinstance(hits, list)

    @pytest.mark.parametrize("k", [1, 200, 1000])
    def test_extreme_top_k(self, client, k):
        assert len(bm25_search(client, INDEX_NAME, "wheel", top_k=k)) <= k

    def test_aggregate_empty(self):
        assert aggregate_by_patent([]) == []

    def test_aggregate_single(self):
        r = aggregate_by_patent([Hit("r", "p", "claim", "T", "x", "B60B", 1.0)])
        assert len(r) == 1 and r[0].score == pytest.approx(1.0)

    def test_dense_without_embedder_raises(self, client):
        with pytest.raises(ValueError, match="requires an embedder"):
            search(client, INDEX_NAME, "wheel", method="dense", embedder=None)

    def test_reranked_without_reranker_raises(self, client, embedder):
        with pytest.raises(ValueError, match="requires a reranker"):
            search(client, INDEX_NAME, "wheel", method="hybrid_reranked",
                   embedder=embedder, reranker=None)


# --------------------------------------------------------------- data integrity

_CANCEL_MARKER = re.compile(r"\(\s*cancell?ed\s*\)", re.IGNORECASE)


class TestIndexIntegrity:
    def test_expected_document_count(self, client):
        assert client.count(index=INDEX_NAME)["count"] == 18743

    def test_no_record_missing_text(self, client):
        n = client.count(index=INDEX_NAME, body={
            "query": {"bool": {"must_not": [{"exists": {"field": "text"}}]}}})["count"]
        assert n == 0

    def test_all_patents_represented(self, client):
        agg = client.search(index=INDEX_NAME, body={
            "size": 0, "aggs": {"p": {"cardinality": {"field": "patent_id"}}}})
        assert agg["aggregations"]["p"]["value"] == 640

    def test_no_literal_cancellation_markers_indexed(self, client):
        """Checked with a literal regex, not match_phrase: the english analyzer stems
        'cancelling' -> 'cancel', so an analysed query matches 'noise cancelling foam'."""
        res = client.search(index=INDEX_NAME, body={
            "size": 100, "_source": ["record_id", "text"],
            "query": {"match_phrase": {"text": "(canceled)"}}})
        leaked = [h["_source"]["record_id"] for h in res["hits"]["hits"]
                  if _CANCEL_MARKER.search(h["_source"]["text"])]
        assert leaked == []

    def test_every_record_has_a_patent_id(self, client):
        n = client.count(index=INDEX_NAME, body={
            "query": {"bool": {"must_not": [{"exists": {"field": "patent_id"}}]}}})["count"]
        assert n == 0

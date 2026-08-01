import pytest

from patsearch.embeddings.service import HashingEmbeddings
from patsearch.pipeline import METHODS, VECTOR_METHODS, search
from patsearch.reranking.service import IdentityReranker


class CountingEmbedder(HashingEmbeddings):
    """Records whether an embedding was actually computed."""

    def __init__(self):
        super().__init__(dimension=8)
        self.query_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return super().embed_query(text)


class ExplodingClient:
    """Any OpenSearch call is a test failure — validation must happen first."""

    def search(self, *a, **k):
        raise AssertionError("OpenSearch was queried before validation completed")


class TestValidationHappensBeforeWork:
    def test_unknown_method_rejected(self):
        with pytest.raises(ValueError, match="unknown method"):
            search(ExplodingClient(), "idx", "q", method="typo")

    def test_unknown_method_does_not_embed(self):
        """Regression: an invalid method used to fall through and raise NameError
        on an unbound `vector`."""
        e = CountingEmbedder()
        with pytest.raises(ValueError, match="unknown method"):
            search(ExplodingClient(), "idx", "q", method="hybird", embedder=e)
        assert e.query_calls == 0

    @pytest.mark.parametrize("method", sorted(VECTOR_METHODS))
    def test_missing_embedder_rejected(self, method):
        with pytest.raises(ValueError, match="requires an embedder"):
            search(ExplodingClient(), "idx", "q", method=method, embedder=None)

    def test_missing_reranker_rejected_before_embedding(self):
        """Regression: the reranker check used to run after retrieval, so a missing
        reranker cost an embedding API call and a full search first."""
        e = CountingEmbedder()
        with pytest.raises(ValueError, match="requires a reranker"):
            search(ExplodingClient(), "idx", "q", method="hybrid_reranked",
                   embedder=e, reranker=None)
        assert e.query_calls == 0

    def test_candidates_below_top_k_rejected(self):
        with pytest.raises(ValueError, match="must be >="):
            search(ExplodingClient(), "idx", "q", method="bm25",
                   candidates=5, top_k=10)

    def test_bm25_needs_no_embedder(self):
        """Should fail on the client, not on validation — proving bm25 is allowed through."""
        with pytest.raises(AssertionError, match="before validation completed"):
            search(ExplodingClient(), "idx", "q", method="bm25")

    def test_valid_reranked_call_passes_validation(self):
        with pytest.raises(AssertionError, match="before validation completed"):
            search(ExplodingClient(), "idx", "q", method="hybrid_reranked",
                   embedder=CountingEmbedder(), reranker=IdentityReranker())


class TestMethodConstants:
    def test_vector_methods_are_a_subset(self):
        assert VECTOR_METHODS < METHODS

    def test_bm25_is_not_a_vector_method(self):
        assert "bm25" in METHODS and "bm25" not in VECTOR_METHODS

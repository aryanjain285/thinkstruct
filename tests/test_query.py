import pytest

from patsearch.embeddings.service import HashingEmbeddings
from patsearch.models import RecordType
from patsearch.reranking.service import IdentityReranker, rerank
from patsearch.search.query import (
    Filters,
    Hit,
    Timer,
    aggregate_by_patent,
    reciprocal_rank_fusion,
)


def _hit(rid, pid, score, **kw):
    return Hit(
        record_id=rid, patent_id=pid, record_type=kw.pop("record_type", "claim"),
        title=kw.pop("title", "T"), text=kw.pop("text", "some text"),
        classification_raw=kw.pop("cls", "B60B104FI"), score=score, **kw,
    )


class TestFilters:
    def test_empty_produces_no_clauses(self):
        must, must_not = Filters().to_clauses()
        assert must == [] and must_not == []
        assert Filters().is_empty()

    def test_classification_prefix_uses_prefix_query(self):
        must, _ = Filters(classification_prefix="B60B").to_clauses()
        assert must == [{"prefix": {"classification_raw": "B60B"}}]

    def test_prefix_works_at_any_depth(self):
        for p in ("B", "B60", "B60B", "B60B11"):
            must, _ = Filters(classification_prefix=p).to_clauses()
            assert must[0]["prefix"]["classification_raw"] == p

    def test_exact_title_uses_keyword_subfield(self):
        must, _ = Filters(exact_title="SPOKE").to_clauses()
        assert must == [{"term": {"title.exact": "SPOKE"}}]

    def test_title_and_abstract_keywords_are_match_queries(self):
        must, _ = Filters(title_keyword="wheel", abstract_keyword="carbon").to_clauses()
        assert {"match": {"title": "wheel"}} in must
        assert {"match": {"abstract": "carbon"}} in must

    def test_record_types_accept_enum_or_str(self):
        must, _ = Filters(record_types=[RecordType.CLAIM, "abstract"]).to_clauses()
        assert must == [{"terms": {"record_type": ["claim", "abstract"]}}]

    def test_exclude_patent_goes_to_must_not(self):
        must, must_not = Filters(exclude_patent_id="123").to_clauses()
        assert must == []
        assert must_not == [{"term": {"patent_id": "123"}}]

    def test_independent_only(self):
        must, _ = Filters(independent_only=True).to_clauses()
        assert must == [{"term": {"is_independent": True}}]

    def test_combined_filters_all_present(self):
        f = Filters(classification_prefix="B60B", title_keyword="wheel",
                    exclude_patent_id="9", record_types=["claim"])
        must, must_not = f.to_clauses()
        assert len(must) == 3 and len(must_not) == 1
        assert not f.is_empty()


class TestReciprocalRankFusion:
    def test_single_list_preserves_order(self):
        hits = [_hit(f"r{i}", "p", 10 - i) for i in range(5)]
        out = reciprocal_rank_fusion([hits], top_k=5)
        assert [h.record_id for h in out] == ["r0", "r1", "r2", "r3", "r4"]

    def test_agreement_ranks_first(self):
        a = [_hit("x", "p1", 9), _hit("y", "p2", 8)]
        b = [_hit("x", "p1", 0.9), _hit("z", "p3", 0.8)]
        out = reciprocal_rank_fusion([a, b])
        assert out[0].record_id == "x"      # appears top of both lists

    def test_is_scale_independent(self):
        """BM25 scores in the tens and cosine in [0,1] must fuse without normalisation."""
        bm25 = [_hit("a", "p", 95.0), _hit("b", "p", 40.0)]
        dense = [_hit("b", "p", 0.91), _hit("a", "p", 0.89)]
        out = reciprocal_rank_fusion([bm25, dense])
        assert {h.record_id for h in out} == {"a", "b"}
        assert out[0].score == pytest.approx(1 / 61 + 1 / 62)

    def test_records_ranks_from_both_sources(self):
        bm25 = [_hit("a", "p", 9)]
        dense = [_hit("a", "p", 0.9)]
        out = reciprocal_rank_fusion([bm25, dense])
        assert out[0].bm25_rank == 1 and out[0].vector_rank == 1

    def test_respects_top_k(self):
        hits = [_hit(f"r{i}", "p", 1) for i in range(20)]
        assert len(reciprocal_rank_fusion([hits], top_k=5)) == 5

    def test_empty_input(self):
        assert reciprocal_rank_fusion([[], []]) == []


class TestAggregateByPatent:
    def test_groups_records_under_patents(self):
        hits = [_hit("a", "p1", 5), _hit("b", "p1", 3), _hit("c", "p2", 4)]
        out = aggregate_by_patent(hits)
        assert [r.patent_id for r in out] == ["p1", "p2"]
        assert out[0].best.record_id == "a"
        assert len(out[0].supporting) == 1

    def test_supporting_records_boost_score(self):
        alone = aggregate_by_patent([_hit("a", "p1", 5.0)], alpha=0.5)
        backed = aggregate_by_patent(
            [_hit("a", "p1", 5.0), _hit("b", "p1", 4.0)], alpha=0.5
        )
        assert backed[0].score > alone[0].score

    def test_alpha_zero_is_max_only(self):
        out = aggregate_by_patent([_hit("a", "p", 5.0), _hit("b", "p", 4.0)], alpha=0.0)
        assert out[0].score == pytest.approx(5.0)

    def test_bonus_uses_at_most_three_supporting(self):
        many = [_hit("a", "p", 10.0)] + [_hit(f"s{i}", "p", 1.0) for i in range(10)]
        out = aggregate_by_patent(many, alpha=1.0)
        assert out[0].score == pytest.approx(11.0)   # 10 + mean of three 1.0s

    def test_respects_top_n(self):
        hits = [_hit(f"r{i}", f"p{i}", i) for i in range(20)]
        assert len(aggregate_by_patent(hits, top_n=3)) == 3

    def test_empty(self):
        assert aggregate_by_patent([]) == []


class TestEmbeddings:
    def test_hashing_is_deterministic(self):
        e = HashingEmbeddings()
        assert e.embed_query("wheel") == e.embed_query("wheel")
        assert e.embed_query("wheel") != e.embed_query("tyre")

    def test_vectors_are_unit_norm(self):
        v = HashingEmbeddings(dimension=16).embed_query("spoke")
        assert sum(x * x for x in v) == pytest.approx(1.0, abs=1e-6)

    def test_dimension_respected(self):
        assert len(HashingEmbeddings(dimension=64).embed_query("x")) == 64

    def test_embed_documents_batches(self):
        out = HashingEmbeddings().embed_documents(["a", "b", "c"])
        assert len(out) == 3
        assert HashingEmbeddings().embed_documents([]) == []


class TestRerank:
    def test_preserves_candidate_set(self):
        hits = [_hit(f"r{i}", "p", i) for i in range(5)]
        out = rerank(IdentityReranker(), "q", hits)
        assert {h.record_id for h in out} == {f"r{i}" for i in range(5)}

    def test_sets_rerank_score(self):
        out = rerank(IdentityReranker(), "q", [_hit("a", "p", 1)])
        assert out[0].rerank_score is not None

    def test_top_k_truncates(self):
        hits = [_hit(f"r{i}", "p", i) for i in range(10)]
        assert len(rerank(IdentityReranker(), "q", hits, top_k=3)) == 3

    def test_empty(self):
        assert rerank(IdentityReranker(), "q", []) == []


class TestTimer:
    def test_records_stages(self):
        t = Timer()
        with t("a"):
            pass
        with t("b"):
            pass
        assert set(t.stages) == {"a", "b"}
        assert t.total_ms >= 0

import pytest

from patsearch.reranking.service import rerank
from patsearch.search.query import Hit
from patsearch.training.features import FEATURE_NAMES, build_rows, extract_features
from patsearch.training.ltr import LTRModel, LTRReranker, split_by_patent, train


def _hit(rid="r1", pid="P1", rt="claim", score=0.5, bm25=None, vec=None,
         text="a wheel spoke connected to the rim", claim_number=1, title="SPOKE"):
    return Hit(record_id=rid, patent_id=pid, record_type=rt, title=title, text=text,
               classification_raw="B60B104FI", score=score,
               claim_number=claim_number, bm25_rank=bm25, vector_rank=vec)


class TestExtractFeatures:
    def test_vector_length_matches_contract(self):
        assert len(extract_features("wheel", _hit())) == len(FEATURE_NAMES)

    def test_all_features_finite(self):
        import math
        for f in extract_features("wheel spoke", _hit(bm25=1, vec=1)):
            assert math.isfinite(f)

    def test_missing_rank_is_zero_not_imputed(self):
        f = dict(zip(FEATURE_NAMES, extract_features("wheel", _hit(bm25=None, vec=3)),
                     strict=True))
        assert f["bm25_rr"] == 0.0
        assert f["vec_rr"] > 0.0

    def test_better_rank_scores_higher(self):
        top = dict(zip(FEATURE_NAMES, extract_features("w", _hit(bm25=1)), strict=True))
        low = dict(zip(FEATURE_NAMES, extract_features("w", _hit(bm25=40)), strict=True))
        assert top["bm25_rr"] > low["bm25_rr"]

    def test_agreement_flags_are_mutually_exclusive(self):
        both = dict(zip(FEATURE_NAMES, extract_features("w", _hit(bm25=1, vec=2)), strict=True))
        assert both["found_by_both"] == 1.0
        assert both["bm25_only"] == 0.0 and both["vec_only"] == 0.0

        lex = dict(zip(FEATURE_NAMES, extract_features("w", _hit(bm25=1, vec=None)), strict=True))
        assert lex["bm25_only"] == 1.0 and lex["found_by_both"] == 0.0

    def test_record_type_one_hot(self):
        f = dict(zip(FEATURE_NAMES, extract_features("w", _hit(rt="abstract")), strict=True))
        assert f["is_abstract"] == 1.0 and f["is_claim"] == 0.0

    def test_term_overlap_responds_to_shared_words(self):
        hi = dict(zip(FEATURE_NAMES,
                      extract_features("wheel spoke rim", _hit(text="wheel spoke rim hub")),
                      strict=True))
        lo = dict(zip(FEATURE_NAMES,
                      extract_features("wheel spoke rim", _hit(text="rubber tread compound")),
                      strict=True))
        assert hi["term_overlap"] > lo["term_overlap"]

    def test_empty_query_does_not_divide_by_zero(self):
        assert len(extract_features("", _hit())) == len(FEATURE_NAMES)

    def test_missing_claim_number_handled(self):
        assert len(extract_features("w", _hit(rt="description", claim_number=None))) \
            == len(FEATURE_NAMES)


class TestBuildRows:
    def test_labels_come_from_qrels(self):
        rows = build_rows("q1", "wheel", [_hit("r1"), _hit("r2")], {"r1": 3})
        assert rows[0].label == 3
        assert rows[1].label == 0

    def test_carries_ids_for_grouping(self):
        rows = build_rows("q1", "wheel", [_hit("r1", pid="P9")], {})
        assert rows[0].query_id == "q1" and rows[0].patent_id == "P9"

    def test_empty_hits(self):
        assert build_rows("q1", "wheel", [], {}) == []


class TestSplitByPatent:
    def test_partitions_are_disjoint(self):
        tr, te = split_by_patent([f"P{i}" for i in range(20)])
        assert not (tr & te)

    def test_covers_every_patent(self):
        pats = [f"P{i}" for i in range(20)]
        tr, te = split_by_patent(pats)
        assert tr | te == set(pats)

    def test_deterministic_for_a_seed(self):
        pats = [f"P{i}" for i in range(30)]
        assert split_by_patent(pats, seed=5) == split_by_patent(pats, seed=5)

    def test_train_never_empty(self):
        tr, _ = split_by_patent(["P1"], test_fraction=0.9)
        assert len(tr) >= 1

    def test_duplicate_ids_collapse(self):
        tr, te = split_by_patent(["P1"] * 50 + ["P2"] * 50)
        assert len(tr | te) == 2


class TestTrain:
    @pytest.fixture
    def rows(self):
        """Synthetic data where relevance tracks the vector rank."""
        out = []
        for p in range(24):
            for r in range(10):
                out.extend(build_rows(
                    f"q{p}", "wheel spoke",
                    [_hit(f"{p}:{r}", pid=f"P{p}", vec=r + 1, bm25=r + 1)],
                    {f"{p}:{r}": 3 if r < 2 else (1 if r < 5 else 0)},
                ))
        return out

    def test_trains_and_reports(self, rows):
        model, report, test_rows = train(rows, test_fraction=0.3, seed=1)
        assert report.n_train_rows > 0
        assert report.n_test_rows > 0
        assert set(report.feature_importance) == set(FEATURE_NAMES)

    def test_train_and_test_patents_are_disjoint(self, rows):
        _, report, test_rows = train(rows, seed=1)
        assert report.n_train_patents > 0 and report.n_test_patents > 0
        # No patent may appear on both sides.
        assert report.n_train_patents + report.n_test_patents == len({r.patent_id for r in rows})

    def test_learns_the_signal(self, rows):
        """A candidate at vector rank 1 must score above one at rank 10."""
        model, _, _ = train(rows, seed=1)
        good = model.score_hits("wheel spoke", [_hit(vec=1, bm25=1)])[0]
        bad = model.score_hits("wheel spoke", [_hit(vec=10, bm25=10)])[0]
        assert good > bad

    def test_empty_rows_raises(self):
        with pytest.raises(ValueError, match="no training rows"):
            train([])

    def test_roundtrip_through_disk(self, rows, tmp_path):
        model, _, _ = train(rows, seed=1)
        p = tmp_path / "m.pkl"
        model.save(p)
        loaded = LTRModel.load(p)
        h = [_hit(vec=1, bm25=1)]
        assert loaded.score_hits("wheel", h) == pytest.approx(model.score_hits("wheel", h))

    def test_feature_contract_mismatch_is_caught(self, rows, tmp_path):
        """A model trained on a different feature set must refuse to load."""
        import pickle
        model, _, _ = train(rows, seed=1)
        p = tmp_path / "m.pkl"
        with p.open("wb") as fh:
            pickle.dump({"estimator": model.estimator, "features": ("only", "two")}, fh)
        with pytest.raises(ValueError, match="retrain"):
            LTRModel.load(p)


class TestLTRReranker:
    @pytest.fixture
    def reranker(self):
        rows = []
        for p in range(20):
            for r in range(8):
                rows.extend(build_rows(
                    f"q{p}", "wheel", [_hit(f"{p}:{r}", pid=f"P{p}", vec=r + 1)],
                    {f"{p}:{r}": 3 if r == 0 else 0},
                ))
        model, _, _ = train(rows, seed=2)
        return LTRReranker(model)

    def test_rerank_preserves_the_candidate_set(self, reranker):
        hits = [_hit(f"r{i}", vec=i + 1) for i in range(10)]
        out = rerank(reranker, "wheel", hits)
        assert {h.record_id for h in out} == {f"r{i}" for i in range(10)}

    def test_rerank_sets_scores(self, reranker):
        out = rerank(reranker, "wheel", [_hit("r1", vec=1)])
        assert out[0].rerank_score is not None

    def test_rerank_reorders_toward_better_ranks(self, reranker):
        hits = [_hit("worst", vec=20), _hit("best", vec=1)]
        assert rerank(reranker, "wheel", hits)[0].record_id == "best"

    def test_score_on_raw_text_is_refused(self, reranker):
        with pytest.raises(NotImplementedError, match="score_hits"):
            reranker.score("wheel", ["some text"])

    def test_empty_hits(self, reranker):
        assert rerank(reranker, "wheel", []) == []

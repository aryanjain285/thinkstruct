import math

import pytest

from patsearch.evaluation.metrics import (
    aggregate,
    average_precision,
    dcg_at_k,
    evaluate_one,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

QRELS = {"a": 3, "b": 2, "c": 1}          # three relevant docs, graded
PERFECT = ["a", "b", "c", "x", "y"]
REVERSED = ["c", "b", "a", "x", "y"]
NONE_REL = ["x", "y", "z"]


class TestRecall:
    def test_all_found(self):
        assert recall_at_k(PERFECT, QRELS, 5) == 1.0

    def test_partial(self):
        assert recall_at_k(PERFECT, QRELS, 2) == pytest.approx(2 / 3)

    def test_none_found(self):
        assert recall_at_k(NONE_REL, QRELS, 3) == 0.0

    def test_no_relevant_docs_is_zero_not_error(self):
        assert recall_at_k(PERFECT, {}, 5) == 0.0

    def test_threshold_excludes_weak_grades(self):
        # only 'a'(3) and 'b'(2) count at threshold 2
        assert recall_at_k(PERFECT, QRELS, 5, threshold=2) == 1.0
        assert recall_at_k(["a"], QRELS, 5, threshold=2) == pytest.approx(0.5)


class TestPrecision:
    def test_full(self):
        assert precision_at_k(PERFECT, QRELS, 3) == 1.0

    def test_half(self):
        assert precision_at_k(["a", "x"], QRELS, 2) == 0.5

    def test_k_zero(self):
        assert precision_at_k(PERFECT, QRELS, 0) == 0.0

    def test_k_beyond_list_length_divides_by_k(self):
        assert precision_at_k(["a"], QRELS, 5) == pytest.approx(1 / 5)


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(PERFECT, QRELS, 10) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], QRELS, 10) == pytest.approx(1 / 3)

    def test_outside_cutoff_is_zero(self):
        assert reciprocal_rank(["x", "y", "a"], QRELS, 2) == 0.0

    def test_none_relevant(self):
        assert reciprocal_rank(NONE_REL, QRELS, 10) == 0.0


class TestDCGandNDCG:
    def test_dcg_hand_computed(self):
        # (2^3-1)/log2(2) + (2^2-1)/log2(3) = 7/1 + 3/1.58496 = 8.8928
        got = dcg_at_k(["a", "b"], QRELS, 2)
        assert got == pytest.approx(7 + 3 / math.log2(3), abs=1e-6)

    def test_ndcg_perfect_ordering_is_one(self):
        assert ndcg_at_k(PERFECT, QRELS, 5) == pytest.approx(1.0)

    def test_ndcg_reversed_is_less_than_one(self):
        assert ndcg_at_k(REVERSED, QRELS, 5) < 1.0

    def test_ndcg_rewards_putting_best_first(self):
        assert ndcg_at_k(PERFECT, QRELS, 5) > ndcg_at_k(REVERSED, QRELS, 5)

    def test_ndcg_no_relevant_is_zero(self):
        assert ndcg_at_k(PERFECT, {}, 5) == 0.0

    def test_ndcg_bounded(self):
        for ranked in (PERFECT, REVERSED, NONE_REL, []):
            assert 0.0 <= ndcg_at_k(ranked, QRELS, 5) <= 1.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision(PERFECT, QRELS, 5) == pytest.approx(1.0)

    def test_hand_computed(self):
        # relevant at positions 1 and 3 -> (1/1 + 2/3)/3
        got = average_precision(["a", "x", "b", "y"], QRELS, 4)
        assert got == pytest.approx((1.0 + 2 / 3) / 3)

    def test_none_relevant(self):
        assert average_precision(NONE_REL, QRELS, 3) == 0.0

    def test_empty_qrels(self):
        assert average_precision(PERFECT, {}, 5) == 0.0


class TestEvaluateOne:
    def test_returns_all_metrics_at_all_ks(self):
        out = evaluate_one(PERFECT, QRELS, ks=(5, 10))
        for m in ("recall", "precision", "ndcg", "mrr", "map"):
            assert f"{m}@5" in out and f"{m}@10" in out

    def test_empty_ranking_scores_zero(self):
        out = evaluate_one([], QRELS, ks=(10,))
        # recall_ceiling is a property of the qrels, not the ranking, so it stays > 0.
        scored = {k: v for k, v in out.items() if not k.startswith("recall_ceiling")}
        assert all(v == 0.0 for v in scored.values())
        assert out["recall_ceiling@10"] > 0.0


class TestAggregate:
    def test_macro_average(self):
        agg = aggregate([{"ndcg@10": 1.0}, {"ndcg@10": 0.0}])
        assert agg["ndcg@10"] == 0.5

    def test_empty(self):
        assert aggregate([]) == {}

    def test_missing_key_treated_as_zero(self):
        assert aggregate([{"a": 1.0}, {}])["a"] == 0.5

import pytest

from patsearch.evaluation.significance import (
    paired_bootstrap,
    paired_t_test,
    to_trec_run,
)


class TestPairedBootstrap:
    def test_identical_vectors_are_not_significant(self):
        a = [0.5] * 40
        assert paired_bootstrap(a, list(a), iterations=2000) == 1.0

    def test_large_consistent_difference_is_significant(self):
        a = [0.9] * 50
        b = [0.1] * 50
        assert paired_bootstrap(a, b, iterations=2000) < 0.05

    def test_noise_is_not_significant(self):
        # alternating +/- differences average to ~0
        a = [0.5 + (0.1 if i % 2 else -0.1) for i in range(60)]
        b = [0.5] * 60
        assert paired_bootstrap(a, b, iterations=2000) > 0.05

    def test_p_value_bounded(self):
        a = [1.0] * 30
        b = [0.0] * 30
        p = paired_bootstrap(a, b, iterations=1000)
        assert 0.0 < p <= 1.0

    def test_deterministic_for_a_seed(self):
        a = [0.4, 0.6, 0.8, 0.2] * 10
        b = [0.3, 0.7, 0.5, 0.4] * 10
        assert paired_bootstrap(a, b, iterations=1000, seed=1) == paired_bootstrap(
            a, b, iterations=1000, seed=1
        )

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_bootstrap([1.0], [1.0, 2.0])

    def test_empty(self):
        assert paired_bootstrap([], []) == 1.0


class TestPairedTTest:
    def test_identical(self):
        assert paired_t_test([0.5] * 20, [0.5] * 20) == 1.0

    def test_large_difference_significant(self):
        assert paired_t_test([0.9] * 40, [0.1] * 40) < 0.05

    def test_agrees_with_bootstrap_on_clear_cases(self):
        a = [0.8] * 50
        b = [0.2] * 50
        assert paired_t_test(a, b) < 0.05
        assert paired_bootstrap(a, b, iterations=2000) < 0.05

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_t_test([1.0], [1.0, 2.0])

    def test_too_few_samples(self):
        assert paired_t_test([1.0], [0.0]) == 1.0


class TestTrecRun:
    def test_format(self):
        out = to_trec_run(["q1"], [["d1", "d2"]], "hybrid")
        lines = out.splitlines()
        assert lines[0].split()[:4] == ["q1", "Q0", "d1", "1"]
        assert lines[1].split()[:4] == ["q1", "Q0", "d2", "2"]
        assert lines[0].split()[-1] == "hybrid"

    def test_empty(self):
        assert to_trec_run([], [], "x") == ""

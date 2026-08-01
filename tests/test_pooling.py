

from patsearch.evaluation.evaluator import EvalQuery, EvalSet
from patsearch.evaluation.pooling import (
    JudgementCache,
    apply_to_eval_set,
    build_pool,
    judge_pool,
    merge_qrels,
    pool_stats,
)


def _q(qid="q1", text="wheel spoke", pid="P1", qrels=None):
    return EvalQuery(query_id=qid, text=text, source_patent_id=pid,
                     query_type="test", qrels=qrels or {})


class FakeAssessor:
    """Deterministic stand-in — no network. Grades by passage length."""

    batch_size = 10

    def __init__(self):
        self.calls = 0

    def judge(self, query, passages):
        self.calls += 1
        return [min(3, len(p) // 5) for p in passages]


class TestBuildPool:
    def test_unions_across_systems(self):
        results = {
            "bm25": [("r1", "P1", "a"), ("r2", "P2", "b")],
            "dense": [("r2", "P2", "b"), ("r3", "P3", "c")],
        }
        pool = build_pool([_q()], lambda s, t: results[s], ["bm25", "dense"])
        assert set(pool["q1"]) == {"r1", "r2", "r3"}

    def test_records_which_systems_found_each(self):
        results = {
            "bm25": [("r1", "P1", "a")],
            "dense": [("r1", "P1", "a")],
        }
        pool = build_pool([_q()], lambda s, t: results[s], ["bm25", "dense"])
        assert sorted(pool["q1"]["r1"].found_by) == ["bm25", "dense"]

    def test_keeps_best_rank(self):
        results = {
            "bm25": [("x", "P", "t"), ("r1", "P1", "a")],   # r1 at rank 2
            "dense": [("r1", "P1", "a")],                    # r1 at rank 1
        }
        pool = build_pool([_q()], lambda s, t: results[s], ["bm25", "dense"])
        assert pool["q1"]["r1"].best_rank == 1

    def test_depth_truncates(self):
        many = [(f"r{i}", "P", "t") for i in range(50)]
        pool = build_pool([_q()], lambda s, t: many, ["bm25"], depth=5)
        assert len(pool["q1"]) == 5

    def test_multiple_queries_isolated(self):
        pool = build_pool(
            [_q("q1"), _q("q2")],
            lambda s, t: [("r1", "P", "a")],
            ["bm25"],
        )
        assert set(pool) == {"q1", "q2"}

    def test_empty_results(self):
        pool = build_pool([_q()], lambda s, t: [], ["bm25"])
        assert pool["q1"] == {}


class TestJudgementCache:
    def test_roundtrip(self, tmp_path):
        c = JudgementCache(tmp_path / "j.jsonl")
        c.put("q1", "r1", 3)
        assert c.get("q1", "r1") == 3
        assert c.get("q1", "missing") is None

    def test_persists_across_instances(self, tmp_path):
        p = tmp_path / "j.jsonl"
        JudgementCache(p).put("q1", "r1", 2)
        assert JudgementCache(p).get("q1", "r1") == 2

    def test_does_not_duplicate(self, tmp_path):
        p = tmp_path / "j.jsonl"
        c = JudgementCache(p)
        c.put("q1", "r1", 2)
        c.put("q1", "r1", 3)          # ignored, already judged
        assert len(p.read_text(encoding="utf-8").strip().splitlines()) == 1
        assert c.get("q1", "r1") == 2

    def test_missing_file_is_empty(self, tmp_path):
        assert len(JudgementCache(tmp_path / "nope.jsonl")) == 0

    def test_len(self, tmp_path):
        c = JudgementCache(tmp_path / "j.jsonl")
        c.put("q1", "r1", 1)
        c.put("q1", "r2", 2)
        assert len(c) == 2


class TestJudgePool:
    def test_judges_everything_in_the_pool(self, tmp_path):
        pool = build_pool([_q()], lambda s, t: [("r1", "P", "aaaaaaaaaa"), ("r2", "P", "bb")],
                          ["bm25"])
        qrels = judge_pool([_q()], pool, FakeAssessor(), JudgementCache(tmp_path / "j.jsonl"),
                           progress=False)
        assert set(qrels["q1"]) == {"r1", "r2"}
        assert qrels["q1"]["r1"] == 2      # len 10 // 5
        assert qrels["q1"]["r2"] == 0      # len 2 // 5

    def test_cache_prevents_repeat_calls(self, tmp_path):
        cache = JudgementCache(tmp_path / "j.jsonl")
        pool = build_pool([_q()], lambda s, t: [("r1", "P", "aaaaa")], ["bm25"])
        a1 = FakeAssessor()
        judge_pool([_q()], pool, a1, cache, progress=False)
        a2 = FakeAssessor()
        judge_pool([_q()], pool, a2, cache, progress=False)
        assert a1.calls == 1
        assert a2.calls == 0               # fully cached second time

    def test_empty_pool(self, tmp_path):
        qrels = judge_pool([_q()], {"q1": {}}, FakeAssessor(),
                           JudgementCache(tmp_path / "j.jsonl"), progress=False)
        assert qrels["q1"] == {}


class TestMergeQrels:
    def test_judged_wins_by_default(self):
        assert merge_qrels({"r1": 3}, {"r1": 1})["r1"] == 1

    def test_structural_kept_when_preferred(self):
        assert merge_qrels({"r1": 3}, {"r1": 1}, prefer_judged=False)["r1"] == 3

    def test_union_of_both(self):
        m = merge_qrels({"r1": 3}, {"r2": 2})
        assert m == {"r1": 3, "r2": 2}

    def test_zero_grades_dropped(self):
        """Non-relevant judgements must not inflate the relevant set."""
        assert merge_qrels({}, {"r1": 0, "r2": 2}) == {"r2": 2}

    def test_structural_zero_also_dropped(self):
        assert "r1" not in merge_qrels({"r1": 0}, {})


class TestApplyToEvalSet:
    def test_adds_cross_patent_judgements(self):
        es = EvalSet([_q("q1", qrels={"P1:claim:1": 3})])
        out = apply_to_eval_set(es, {"q1": {"P2:claim:5": 2}})
        assert out.queries[0].qrels == {"P1:claim:1": 3, "P2:claim:5": 2}

    def test_drops_queries_with_no_relevant_docs(self):
        es = EvalSet([_q("q1", qrels={})])
        assert apply_to_eval_set(es, {"q1": {"r": 0}}).queries == []

    def test_notes_record_the_method(self):
        out = apply_to_eval_set(EvalSet([_q("q1", qrels={"a": 1})]), {"q1": {}})
        assert "pooling" in out.notes["generation"].lower()
        assert "unjudged_policy" in out.notes

    def test_can_discard_structural_labels(self):
        es = EvalSet([_q("q1", qrels={"P1:claim:1": 3})])
        out = apply_to_eval_set(es, {"q1": {"P2:claim:5": 2}}, keep_structural=False)
        assert out.queries[0].qrels == {"P2:claim:5": 2}


class TestPoolStats:
    def test_shape(self):
        pool = build_pool(
            [_q("q1"), _q("q2")],
            lambda s, t: [("r1", "P", "a"), ("r2", "P", "b")],
            ["bm25", "dense"],
        )
        s = pool_stats(pool)
        assert s["queries"] == 2
        assert s["total_candidates"] == 4
        assert s["mean_pool_size"] == 2.0
        assert s["mean_found_by_multiple_systems"] == 2.0

    def test_empty(self):
        assert pool_stats({})["queries"] == 0

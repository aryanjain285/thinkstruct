import pytest

from patsearch.ingestion.status_store import JobStatus, StatusStore, content_hash


@pytest.fixture
def store():
    s = StatusStore(":memory:")
    yield s
    s.close()


class TestEnqueue:
    def test_new_patent_is_queued(self, store):
        assert store.enqueue("p1", "f.json", "h1") is True
        assert store.get("p1").status is JobStatus.PENDING

    def test_reenqueue_unchanged_completed_is_noop(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.complete("p1", record_count=12)
        assert store.enqueue("p1", "f.json", "h1") is False
        assert store.get("p1").status is JobStatus.COMPLETED
        assert store.get("p1").record_count == 12

    def test_changed_content_requeues(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.complete("p1", record_count=5)
        assert store.enqueue("p1", "f.json", "h2") is True
        assert store.get("p1").status is JobStatus.PENDING

    def test_version_bump_requeues(self, store):
        store.enqueue("p1", "f.json", "h1", parser_version="v1")
        store.complete("p1", record_count=5)
        assert store.enqueue("p1", "f.json", "h1", parser_version="v2") is True

    def test_embedding_model_change_requeues(self, store):
        store.enqueue("p1", "f.json", "h1", embedding_model="a")
        store.complete("p1", record_count=1)
        assert store.enqueue("p1", "f.json", "h1", embedding_model="b") is True

    def test_incomplete_job_always_requeues(self, store):
        store.enqueue("p1", "f.json", "h1")
        assert store.enqueue("p1", "f.json", "h1") is True


class TestClaim:
    def test_claim_moves_to_in_flight(self, store):
        store.enqueue("p1", "f.json", "h1")
        jobs = store.claim("worker-a")
        assert [j.patent_id for j in jobs] == ["p1"]
        assert store.get("p1").status is JobStatus.VALIDATING

    def test_claim_is_exclusive(self, store):
        store.enqueue("p1", "f.json", "h1")
        assert len(store.claim("worker-a")) == 1
        assert store.claim("worker-b") == []          # already taken

    def test_claim_respects_limit(self, store):
        for i in range(5):
            store.enqueue(f"p{i}", "f.json", f"h{i}")
        assert len(store.claim("w", limit=3)) == 3
        assert len(store.claim("w", limit=10)) == 2

    def test_claim_skips_exhausted_retries(self, store):
        store.enqueue("p1", "f.json", "h1")
        for _ in range(3):
            store.claim("w")
            store.fail("p1", "boom", max_retries=3)
        assert store.get("p1").status is JobStatus.FAILED
        assert store.claim("w") == []

    def test_empty_queue(self, store):
        assert store.claim("w") == []


class TestTransitions:
    def test_advance_through_stages(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("w")
        for s in (JobStatus.RECONSTRUCTING, JobStatus.EMBEDDING, JobStatus.INDEXING):
            store.advance("p1", s)
            assert store.get("p1").status is s

    def test_complete_clears_error_and_lease(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("w")
        store.fail("p1", "transient")
        store.claim("w")
        store.complete("p1", record_count=7)
        j = store.get("p1")
        assert j.status is JobStatus.COMPLETED
        assert j.last_error is None
        assert j.record_count == 7

    def test_fail_retries_then_gives_up(self, store):
        store.enqueue("p1", "f.json", "h1")
        assert store.fail("p1", "e1", max_retries=3) is JobStatus.PENDING
        assert store.fail("p1", "e2", max_retries=3) is JobStatus.PENDING
        assert store.fail("p1", "e3", max_retries=3) is JobStatus.FAILED
        assert store.get("p1").retry_count == 3

    def test_fail_records_error_text(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.fail("p1", "ValueError: bad claim")
        assert "bad claim" in store.get("p1").last_error

    def test_long_error_truncated(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.fail("p1", "x" * 5000)
        assert len(store.get("p1").last_error) <= 2000


class TestCrashRecovery:
    def test_stale_lease_returns_to_pending(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("dead-worker")
        assert store.reclaim_stale(lease_seconds=0) == 1
        assert store.get("p1").status is JobStatus.PENDING

    def test_fresh_lease_is_not_reclaimed(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("live-worker")
        assert store.reclaim_stale(lease_seconds=300) == 0
        assert store.get("p1").status is JobStatus.VALIDATING

    def test_completed_jobs_never_reclaimed(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("w")
        store.complete("p1", record_count=1)
        assert store.reclaim_stale(lease_seconds=0) == 0

    def test_reclaimed_job_is_claimable_again(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.claim("dead")
        store.reclaim_stale(lease_seconds=0)
        assert len(store.claim("alive")) == 1


class TestReporting:
    def test_counts_by_status(self, store):
        for i in range(4):
            store.enqueue(f"p{i}", "f.json", f"h{i}")
        store.claim("w", limit=2)
        c = store.counts()
        assert c[JobStatus.PENDING.value] == 2
        assert c[JobStatus.VALIDATING.value] == 2

    def test_summary_shape(self, store):
        store.enqueue("p1", "f.json", "h1")
        store.complete("p1", record_count=10)
        store.enqueue("p2", "f.json", "h2")
        s = store.summary()
        assert s["total"] == 2
        assert s["completed_pct"] == 50.0
        assert s["records_indexed"] == 10

    def test_failures_listed(self, store):
        store.enqueue("p1", "f.json", "h1")
        for _ in range(3):
            store.fail("p1", "nope", max_retries=3)
        assert [j.patent_id for j in store.failures()] == ["p1"]

    def test_empty_summary(self, store):
        assert store.summary()["total"] == 0


class TestContentHash:
    def test_deterministic(self):
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")


def test_persists_across_connections(tmp_path):
    db = tmp_path / "jobs.db"
    with StatusStore(db) as s:
        s.enqueue("p1", "f.json", "h1")
        s.complete("p1", record_count=3)
    with StatusStore(db) as s:
        assert s.get("p1").status is JobStatus.COMPLETED
        assert s.get("p1").record_count == 3

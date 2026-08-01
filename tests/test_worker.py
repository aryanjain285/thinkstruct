import json

import pytest

from patsearch.ingestion.status_store import StatusStore
from patsearch.ingestion.worker import (
    SOURCE_CACHE_SIZE,
    IngestionWorker,
    enqueue_corpus,
)


def _rec(doc_number="20240051333"):
    return {
        "title": "SPOKE",
        "doc_number": doc_number,
        "filename": f"US{doc_number}A1.XML",
        "abstract": "A spoke includes an axle body.",
        "detailed_description": ["A real paragraph of description text here."],
        "claims": ["an axle body having a middle segment;", "2 . The spoke of claim 1 ."],
        "bibtex": "@misc{}",
        "classification": "B60B104FI",
    }


@pytest.fixture
def corpus(tmp_path):
    for w in range(6):
        recs = [_rec(f"2024005{w}{i:03d}") for i in range(3)]
        (tmp_path / f"patents_ipa24020{w}.json").write_text(
            json.dumps(recs), encoding="utf-8"
        )
    return tmp_path


class FakeOpenSearch:
    def __init__(self):
        self.indexed = []

    def indices_refresh(self, **k):
        pass

    @property
    def indices(self):
        class _I:
            def refresh(self, **k):
                pass
        return _I()


class TestEnqueueCorpus:
    def test_enqueues_every_patent(self, corpus):
        with StatusStore(":memory:") as s:
            res = enqueue_corpus(corpus, s)
            assert res["queued"] == 18
            assert res["skipped_unchanged"] == 0

    def test_second_run_after_completion_is_idempotent(self, corpus):
        with StatusStore(":memory:") as s:
            enqueue_corpus(corpus, s)
            for job in s.claim("w", limit=100):
                s.complete(job.patent_id, record_count=1)
            res = enqueue_corpus(corpus, s)
            assert res["queued"] == 0
            assert res["skipped_unchanged"] == 18

    def test_content_change_requeues_only_that_patent(self, corpus):
        with StatusStore(":memory:") as s:
            enqueue_corpus(corpus, s)
            for job in s.claim("w", limit=100):
                s.complete(job.patent_id, record_count=1)

            f = next(corpus.glob("patents_*.json"))
            recs = json.loads(f.read_text(encoding="utf-8"))
            recs[0]["abstract"] = "materially different abstract"
            f.write_text(json.dumps(recs), encoding="utf-8")

            res = enqueue_corpus(corpus, s)
            assert res["queued"] == 1
            assert res["skipped_unchanged"] == 17

    def test_parser_version_bump_requeues_all(self, corpus):
        with StatusStore(":memory:") as s:
            enqueue_corpus(corpus, s, parser_version="v1")
            for job in s.claim("w", limit=100):
                s.complete(job.patent_id, record_count=1)
            assert enqueue_corpus(corpus, s, parser_version="v2")["queued"] == 18

    def test_records_without_doc_number_skipped(self, tmp_path):
        (tmp_path / "patents_x.json").write_text(
            json.dumps([_rec(), {"title": "no id"}]), encoding="utf-8"
        )
        with StatusStore(":memory:") as s:
            assert enqueue_corpus(tmp_path, s)["queued"] == 1


class TestSourceCache:
    def test_cache_is_bounded(self, corpus):
        """Regression: the cache used to grow without limit, holding the whole
        corpus in memory for the worker's lifetime."""
        with StatusStore(":memory:") as s:
            w = IngestionWorker(s, FakeOpenSearch(), "idx", raw_dir=corpus)
            for f in sorted(corpus.glob("patents_*.json")):
                w._load_source(f.name, "nonexistent")
            assert len(w._source_cache) <= SOURCE_CACHE_SIZE

    def test_cache_returns_correct_record(self, corpus):
        with StatusStore(":memory:") as s:
            w = IngestionWorker(s, FakeOpenSearch(), "idx", raw_dir=corpus)
            f = sorted(corpus.glob("patents_*.json"))[0]
            pid = json.loads(f.read_text(encoding="utf-8"))[0]["doc_number"]
            assert w._load_source(f.name, pid)["doc_number"] == pid

    def test_missing_patent_returns_none(self, corpus):
        with StatusStore(":memory:") as s:
            w = IngestionWorker(s, FakeOpenSearch(), "idx", raw_dir=corpus)
            f = sorted(corpus.glob("patents_*.json"))[0]
            assert w._load_source(f.name, "does-not-exist") is None

    def test_recently_used_file_is_retained(self, corpus):
        with StatusStore(":memory:") as s:
            w = IngestionWorker(s, FakeOpenSearch(), "idx", raw_dir=corpus)
            files = [f.name for f in sorted(corpus.glob("patents_*.json"))]
            w._load_source(files[0], "x")
            for f in files[1:SOURCE_CACHE_SIZE]:
                w._load_source(f, "x")
            w._load_source(files[0], "x")          # refresh recency
            w._load_source(files[SOURCE_CACHE_SIZE], "x")  # forces an eviction
            assert files[0] in w._source_cache

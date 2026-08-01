import pytest

from patsearch.config import RAW_DIR
from patsearch.ingestion.loader import load_all
from patsearch.models import Patent, RecordType
from patsearch.processing.reconstruct import reconstruct_claims
from patsearch.processing.records import build_records, chunk_paragraphs


def _patent(**over):
    base = {
        "patent_id": "P1", "title": "SPOKE", "abstract": "A spoke includes an axle body.",
        "classification_raw": "B60B104FI", "claims_raw": [], "description_paragraphs": [],
        "bibtex": "", "source_file": "f.json",
    }
    base.update(over)
    return Patent(**base)


class TestChunkParagraphs:
    def test_empty(self):
        assert list(chunk_paragraphs([])) == []

    def test_single_short_paragraph(self):
        out = list(chunk_paragraphs(["hello world"]))
        assert out == [(0, 0, "hello world")]

    def test_groups_until_target(self):
        paras = ["word " * 100] * 5
        out = list(chunk_paragraphs(paras, target_words=200, max_words=400, overlap=0))
        assert len(out) == 3           # 100-word paras -> 2 per chunk at target 200
        assert out[0][0] == 0

    def test_oversized_paragraph_emitted_alone_not_split(self):
        paras = ["w " * 900, "short one"]
        out = list(chunk_paragraphs(paras, target_words=100, max_words=200, overlap=0))
        assert out[0][0] == out[0][1] == 0
        assert len(out[0][2].split()) == 900

    def test_overlap_applied(self):
        paras = [f"para{i} " * 60 for i in range(6)]
        out = list(chunk_paragraphs(paras, target_words=60, max_words=120, overlap=1))
        starts = [s for s, _, _ in out]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)     # always progresses

    def test_always_terminates_with_large_overlap(self):
        paras = [f"p{i} " * 50 for i in range(20)]
        out = list(chunk_paragraphs(paras, target_words=50, max_words=60, overlap=99))
        assert len(out) <= len(paras)              # no infinite loop

    def test_indexes_cover_input(self):
        paras = [f"p{i} " * 40 for i in range(10)]
        out = list(chunk_paragraphs(paras, target_words=80, max_words=160, overlap=0))
        assert out[0][0] == 0
        assert out[-1][1] == len(paras) - 1


class TestBuildRecords:
    def test_summary_and_abstract_emitted(self):
        recs = build_records(_patent(), [])
        types = {r.record_type for r in recs}
        assert RecordType.SUMMARY in types
        assert RecordType.ABSTRACT in types

    def test_summary_combines_title_and_abstract(self):
        rec = next(r for r in build_records(_patent(), []) if r.record_type is RecordType.SUMMARY)
        assert "SPOKE" in rec.text and "axle body" in rec.text

    def test_claim_records_carry_metadata(self):
        p = _patent(claims_raw=["a body;", "2 . The spoke of claim 1 ."])
        claims = reconstruct_claims(p.patent_id, p.claims_raw)
        recs = [r for r in build_records(p, claims) if r.record_type is RecordType.CLAIM]
        assert len(recs) == 2
        assert recs[0].claim_number == 1 and recs[0].is_independent is True
        assert recs[1].claim_number == 2 and recs[1].is_independent is False

    def test_canceled_claims_not_indexed(self):
        p = _patent(claims_raw=["1 - 5 . (canceled)", "a tread portion."])
        claims = reconstruct_claims(p.patent_id, p.claims_raw)
        recs = [r for r in build_records(p, claims) if r.record_type is RecordType.CLAIM]
        assert len(recs) == 1
        assert "canceled" not in recs[0].text.lower()

    def test_blank_description_yields_no_description_records(self):
        p = _patent(description_paragraphs=["", "   ", " "])
        recs = build_records(p, [])
        assert not [r for r in recs if r.record_type is RecordType.DESCRIPTION]

    def test_classification_fields_populated(self):
        rec = build_records(_patent(), [])[0]
        assert rec.classification_subclass == "B60B"
        assert rec.classification_class == "B60"
        assert rec.classification_section == "B"

    def test_record_ids_unique_within_patent(self):
        p = _patent(
            claims_raw=["a body;", "2 . The x of claim 1 ."],
            description_paragraphs=[f"para {i} " * 80 for i in range(6)],
        )
        recs = build_records(p, reconstruct_claims(p.patent_id, p.claims_raw))
        ids = [r.record_id for r in recs]
        assert len(ids) == len(set(ids))

    def test_no_empty_text_records(self):
        p = _patent(description_paragraphs=["", "real content here", ""])
        assert all(r.text.strip() for r in build_records(p, []))

    def test_to_dict_drops_none_fields(self):
        d = build_records(_patent(), [])[0].to_dict()
        assert "claim_number" not in d
        assert d["record_type"] == "summary"


@pytest.mark.skipif(not RAW_DIR.exists(), reason="corpus not extracted")
class TestAgainstRealCorpus:
    @pytest.fixture(scope="class")
    def records(self):
        pats, _ = load_all(RAW_DIR)
        out = []
        for p in pats:
            out.extend(build_records(p, reconstruct_claims(p.patent_id, p.claims_raw)))
        return out

    def test_record_ids_globally_unique(self, records):
        ids = [r.record_id for r in records]
        assert len(ids) == len(set(ids))

    def test_every_record_has_text_and_patent(self, records):
        for r in records:
            assert r.text.strip() and r.patent_id

    def test_every_patent_represented(self, records):
        assert len({r.patent_id for r in records}) == 640

    def test_claim_records_match_reconstruction(self, records):
        n = sum(1 for r in records if r.record_type is RecordType.CLAIM)
        assert n == 10473 - 167      # all claims minus canceled markers

    def test_all_record_types_present(self, records):
        types = {r.record_type for r in records}
        assert types == set(RecordType)

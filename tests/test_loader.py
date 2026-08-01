import json

import pytest

from patsearch.config import RAW_DIR
from patsearch.ingestion.loader import (
    PatentLoadError,
    iter_patents,
    load_all,
    quality_report,
    validate_record,
)
from patsearch.models import Patent


def _rec(**over):
    base = {
        "title": "SPOKE",
        "doc_number": "20240051333",
        "filename": "US20240051333A1-20240215.XML",
        "abstract": "A spoke includes an axle body.",
        "detailed_description": ["Some real paragraph text here."],
        "claims": ["an axle body having a middle segment;", "2 . The spoke of claim 1 ."],
        "bibtex": "@misc{}",
        "classification": "B60B104FI",
    }
    base.update(over)
    return base


class TestValidateRecord:
    def test_clean_record_has_no_issues(self):
        assert validate_record(_rec(), "f.json") == []

    def test_missing_field_is_error(self):
        rec = _rec()
        del rec["claims"]
        issues = validate_record(rec, "f.json")
        assert any(i.field == "claims" and i.severity == "error" for i in issues)

    def test_blank_description_is_warning_not_error(self):
        # 119/640 patents in the real corpus look like this. They stay searchable.
        issues = validate_record(_rec(detailed_description=["", "  ", " "]), "f.json")
        desc = [i for i in issues if i.field == "detailed_description"]
        assert len(desc) == 1
        assert desc[0].severity == "warning"
        assert not any(i.severity == "error" for i in issues)

    def test_no_claims_is_error(self):
        issues = validate_record(_rec(claims=[]), "f.json")
        assert any(i.field == "claims" and i.severity == "error" for i in issues)

    def test_all_blank_claims_is_error(self):
        issues = validate_record(_rec(claims=["", "   "]), "f.json")
        assert any(i.field == "claims" and i.severity == "error" for i in issues)

    def test_blank_title_is_warning(self):
        issues = validate_record(_rec(title="  "), "f.json")
        assert any(i.field == "title" and i.severity == "warning" for i in issues)


class TestIterPatents:
    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(PatentLoadError, match="no patents_"):
            list(iter_patents(tmp_path))

    def test_raises_on_bad_json(self, tmp_path):
        (tmp_path / "patents_x.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(PatentLoadError):
            list(iter_patents(tmp_path))

    def test_raises_when_payload_not_list(self, tmp_path):
        (tmp_path / "patents_x.json").write_text('{"a": 1}', encoding="utf-8")
        with pytest.raises(PatentLoadError, match="expected a list"):
            list(iter_patents(tmp_path))

    def test_duplicate_doc_number_flagged_and_skipped(self, tmp_path):
        (tmp_path / "patents_a.json").write_text(json.dumps([_rec(), _rec()]), encoding="utf-8")
        issues = []
        pats = list(iter_patents(tmp_path, collect_issues=issues))
        assert len(pats) == 1
        assert any(i.issue == "duplicate across corpus" for i in issues)

    def test_skip_errors_false_yields_everything(self, tmp_path):
        (tmp_path / "patents_a.json").write_text(json.dumps([_rec(claims=[])]), encoding="utf-8")
        assert len(list(iter_patents(tmp_path, skip_errors=False))) == 1
        assert len(list(iter_patents(tmp_path, skip_errors=True))) == 0

    def test_non_dict_entries_ignored(self, tmp_path):
        (tmp_path / "patents_a.json").write_text(json.dumps([_rec(), "junk", 5]), encoding="utf-8")
        assert len(list(iter_patents(tmp_path))) == 1

    def test_is_lazy(self, tmp_path):
        (tmp_path / "patents_a.json").write_text(json.dumps([_rec()]), encoding="utf-8")
        gen = iter_patents(tmp_path)
        assert hasattr(gen, "__next__")


class TestClassificationParsing:
    def test_hierarchy_levels(self):
        p = Patent(
            patent_id="1", title="t", abstract="a", classification_raw="B60B1110FI",
            claims_raw=[], description_paragraphs=[], bibtex="", source_file="f",
        )
        assert p.classification_section == "B"
        assert p.classification_class == "B60"
        assert p.classification_subclass == "B60B"

    def test_handles_short_code(self):
        p = Patent(
            patent_id="1", title="t", abstract="a", classification_raw="",
            claims_raw=[], description_paragraphs=[], bibtex="", source_file="f",
        )
        assert p.classification_subclass == ""


@pytest.mark.skipif(not RAW_DIR.exists(), reason="corpus not extracted")
class TestAgainstRealCorpus:
    """Assertions pinned to measured facts about the actual dataset."""

    def test_loads_expected_volume(self):
        pats, issues = load_all(RAW_DIR)
        assert len(pats) == 640
        assert len({p.patent_id for p in pats}) == 640

    def test_no_error_severity_issues(self):
        # Measured: nothing in this corpus is unusable.
        _, issues = load_all(RAW_DIR)
        assert [i for i in issues if i.severity == "error"] == []

    def test_blank_description_count(self):
        pats, _ = load_all(RAW_DIR)
        assert sum(1 for p in pats if not p.has_description) == 119

    def test_quality_report_shape(self):
        pats, issues = load_all(RAW_DIR)
        rep = quality_report(pats, issues)
        assert rep["patents_loaded"] == 640
        assert rep["source_files"] == 64
        assert rep["patents_without_description"] == 119
        assert rep["classification_subclasses"]["B60B"] == 298
        assert rep["classification_subclasses"]["B60C"] == 318
        assert rep["claims_entries_total"] == 10578

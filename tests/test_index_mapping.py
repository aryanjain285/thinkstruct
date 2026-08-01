import pytest

from patsearch.search.index import (
    build_analysis,
    build_mapping,
    load_synonyms,
)


class TestLoadSynonyms:
    def test_reads_shipped_config(self):
        syns = load_synonyms()
        assert any("tyre" in s and "tire" in s for s in syns)

    def test_skips_comments_and_blanks(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("# a comment\n\n  \nfoo, bar\n# another\nbaz, qux\n", encoding="utf-8")
        assert load_synonyms(p) == ["foo, bar", "baz, qux"]

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_synonyms(tmp_path / "nope.txt") == []

    def test_empty_file(self, tmp_path):
        p = tmp_path / "s.txt"
        p.write_text("# only comments\n", encoding="utf-8")
        assert load_synonyms(p) == []


class TestBuildAnalysis:
    def test_includes_synonym_filter_when_given(self):
        a = build_analysis(["tyre, tire"])
        assert "patent_spelling" in a["filter"]
        assert a["filter"]["patent_spelling"]["synonyms"] == ["tyre, tire"]
        assert "patent_spelling" in a["analyzer"]["patent_english"]["filter"]

    def test_omits_synonym_filter_when_empty(self):
        """No synonym file must not produce an empty synonym_graph, which OpenSearch rejects."""
        a = build_analysis([])
        assert "patent_spelling" not in a["filter"]
        assert "patent_spelling" not in a["analyzer"]["patent_english"]["filter"]

    def test_synonyms_applied_before_stemming(self):
        chain = build_analysis(["tyre, tire"])["analyzer"]["patent_english"]["filter"]
        assert chain.index("patent_spelling") < chain.index("english_stemmer")

    def test_lowercase_first(self):
        chain = build_analysis(["tyre, tire"])["analyzer"]["patent_english"]["filter"]
        assert chain[0] == "lowercase"

    def test_stemmers_always_present(self):
        a = build_analysis([])
        for f in ("english_stemmer", "english_stop", "english_possessive_stemmer"):
            assert f in a["filter"]


class TestBuildMapping:
    def test_no_knn_without_dimension(self):
        m = build_mapping(None)
        assert "embedding" not in m["mappings"]["properties"]
        assert not m["settings"]["index"].get("knn")

    def test_knn_enabled_with_dimension(self):
        m = build_mapping(384)
        emb = m["mappings"]["properties"]["embedding"]
        assert m["settings"]["index"]["knn"] is True
        assert emb["type"] == "knn_vector"
        assert emb["dimension"] == 384

    def test_uses_lucene_engine_for_filtered_knn(self):
        """Pre-filtering during graph traversal requires the lucene engine."""
        method = build_mapping(384)["mappings"]["properties"]["embedding"]["method"]
        assert method["engine"] == "lucene"
        assert method["space_type"] == "cosinesimil"

    def test_text_fields_use_the_custom_analyzer(self):
        props = build_mapping(None)["mappings"]["properties"]
        for f in ("text", "title", "abstract"):
            assert props[f]["analyzer"] == "patent_english"

    def test_title_has_exact_keyword_subfield(self):
        props = build_mapping(None)["mappings"]["properties"]
        assert props["title"]["fields"]["exact"]["type"] == "keyword"

    def test_metadata_fields_are_keywords(self):
        props = build_mapping(None)["mappings"]["properties"]
        for f in ("patent_id", "record_id", "record_type",
                  "classification_raw", "classification_subclass"):
            assert props[f]["type"] == "keyword"

    def test_analysis_present_in_settings(self):
        assert "analysis" in build_mapping(None)["settings"]

    @pytest.mark.parametrize("dim", [128, 384, 768, 1536, 3072])
    def test_common_dimensions_accepted(self, dim):
        assert build_mapping(dim)["mappings"]["properties"]["embedding"]["dimension"] == dim

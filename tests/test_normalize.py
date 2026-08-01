from patsearch.processing.normalize import is_blank, normalize_paragraphs, normalize_text


class TestNormalizeText:
    def test_none_and_empty(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""

    def test_collapses_whitespace(self):
        assert normalize_text("a   b\n\nc\t d") == "a b c d"

    def test_nfkc_em_space(self):
        # U+2003 EM SPACE appears in formula paragraphs in this corpus.
        assert normalize_text("A-B-C   (I)") == "A-B-C (I)"

    def test_strips_invisible_format_chars(self):
        # U+2062 INVISIBLE TIMES survives NFKC; it must be removed explicitly.
        assert "⁢" not in normalize_text("Z=112⁢π⁢i")
        assert normalize_text("Z=112⁢π⁢i") == "Z=112πi"

    def test_preserves_legal_punctuation(self):
        s = "2 . The spoke according to claim 1 , wherein the surface is wavy."
        assert normalize_text(s) == s

    def test_figure_deglue(self):
        assert normalize_text("Referring toFIG.1, the wheel") == "Referring to FIG. 1, the wheel"
        assert normalize_text("as shown inFIGS.2and3") == "as shown in FIGS. 2and3"

    def test_does_not_split_alphanumeric_tokens(self):
        # These are meaningful as-is and must not be broken up.
        for token in ("B60B104FI", "PEEK", "4WD", "epdm", "E*"):
            assert token in normalize_text(f"material {token} used")

    def test_idempotent(self):
        s = "Referring toFIG.1,  wheel  has"
        assert normalize_text(normalize_text(s)) == normalize_text(s)


class TestIsBlank:
    def test_blank_variants(self):
        assert is_blank(None)
        assert is_blank("")
        assert is_blank("   \n\t ")
        assert is_blank("  ")       # em-spaces only
        assert is_blank("⁢")             # invisible-times only

    def test_non_blank(self):
        assert not is_blank("a")
        assert not is_blank("  x  ")


class TestNormalizeParagraphs:
    def test_drops_blanks_preserves_order(self):
        paras = ["first", "", "   ", "second", " ", "third"]
        assert normalize_paragraphs(paras) == ["first", "second", "third"]

    def test_empty_input(self):
        assert normalize_paragraphs(None) == []
        assert normalize_paragraphs([]) == []

    def test_all_blank_yields_empty(self):
        # 119 patents in this corpus have descriptions that are entirely blank.
        assert normalize_paragraphs(["", "  ", " "]) == []

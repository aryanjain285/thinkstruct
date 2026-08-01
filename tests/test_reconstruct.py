import pytest

from patsearch.config import RAW_DIR
from patsearch.ingestion.loader import load_all
from patsearch.models import ReconstructionStatus as RS
from patsearch.processing.reconstruct import (
    parse_dependencies,
    reconstruct_claims,
    reconstruction_stats,
)


class TestParseDependencies:
    def test_independent_claim_has_none(self):
        assert parse_dependencies("An axle body having a middle segment.") == []

    def test_according_to_claim(self):
        assert parse_dependencies("2 . The spoke according to claim 1 , wherein x.") == [1]

    def test_of_claim(self):
        assert parse_dependencies("3 . The insert of claim 2 , wherein y.") == [2]

    def test_as_recited_in(self):
        assert parse_dependencies("4 . The wheel as recited in claim 1 .") == [1]

    def test_as_set_forth_in(self):
        assert parse_dependencies("5 . The tire as set forth in claim 3 .") == [3]

    def test_range(self):
        assert parse_dependencies("The device according to any one of claims 1 to 4 .") == [1, 2, 3, 4]

    def test_range_with_dash(self):
        assert parse_dependencies("The device of claims 2-4 .") == [2, 3, 4]

    def test_list_and(self):
        assert parse_dependencies("The device according to claims 1 and 3 .") == [1, 3]

    def test_bare_word_claim_is_not_a_dependency(self):
        # No lead-in phrase -> not a dependency reference.
        assert parse_dependencies("A method of manufacturing, the claim being broad.") == []

    def test_no_crash_on_empty(self):
        assert parse_dependencies("") == []
        assert parse_dependencies(None) == []


class TestSegmentation:
    def test_index_zero_fragment_becomes_claim_one(self):
        """88.6% of real patents look like this: claim 1's preamble is gone."""
        entries = [
            "an axle body, having a middle segment; and two connecting elements.",
            "2 . The spoke according to claim 1 , wherein the surface is wavy.",
            "3 . The spoke according to claim 2 , wherein it is convex.",
        ]
        claims = reconstruct_claims("P1", entries)
        assert [c.claim_number for c in claims] == [1, 2, 3]
        assert claims[0].status is RS.PREAMBLE_STRIPPED
        assert claims[0].number_inferred is True
        assert claims[0].is_independent is True
        assert claims[0].claim_id == "P1:claim:1"

    def test_plain_numbered_claims(self):
        entries = ["1 . A spoke comprising an axle.", "2 . The spoke of claim 1 , wherein x."]
        claims = reconstruct_claims("P1", entries)
        assert [c.status for c in claims] == [RS.ORIGINAL, RS.ORIGINAL]
        assert all(not c.number_inferred for c in claims)

    def test_true_continuation_is_appended(self):
        """Previous entry ends mid-sentence (';') so the fragment continues it."""
        entries = [
            "1 . A wheel comprising: a rim;",
            "a hub connected to the rim.",
            "2 . The wheel of claim 1 .",
        ]
        claims = reconstruct_claims("P1", entries)
        assert len(claims) == 2
        assert claims[0].status is RS.REJOINED
        assert claims[0].raw_fragment_indexes == [0, 1]
        assert "a hub connected to the rim" in claims[0].text

    def test_fragment_after_completed_sentence_starts_new_claim(self):
        """The bug in the naive rule: this fragment is a NEW independent claim."""
        entries = [
            "11 . The device of claim 1 , processed through integrated molding.",
            "a wheel rim, a wheel spoke, and a tire, wherein the tire is sleeved on the rim.",
            "13 . The wheel of claim 12 .",
        ]
        claims = reconstruct_claims("P1", entries)
        assert len(claims) == 3
        assert claims[1].claim_number == 12
        assert claims[1].status is RS.PREAMBLE_STRIPPED
        assert claims[1].is_independent is True
        # Critically, it was NOT glued onto claim 11.
        assert "wheel rim" not in claims[0].text

    def test_fragment_after_canceled_marker_starts_new_claim(self):
        """Real case from 20240051338 — must not append to a '(canceled)' marker."""
        entries = [
            "1 - 5 . (canceled)",
            "a tread portion extending in a tire circumferential direction.",
            "7 . The tire of claim 6 , wherein x.",
        ]
        claims = reconstruct_claims("P1", entries)
        assert claims[0].status is RS.CANCELED
        assert claims[1].claim_number == 6
        assert claims[1].status is RS.PREAMBLE_STRIPPED
        assert "tread portion" in claims[1].text
        assert "canceled" not in claims[1].text.lower()

    def test_canceled_variants(self):
        for marker in ["1 - 5 . (canceled)", "7. (cancelled)", "3 . (Canceled)"]:
            claims = reconstruct_claims("P1", [marker])
            assert claims[0].status is RS.CANCELED
            assert claims[0].is_independent is False

    def test_blank_entries_skipped(self):
        entries = ["1 . A spoke.", "", "   ", "2 . The spoke of claim 1 ."]
        claims = reconstruct_claims("P1", entries)
        assert [c.claim_number for c in claims] == [1, 2]

    def test_empty_input(self):
        assert reconstruct_claims("P1", []) == []
        assert reconstruct_claims("P1", None) == []

    def test_numbering_gap_from_cancellation_preserved(self):
        entries = ["1 . A spoke.", "2 - 4 . (canceled)", "5 . The spoke of claim 1 ."]
        claims = reconstruct_claims("P1", entries)
        assert [c.claim_number for c in claims] == [1, 2, 5]

    def test_self_reference_stripped(self):
        claims = reconstruct_claims("P1", ["3 . The device of claim 3 ."])
        assert claims[0].depends_on == []

    def test_claim_ids_are_stable_and_unique(self):
        entries = ["a body;", "2 . The x of claim 1 .", "3 . The x of claim 2 ."]
        ids = [c.claim_id for c in reconstruct_claims("P9", entries)]
        assert ids == ["P9:claim:1", "P9:claim:2", "P9:claim:3"]
        assert len(set(ids)) == 3

    def test_raw_text_preserved_verbatim(self):
        entries = ["1 . A spoke  with   odd   spacing."]
        c = reconstruct_claims("P1", entries)[0]
        assert c.raw_text == "1 . A spoke  with   odd   spacing."
        assert c.text == "1 . A spoke with odd spacing."


class TestGeneralisesBeyondThisCorpus:
    """Formats that do NOT occur in the supplied sample. These guard against the
    parser being tuned to one dataset."""

    @pytest.mark.parametrize(
        "marker",
        [
            "1 - 5 . (canceled)",     # in corpus
            "1 .- 15 . (canceled)",   # in corpus
            "1 - 15 : (canceled)",    # in corpus
            "1-10 (cancelled)",       # not in corpus
            "1—12. (Canceled)",       # em-dash range, not in corpus
            "3, 7 (canceled)",        # comma separator, not in corpus
            "8. (CANCELED)",          # uppercase, not in corpus
        ],
    )
    def test_cancellation_marker_shapes(self, marker):
        claims = reconstruct_claims("P", [marker])
        assert claims[0].status is RS.CANCELED, marker

    def test_paren_style_claim_numbers(self):
        """'1)' instead of '1.' — used by some offices, absent from this sample."""
        entries = ["1) A wheel comprising a rim.", "2) The wheel of claim 1, wherein x."]
        claims = reconstruct_claims("P", entries)
        assert [c.claim_number for c in claims] == [1, 2]
        assert all(not c.number_inferred for c in claims)

    def test_decimal_quantities_never_read_as_claim_numbers(self):
        """The failure that produced claim 0 and a duplicate claim 5."""
        entries = [
            "1 . A rubber mixture comprising:",
            "0.5 phr to 6.0 phr of accelerator;",
            "5.0 phr to 70 phr of liquid polymer.",
        ]
        claims = reconstruct_claims("P", entries)
        assert [c.claim_number for c in claims] == [1]
        assert "accelerator" in claims[0].text and "liquid polymer" in claims[0].text

    def test_large_claim_counts(self):
        entries = [f"{i} . The device of claim 1 , variant {i}." for i in range(1, 121)]
        claims = reconstruct_claims("P", entries)
        assert len(claims) == 120
        assert claims[-1].claim_number == 120

    def test_no_numbered_entries_at_all(self):
        """Degenerate input: every entry is a bare fragment."""
        entries = ["a first element.", "a second element.", "a third element."]
        claims = reconstruct_claims("P", entries)
        assert [c.claim_number for c in claims] == [1, 2, 3]
        assert all(c.number_inferred for c in claims)

    def test_single_entry_patent(self):
        claims = reconstruct_claims("P", ["1 . A wheel."])
        assert len(claims) == 1 and claims[0].claim_number == 1

    def test_unicode_and_whitespace_noise(self):
        entries = ["1 . A wheel⁢comprising a rim."]
        assert reconstruct_claims("P", entries)[0].claim_number == 1

    def test_no_source_text_is_lost(self):
        """Core invariant: every input index is consumed exactly once."""
        entries = [
            "a body;", "with a hub.", "2 . The x of claim 1 .",
            "3 - 4 . (canceled)", "a further element.", "6 . The x of claim 2 .",
        ]
        claims = reconstruct_claims("P", entries)
        used = sorted(i for c in claims for i in c.raw_fragment_indexes)
        assert used == list(range(len(entries)))


@pytest.mark.skipif(not RAW_DIR.exists(), reason="corpus not extracted")
class TestAgainstRealCorpus:
    @pytest.fixture(scope="class")
    def all_claims(self):
        pats, _ = load_all(RAW_DIR)
        return {p.patent_id: reconstruct_claims(p.patent_id, p.claims_raw) for p in pats}

    def test_every_patent_yields_claims(self, all_claims):
        assert len(all_claims) == 640
        assert all(len(v) > 0 for v in all_claims.values())

    def test_claim_numbers_unique_within_patent(self, all_claims):
        for pid, claims in all_claims.items():
            nums = [c.claim_number for c in claims if c.status is not RS.CANCELED]
            assert len(nums) == len(set(nums)), f"{pid} has duplicate claim numbers: {nums}"

    def test_claim_ids_globally_unique(self, all_claims):
        ids = [c.claim_id for claims in all_claims.values() for c in claims]
        assert len(ids) == len(set(ids))

    def test_out_of_order_numbering_is_rare_and_reported(self, all_claims):
        """Some source patents really do label claims out of sequence. We measure it
        rather than assert it away — but it must stay a small minority."""
        flat = [c for cs in all_claims.values() for c in cs]
        n = reconstruction_stats(flat)["patents_with_out_of_order_numbering"]
        assert n < len(all_claims) * 0.05, f"{n}/640 patents out of order — parser regression?"

    def test_all_numbers_positive(self, all_claims):
        for pid, claims in all_claims.items():
            assert all(c.claim_number >= 1 for c in claims), pid

    def test_no_claim_text_is_empty(self, all_claims):
        for pid, claims in all_claims.items():
            for c in claims:
                assert c.text.strip(), f"{pid}:{c.claim_number} produced empty text"

    def test_known_patent_20240051333(self, all_claims):
        """SPOKE — 10 entries, index-0 fragment, numbered 2..10 following."""
        claims = all_claims["20240051333"]
        assert [c.claim_number for c in claims] == list(range(1, 11))
        assert claims[0].status is RS.PREAMBLE_STRIPPED
        assert claims[0].is_independent
        assert claims[0].text.startswith("an axle body")
        assert claims[1].depends_on == [1]
        assert claims[5].depends_on == [5]

    def test_no_canceled_text_leaks_into_real_claims(self, all_claims):
        for pid, claims in all_claims.items():
            for c in claims:
                if c.status is not RS.CANCELED:
                    assert "(canceled)" not in c.text.lower(), f"{pid}:{c.claim_number}"

    def test_independent_claims_exist_everywhere(self, all_claims):
        """Every patent must have at least one independent claim, or retrieval is broken."""
        missing = [pid for pid, cs in all_claims.items() if not any(c.is_independent for c in cs)]
        assert missing == [], f"patents with no independent claim: {missing[:10]}"

    def test_stats_shape(self, all_claims):
        flat = [c for cs in all_claims.values() for c in cs]
        s = reconstruction_stats(flat)
        assert s["claims_total"] == len(flat)
        assert s["independent"] + s["dependent"] <= s["claims_total"] + s["status_canceled"]

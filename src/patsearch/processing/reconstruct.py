"""Reconstruct canonical claims from the raw fragment list.

Upstream XML parsing stripped claim preambles: only 68.9% of entries start with a
number, and 88.6% of patents open with a numberless fragment. Appending every
numberless fragment to the previous claim (the obvious rule) corrupts data — it drops
claim 1 and glues new independent claims onto the preceding claim.

Approach: segment first, number second. A fragment starts a new claim if it is first,
or if the previous entry ended on '.'; otherwise it continues. Numbers are resolved
afterwards from the next explicit number. Uncertainty is recorded on each Claim via
`status`, `number_inferred` and `raw_fragment_indexes`.

Assumptions, all drawn from patent drafting convention rather than this sample:
  - claims are numbered "N." or "N)" and numbered ascending
  - a claim is one or more sentences, so a trailing '.' closes it
  - dependency is expressed by a lead-in phrase ("according to claim 3")
  - cancellation is written as a number or range followed by "(canceled)"
Feeding a corpus that violates these raises `number_inferred` and the out-of-order
count in `reconstruction_stats` — check those before trusting output on new data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from patsearch.models import Claim, ReconstructionStatus
from patsearch.processing.normalize import is_blank, normalize_text

# Claim numbers are written "1." or "1)" across patent offices. The (?!\d) is
# load-bearing: without it "0.5 phr of accelerator" parses as claim 0. The digit
# bound rejects runaway matches on measurements.
_NUMBERED = re.compile(r"^\s*(\d{1,3})\s*[.)](?!\d)")

# Separator-agnostic rather than an enumeration of observed shapes: a leading number,
# an optional second number for a range, any punctuation between, then "(canceled)".
# Covers "1 - 5 . (canceled)", "1 .- 15 . (canceled)", "1 - 15 : (canceled)",
# "1-10 (cancelled)", "16 . (canceled)" without special-casing any of them.
_SEP = r"[\s.:;,\-–—]"
_CANCELED = re.compile(
    rf"^\s*(\d+)(?:{_SEP}+(\d+))?{_SEP}*\(\s*cancell?ed\s*\)",
    re.IGNORECASE,
)

_DEP_NUMS = r"([\d\s,;–—\-]+(?:\s*(?:to|through|and|or)\s*\d+)*)"
# Strong lead-ins are unambiguous, so "claim" may be omitted ("according to 26 wherein").
_DEP_STRONG = re.compile(
    r"(?:according to|as claimed in|as recited in|as set forth in|as defined in|"
    r"defined in|recited in|set forth in)\s+"
    r"(?:any\s+one\s+of\s+|any\s+of\s+|either\s+of\s+)?(?:claims?\s+)?" + _DEP_NUMS,
    re.IGNORECASE,
)
# Weak lead-ins need the word "claim" or they would match measurements ("of 26 mm").
_DEP_WEAK = re.compile(
    r"\b(?:of|in)\s+(?:any\s+one\s+of\s+|any\s+of\s+|either\s+of\s+)?claims?\s+" + _DEP_NUMS,
    re.IGNORECASE,
)
_DEP_RANGE = re.compile(r"(\d+)\s*(?:to|through|[-–—])\s*(\d+)", re.IGNORECASE)
_DEP_NUM = re.compile(r"\d+")

_TERMINAL = (".", ".\"", ".'", ".)")


@dataclass(slots=True)
class _Segment:
    """One claim before numbering: a head entry plus any continuation fragments."""

    indexes: list[int]
    texts: list[str]
    explicit_number: int | None
    is_canceled: bool
    canceled_range: tuple[int, int] | None = None


def _ends_complete(text: str) -> bool:
    """True if the entry closed its sentence, so a following fragment starts a new claim."""
    t = normalize_text(text)
    return bool(t) and t.endswith(_TERMINAL)


def parse_dependencies(text: str) -> list[int]:
    """Claim numbers this claim depends on. Empty list means independent."""
    deps: set[int] = set()
    for pattern in (_DEP_STRONG, _DEP_WEAK):
        for m in pattern.finditer(text or ""):
            body = m.group(1)
            for a, b in _DEP_RANGE.findall(body):
                lo, hi = int(a), int(b)
                if lo <= hi and hi - lo < 100:
                    deps.update(range(lo, hi + 1))
            residue = _DEP_RANGE.sub(" ", body)  # ranges already consumed
            deps.update(int(n) for n in _DEP_NUM.findall(residue))
    return sorted(deps)


def _segment(entries: list[str]) -> list[_Segment]:
    """Group raw entries into per-claim segments without assigning numbers yet."""
    segs: list[_Segment] = []
    for i, raw in enumerate(entries):
        if is_blank(raw):
            continue

        cm = _CANCELED.match(raw)
        if cm:
            lo = int(cm.group(1))
            hi = int(cm.group(2)) if cm.group(2) else lo
            segs.append(
                _Segment([i], [raw], explicit_number=lo, is_canceled=True, canceled_range=(lo, hi))
            )
            continue

        nm = _NUMBERED.match(raw)
        if nm:
            segs.append(_Segment([i], [raw], explicit_number=int(nm.group(1)), is_canceled=False))
            continue

        # Numberless fragment: continuation, or a new claim whose preamble was lost?
        prev = segs[-1] if segs else None
        starts_new = (
            prev is None
            or prev.is_canceled
            or _ends_complete(prev.texts[-1])
        )
        if starts_new:
            segs.append(_Segment([i], [raw], explicit_number=None, is_canceled=False))
        else:
            prev.indexes.append(i)
            prev.texts.append(raw)

    return segs


def _assign_numbers(segs: list[_Segment]) -> list[int | None]:
    """Resolve a claim number for every segment."""
    numbers: list[int | None] = [s.explicit_number for s in segs]

    for i, n in enumerate(numbers):
        if n is not None:
            continue
        # Anchor to the next explicit number: a stripped claim before "2 ." is claim 1.
        nxt = next((j for j in range(i + 1, len(segs)) if segs[j].explicit_number is not None), None)
        if nxt is not None:
            candidate = segs[nxt].explicit_number - (nxt - i)
            if candidate >= 1:
                numbers[i] = candidate
                continue
        prev = next((j for j in range(i - 1, -1, -1) if numbers[j] is not None), None)
        if prev is None:
            numbers[i] = 1
            continue
        # A canceled range ends at `hi`, so the claim after "1 - 15 . (canceled)" is 16.
        base = segs[prev].canceled_range[1] if segs[prev].canceled_range else numbers[prev]
        numbers[i] = base + (i - prev)

    return numbers


def reconstruct_claims(patent_id: str, entries: list[str]) -> list[Claim]:
    """Turn a raw claims array into canonical Claim objects.

    Canceled markers are returned with status CANCELED so numbering gaps stay
    explainable; callers filter them out before indexing.
    """
    segs = _segment(entries or [])
    if not segs:
        return []

    numbers = _assign_numbers(segs)
    claims: list[Claim] = []

    for seg, number in zip(segs, numbers, strict=True):
        raw_text = " ".join(seg.texts).strip()
        text = normalize_text(raw_text)

        if seg.is_canceled:
            status = ReconstructionStatus.CANCELED
        elif seg.explicit_number is None:
            status = ReconstructionStatus.PREAMBLE_STRIPPED
        elif len(seg.texts) > 1:
            status = ReconstructionStatus.REJOINED
        else:
            status = ReconstructionStatus.ORIGINAL

        deps = [] if seg.is_canceled else parse_dependencies(text)
        deps = [d for d in deps if d != number]  # a claim never depends on itself

        # Canceled markers cover a range and are not claims, so they get their own
        # id namespace — otherwise "1 - 15 . (canceled)" would collide with claim 1.
        if seg.is_canceled and seg.canceled_range:
            lo, hi = seg.canceled_range
            claim_id = f"{patent_id}:canceled:{lo}-{hi}"
        else:
            claim_id = f"{patent_id}:claim:{number}"

        claims.append(
            Claim(
                claim_id=claim_id,
                patent_id=patent_id,
                claim_number=number,
                text=text,
                raw_text=raw_text,
                is_independent=(not deps) and not seg.is_canceled,
                depends_on=deps,
                status=status,
                raw_fragment_indexes=list(seg.indexes),
                number_inferred=seg.explicit_number is None,
            )
        )

    return claims


def reconstruction_stats(claims: list[Claim]) -> dict[str, int]:
    """Counts for reports/extraction_quality.json."""
    from collections import Counter, defaultdict

    by_status = Counter(c.status.value for c in claims)

    # Some source patents label claims out of sequence (e.g. "14 ." sitting between
    # 9 and 10). We report that rather than forcing monotonicity onto real data.
    by_patent: dict[str, list[int]] = defaultdict(list)
    for c in claims:
        if c.status is not ReconstructionStatus.CANCELED:
            by_patent[c.patent_id].append(c.claim_number)
    out_of_order = sum(1 for nums in by_patent.values() if nums != sorted(nums))

    return {
        "claims_total": len(claims),
        "independent": sum(1 for c in claims if c.is_independent),
        "dependent": sum(1 for c in claims if c.depends_on),
        "number_inferred": sum(1 for c in claims if c.number_inferred),
        # Counted separately from status: a claim can be both preamble-stripped and
        # multi-fragment, and status only records one of those.
        "multi_fragment": sum(1 for c in claims if len(c.raw_fragment_indexes) > 1),
        "patents_with_out_of_order_numbering": out_of_order,
        **{f"status_{k}": v for k, v in sorted(by_status.items())},
    }

"""Turn patents and reconstructed claims into indexable search records.

Retrieval happens at claim/passage granularity because indexing a whole patent as one
unit destroys precision — a 40-claim patent matches almost any query. Results are
regrouped by patent at query time.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

from patsearch.models import Claim, Patent, RecordType, ReconstructionStatus, SearchRecord
from patsearch.processing.normalize import normalize_paragraphs

# Passage sizing. Long enough to carry context, short enough to stay precise.
TARGET_WORDS = 320
MAX_WORDS = 500
OVERLAP_PARAGRAPHS = 1


def _classification_fields(p: Patent) -> dict[str, str]:
    return {
        "classification_raw": p.classification_raw,
        "classification_section": p.classification_section,
        "classification_class": p.classification_class,
        "classification_subclass": p.classification_subclass,
    }


def chunk_paragraphs(
    paragraphs: list[str],
    *,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_PARAGRAPHS,
) -> Iterator[tuple[int, int, str]]:
    """Group paragraphs into passages, yielding (start_idx, end_idx, text).

    Indexes refer to positions in the supplied list. A single paragraph longer than
    max_words is emitted alone rather than split, so sentences stay intact.
    """
    if not paragraphs:
        return

    i = 0
    n = len(paragraphs)
    while i < n:
        start = i
        words = 0
        parts: list[str] = []
        while i < n:
            w = len(paragraphs[i].split())
            if parts and words + w > max_words:
                break
            parts.append(paragraphs[i])
            words += w
            i += 1
            if words >= target_words:
                break
        yield start, i - 1, " ".join(parts)
        if i >= n:
            break
        i = max(i - overlap, start + 1)  # overlap, but always make progress


def build_records(patent: Patent, claims: Iterable[Claim]) -> list[SearchRecord]:
    """Emit every indexable record for one patent."""
    cls = _classification_fields(patent)
    recs: list[SearchRecord] = []

    def _base(record_type: RecordType, suffix: str, text: str, **extra) -> SearchRecord:
        return SearchRecord(
            record_id=f"{patent.patent_id}:{suffix}",
            patent_id=patent.patent_id,
            record_type=record_type,
            text=text,
            title=patent.title,
            abstract=patent.abstract,
            **cls,
            **extra,
        )

    # Summary: title + abstract together, for broad "what is this patent about" queries.
    summary_text = " ".join(x for x in (patent.title, patent.abstract) if x)
    if summary_text:
        recs.append(_base(RecordType.SUMMARY, "summary", summary_text))

    if patent.abstract:
        recs.append(_base(RecordType.ABSTRACT, "abstract", patent.abstract))

    for c in claims:
        if c.status is ReconstructionStatus.CANCELED or not c.text:
            continue
        recs.append(
            _base(
                RecordType.CLAIM,
                f"claim:{c.claim_number}",
                c.text,
                claim_number=c.claim_number,
                is_independent=c.is_independent,
            )
        )

    paras = normalize_paragraphs(patent.description_paragraphs)
    for start, end, text in chunk_paragraphs(paras):
        if not text.strip():
            continue
        recs.append(
            _base(
                RecordType.DESCRIPTION,
                f"description:{start}-{end}",
                text,
                paragraph_start=start,
                paragraph_end=end,
            )
        )

    return recs

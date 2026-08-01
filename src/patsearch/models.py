"""Domain model.

Dataclasses rather than a validation framework: the schema is small, fixed, and we
want the validation *rules* to be explicit and testable rather than declarative.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class ReconstructionStatus(str, Enum):
    """How a claim's text was derived from the raw fragment list."""

    ORIGINAL = "original_complete"        # entry began with "N ." and stands alone
    REJOINED = "rejoined_continuation"    # numbered start + appended continuation fragments
    PREAMBLE_STRIPPED = "preamble_stripped"  # fragment with no number; preamble lost upstream
    CANCELED = "canceled"                 # "1 - 5 . (canceled)" marker


class RecordType(str, Enum):
    SUMMARY = "summary"
    CLAIM = "claim"
    DESCRIPTION = "description"
    ABSTRACT = "abstract"


@dataclass(slots=True)
class Claim:
    claim_id: str
    patent_id: str
    claim_number: int | None
    text: str
    raw_text: str
    is_independent: bool
    depends_on: list[int] = field(default_factory=list)
    status: ReconstructionStatus = ReconstructionStatus.ORIGINAL
    raw_fragment_indexes: list[int] = field(default_factory=list)
    number_inferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(slots=True)
class Patent:
    patent_id: str
    title: str
    abstract: str
    classification_raw: str
    claims_raw: list[str]
    description_paragraphs: list[str]
    bibtex: str
    source_file: str
    filename: str = ""

    # populated by the quality pass
    has_description: bool = True

    @property
    def classification_section(self) -> str:
        """'B' — the top-level CPC section."""
        return self.classification_raw[:1]

    @property
    def classification_class(self) -> str:
        """'B60' — CPC class."""
        return self.classification_raw[:3]

    @property
    def classification_subclass(self) -> str:
        """'B60B' — CPC subclass. The deepest level this corpus encodes unambiguously.

        Codes are packed ('B60B1110FI'); group/subgroup cannot be recovered without
        guessing digit boundaries (11/10 vs 1/110), so we stop here deliberately.
        """
        return self.classification_raw[:4]


@dataclass(slots=True)
class SearchRecord:
    """One indexable unit. Retrieval happens here; results are grouped by patent later."""

    record_id: str
    patent_id: str
    record_type: RecordType
    text: str

    title: str = ""
    # Denormalised onto every record so title/abstract keyword filters resolve in one
    # pass instead of a patent-level join. Costs index size, buys filter latency.
    abstract: str = ""
    classification_raw: str = ""
    classification_section: str = ""
    classification_class: str = ""
    classification_subclass: str = ""

    # claim records only
    claim_number: int | None = None
    is_independent: bool | None = None

    # description records only
    paragraph_start: int | None = None
    paragraph_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["record_type"] = self.record_type.value
        return {k: v for k, v in d.items() if v is not None}


@dataclass(slots=True)
class ValidationIssue:
    patent_id: str
    field: str
    issue: str
    severity: str  # "error" (unusable) | "warning" (degraded but usable)

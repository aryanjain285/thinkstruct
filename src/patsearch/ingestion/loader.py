"""Load and validate raw patent JSON.

Scalability note: `iter_patents` is a generator and reads one file at a time. The
sample is 640 patents, but the same code path handles the 10M-patent target without
change — memory is bounded by the largest single file, not the corpus.

Validation philosophy: we do not silently drop records. Every deviation produces a
ValidationIssue with a severity, and the caller decides. Measured on this corpus,
zero patents are unusable; 119 are degraded (blank descriptions) and are kept.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from patsearch.models import Patent, ValidationIssue
from patsearch.processing.normalize import is_blank, normalize_text

REQUIRED_FIELDS = (
    "title",
    "doc_number",
    "abstract",
    "detailed_description",
    "claims",
    "classification",
)


class PatentLoadError(Exception):
    """Raised when a source file cannot be parsed at all."""


def _coerce_str_list(value: Any) -> list[str]:
    """Source arrays are list[str] in this corpus, but tolerate None and stray types."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [x if isinstance(x, str) else str(x) for x in value if x is not None]


def validate_record(rec: dict[str, Any], source: str) -> list[ValidationIssue]:
    """Return all issues for one raw record. Empty list means clean."""
    pid = str(rec.get("doc_number") or "<unknown>")
    issues: list[ValidationIssue] = []

    for f in REQUIRED_FIELDS:
        if f not in rec:
            issues.append(ValidationIssue(pid, f, "field absent", "error"))

    if not rec.get("doc_number"):
        issues.append(ValidationIssue(pid, "doc_number", "missing or empty", "error"))

    if is_blank(rec.get("title")):
        issues.append(ValidationIssue(pid, "title", "blank", "warning"))
    if is_blank(rec.get("abstract")):
        issues.append(ValidationIssue(pid, "abstract", "blank", "warning"))
    if is_blank(rec.get("classification")):
        issues.append(ValidationIssue(pid, "classification", "blank", "warning"))

    claims = _coerce_str_list(rec.get("claims"))
    if not claims or all(is_blank(c) for c in claims):
        # A patent with no claims cannot serve the core use case.
        issues.append(ValidationIssue(pid, "claims", "no non-blank claims", "error"))

    desc = _coerce_str_list(rec.get("detailed_description"))
    if not desc or all(is_blank(p) for p in desc):
        # Measured: 119/640. Still searchable via claims/abstract, so warning not error.
        issues.append(ValidationIssue(pid, "detailed_description", "all paragraphs blank", "warning"))

    return issues


def _to_patent(rec: dict[str, Any], source: str) -> Patent:
    desc = _coerce_str_list(rec.get("detailed_description"))
    return Patent(
        patent_id=str(rec["doc_number"]).strip(),
        title=normalize_text(rec.get("title")),
        abstract=normalize_text(rec.get("abstract")),
        classification_raw=(rec.get("classification") or "").strip(),
        claims_raw=_coerce_str_list(rec.get("claims")),
        description_paragraphs=desc,
        bibtex=(rec.get("bibtex") or "").strip(),
        filename=(rec.get("filename") or "").strip(),
        source_file=source,
        has_description=any(not is_blank(p) for p in desc),
    )


def iter_patents(
    raw_dir: Path,
    *,
    collect_issues: list[ValidationIssue] | None = None,
    skip_errors: bool = True,
) -> Iterator[Patent]:
    """Stream Patent objects from every ``patents_*.json`` in ``raw_dir``.

    Args:
        raw_dir: directory of weekly JSON files.
        collect_issues: if provided, all ValidationIssues are appended to it.
        skip_errors: when True, records with error-severity issues are not yielded.

    Raises:
        PatentLoadError: if a file is not valid JSON or is not a list.
    """
    files = sorted(raw_dir.glob("patents_*.json"))
    if not files:
        raise PatentLoadError(f"no patents_*.json under {raw_dir}")

    seen: set[str] = set()
    for path in files:
        try:
            with path.open(encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise PatentLoadError(f"{path.name}: {exc}") from exc

        if not isinstance(payload, list):
            raise PatentLoadError(f"{path.name}: expected a list, got {type(payload).__name__}")

        for rec in payload:
            if not isinstance(rec, dict):
                continue
            issues = validate_record(rec, path.name)
            pid = str(rec.get("doc_number") or "").strip()

            if pid and pid in seen:
                issues.append(ValidationIssue(pid, "doc_number", "duplicate across corpus", "error"))
            elif pid:
                seen.add(pid)

            if collect_issues is not None:
                collect_issues.extend(issues)

            if skip_errors and any(i.severity == "error" for i in issues):
                continue
            yield _to_patent(rec, path.name)


def load_all(raw_dir: Path) -> tuple[list[Patent], list[ValidationIssue]]:
    """Eager convenience wrapper. Prefer iter_patents for large corpora."""
    issues: list[ValidationIssue] = []
    pats = list(iter_patents(raw_dir, collect_issues=issues))
    return pats, issues


def quality_report(patents: list[Patent], issues: list[ValidationIssue]) -> dict[str, Any]:
    """Summary suitable for reports/data_quality.json and the README."""
    return {
        "patents_loaded": len(patents),
        "source_files": len({p.source_file for p in patents}),
        "issues_total": len(issues),
        "issues_by_severity": dict(Counter(i.severity for i in issues)),
        "issues_by_field": dict(Counter(i.field for i in issues)),
        "patents_without_description": sum(1 for p in patents if not p.has_description),
        "classification_subclasses": dict(
            Counter(p.classification_subclass for p in patents).most_common()
        ),
        "claims_entries_total": sum(len(p.claims_raw) for p in patents),
        "description_paragraphs_total": sum(len(p.description_paragraphs) for p in patents),
    }

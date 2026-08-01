"""Deterministic text normalization.

Design rule: this layer is *conservative*. Patent claims are legal text; aggressive
rewriting changes meaning. We only do transformations that are provably safe:

  - Unicode NFKC (collapses em-spaces, ligatures, fullwidth forms)
  - removal of Unicode format characters (invisible-times U+2062 etc. appear in
    formulae in this corpus and break tokenization)
  - whitespace collapse

Raw text is always preserved alongside normalized text so the UI can show either.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")

# XML tag-stripping in the source data glued words to figure references:
#   "Referring toFIG.1, wheel100has..."  ->  "Referring to FIG. 1, wheel100has..."
# We only split the FIG. case, which is unambiguous. We deliberately do NOT split
# letter/digit boundaries generally: "B60B104FI", "PEEK", "4WD", "E*" are meaningful.
_FIG_GLUE = re.compile(r"(?<=[a-z])(?=FIGS?\.)")
_FIG_NUM = re.compile(r"\b(FIGS?\.)\s*(\d)")


def normalize_text(s: str | None) -> str:
    """Normalize a single string. Returns '' for None/blank input."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    # Drop format/control characters (Cf) — invisible-times, ZWJ, soft hyphen.
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Cf")
    s = _FIG_GLUE.sub(" ", s)
    s = _FIG_NUM.sub(r"\1 \2", s)
    s = _WS.sub(" ", s)
    return s.strip()


def is_blank(s: str | None) -> bool:
    """True if the string is None, empty, or whitespace/format characters only."""
    return not normalize_text(s)


def normalize_paragraphs(paras: list[str] | None) -> list[str]:
    """Normalize a paragraph list, dropping blanks. Order is preserved."""
    if not paras:
        return []
    out = []
    for p in paras:
        n = normalize_text(p)
        if n:
            out.append(n)
    return out

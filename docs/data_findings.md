# Dataset findings — `patent_data_small`

Generated from `_scratch/explore.py` and `_scratch/deep.py`. Every number here is measured, not assumed.

## 1. Volume

| Metric | Value |
|---|---|
| JSON files | 64 (weekly, `ipa240215` → `ipa250501`) |
| Patents per file | exactly 10, every file |
| **Total patents** | **640** |
| Total text | 15.36 M chars (~3.8 M tokens) |

> **The plan assumed 60 patents. It is 640.** This changes training-data feasibility,
> LangExtract cost (10,578 claims, not ~1,000), and makes a patent-level train/val/test
> split viable without leaning on grouped cross-validation.

## 2. Schema — nothing is missing

All 8 fields (`title`, `doc_number`, `filename`, `abstract`, `detailed_description`,
`claims`, `bibtex`, `classification`) are present on **all 640** patents.

The task says *"some patents may be missing some fields, you may choose to either exclude
those patents or handle them separately. Document which choice you make."*

**Measured answer: no field is ever absent, and no field is ever an empty string/list —
except descriptions, which are present-but-blank (below). So nothing is excluded on
missing-field grounds.** This is a README line we can defend with evidence.

- `doc_number`: 640 unique, zero duplicates, all 11 chars. Safe as a primary key.
- Titles: 43 duplicated title strings — titles are **not** unique, do not key on them.

## 3. Claim fragmentation — the crux, and the plan's algorithm is wrong here

`claims` is a flat `list[str]` — 10,578 entries total (mean 16.5/patent, max 46).
Only **68.9%** of entries begin with a claim number (`N.`). The other **3,294 are fragments**.

| Finding | Count | Share |
|---|---|---|
| Fragment at **index 0** (before any numbered claim) | 567 patents | **88.6%** |
| Patents with **no `1.` entry at all** | 510 | 79.7% |
| Fragments **mid-list** | 2,727 | — |
| Patents with contiguous ascending numbering | 195 | 30.5% |
| Patents with gaps / out-of-order | 388 | 60.6% |

### What the fragments actually are

Claim **preambles are stripped**, not merely split. Example — `20240051333` ("SPOKE"),
10 entries, where entry 0 is claim 1 with `1 . A spoke comprising:` missing entirely:

```
[0] 'an axle body, having a middle segment and two connecting segments, wherein ...'
[1] '2 . The spoke according to claim 1 , wherein ...'
```

### ⚠️ Why the proposed reconstruction rule breaks

The plan's Step 2 says: *"A fragment without a new claim number should normally be appended
to the current claim until the next numbered start."*

That rule fails on both dominant cases:

1. **Index-0 fragments (88.6% of patents)** have no "current claim" to append to. Claim 1
   would be dropped or misassigned.
2. **Mid-list fragments are frequently new *independent* claims** whose preamble was
   stripped — not continuations. Appending them silently corrupts the previous claim:

```
20240051338, entry 1:
  prev: '1 - 5 . (canceled)'
  FRAG: 'a tread portion extending in a tire circumferential direction and having
         an annular shape; a pair of sidewall portions ...'
```
   That is claim 6, a fresh independent claim. The naive rule would glue it onto a
   `(canceled)` marker.

```
20240051334, entry 14 of 20 (after claim 11):
  FRAG: 'a wheel rim, a wheel spoke, and a tire, wherein the tire is sleeved on ...'
```
   A new independent apparatus claim, not a continuation of claim 11.

**Required fix:** classify each fragment before grouping. Signals available:
- starts with `a`/`an`/`the` + noun-phrase and ends with `;` or `.` → likely a stripped
  independent-claim body
- contains dependency language (`according to claim N`) → dependent claim
- previous entry ends mid-sentence (no terminal punctuation) → true continuation
- `N - M . (canceled)` markers explain numbering gaps and must be preserved, not dropped

7,159 entries contain dependency language, so dependency parsing has plenty of signal.

## 4. Descriptions — half of all paragraphs are blank

| Metric | Value |
|---|---|
| Total paragraphs | 42,040 |
| **Blank paragraphs** | **20,819 (49.5%)** |
| Patents where description is **entirely blank** | **119 (18.6%)** |
| Non-blank paragraph length | median 463, mean 557 chars |

The 119 all-blank-description patents are the "handle separately" case: they are still
fully searchable via title/abstract/claims, so **keep them, flag them, and report the
count** rather than excluding.

Short non-blank paragraphs are formulae and Unicode artifacts, needing normalization:
```
'A-B-C   (I)'          'OD (mm)≥2.135×SW (mm)+282.3'
'1.1≤b/C≤1.4  (1)'     'Z=112⁢π⁢i×ρarea+1Rf,'
```
(` ` em-space, `⁢` invisible-times — NFKC normalization required.)

## 5. Classification codes

All 640 are `B60*` (vehicles). 145 distinct codes, every one suffixed `FI`.

| Subclass | Count | Meaning |
|---|---|---|
| B60C | 318 | tyres |
| **B60B** | **298** | **vehicle wheels — the filter the task names explicitly** |
| B60D/F/G/R/J | 24 | couplings, misc |

Format is packed: `B60B104FI`, `B60B1110FI`. Section/class/subclass (`B`, `B60`, `B60B`)
parse cleanly from chars 0–4. **Group/subgroup is ambiguous** — `B60B1110FI` could be
11/10 or 1/110. Prefix filtering is unambiguous and is all the task requires; do not
over-parse.

## 6. Indexable corpus

| Record type | Count |
|---|---|
| Claim entries (raw) | 10,578 |
| Description paragraphs ≥100 chars | 20,284 |
| Abstracts | 640 |
| **Total after cleaning** | **~31,500** |

Trivially small for a single local OpenSearch node — no sharding concerns at this scale.
The scaling story in Part 2 is therefore genuinely hypothetical and should be argued
from the 10M-patent target, not from observed pain.

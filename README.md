# Patent claim search

A hybrid search engine over USPTO vehicle patent applications, built for the
Thinkstruct coding task.

---

## Problem statement

> **Given a description of an invention in plain language, find the specific patent
> claims that cover similar technical ground — ranked, filtered by the constraints a
> patent professional actually uses, and with the matching claim shown as evidence.**

### The business problem

Before filing, litigating, or committing R&D budget, someone has to answer: *has
anyone already claimed this?* Getting it wrong is expensive in both directions.

| Failure | Consequence |
|---|---|
| **Miss** relevant prior art | A patent issues that is invalid. It gets discovered in litigation, when invalidation costs $1–3M and the underlying R&D is already sunk |
| **False positive** noise | Attorneys read hundreds of irrelevant patents at $400–700/hour. Search that returns 200 vaguely-related hits has not saved anyone time |

The hard part is that **prior art rarely uses your vocabulary.** A "carbon fibre spoke"
may be claimed elsewhere as an "elongate composite tension member". Keyword search
misses it; a human reading 10M patents is not an option. That vocabulary gap is
precisely what a hybrid lexical + semantic engine exists to close — and this repo
[measures whether it actually does](#results), rather than assuming it.

### Why claim-level, not patent-level

Retrieval operates on **individual claims and description passages**, never whole
patents. Three reasons, all practical:

1. **Claims are the legal unit.** Infringement and novelty are determined claim by
   claim. "Patent X is relevant" is not an actionable answer; "claim 7 of patent X
   recites the same spoke-to-rim bonding" is.
2. **Precision.** A 46-claim patent indexed as one blob matches nearly any query in
   its field. The corpus here averages 16.5 claims per patent.
3. **Evidence.** Every result names the record that matched, so the output can be
   checked rather than trusted.

Results are then **regrouped by patent** for display, scoring
`max(record) + α·mean(next 3)` so a patent matching on several claims outranks one
matching by a fluke on a single passage.

### Scope boundary

This system **surfaces and ranks technically similar claims**. It does not, and must
not, output novelty, validity, or infringement conclusions — those are legal
determinations. The reranker prompt says so explicitly. The product is a
*prioritised reading list with evidence*, which is the part that actually scales.

The three metadata constraints the brief names — classification-code prefix,
title/abstract keywords, exact title — are supported as **pre-filters** shared by both
retrieval branches, because examiners almost never search the whole corpus at once.

## Why claim reconstruction is the core of this project

The dataset's `claims` field is a flat `list[str]` whose structure was damaged by
upstream XML parsing. Measured across all 640 patents:

| | |
|---|---|
| Entries beginning with a claim number (`N .`) | **68.9%** |
| Patents whose entry 0 is a numberless fragment | **88.6%** |
| Patents with no `1 .` entry at all | **79.7%** |
| Patents with non-contiguous numbering | 60.6% |

Claim **preambles are stripped, not merely split**. `20240051333` ("SPOKE") opens with
`'an axle body, having a middle segment...'` — the `1 . A spoke comprising:` is gone.

The obvious fix — *append every numberless fragment to the previous claim* — corrupts
the data in two ways. It drops claim 1 (nothing precedes an index-0 fragment), and it
glues **new independent claims** onto the preceding one:

```
20240051338, entry 1:
  prev: '1 - 5 . (canceled)'
  FRAG: 'a tread portion extending in a tire circumferential direction...'   <- claim 6
```

`src/patsearch/processing/reconstruct.py` instead **segments first, numbers second**: a
fragment starts a new claim if it is first or if the previous entry closed on `.`;
numbers are resolved afterwards by anchoring to the next explicit number. Its safety
net is an invariant verified over the whole corpus — **every one of the 10,578 source
entries is consumed exactly once**, so nothing is dropped or double-counted.

Full analysis: [`docs/data_findings.md`](docs/data_findings.md).

## What is built

```
raw JSON ─► validate ─► normalize ─► reconstruct claims ─► search records ─► OpenSearch
                                                                                  │
                    ┌─────────────────────────────────────────────────────────────┤
                    ▼                                                             ▼
              BM25 (lexical)                                            k-NN (dense vectors)
                    └──────────────► RRF fusion ──► LLM/cross-encoder rerank ──► group by patent
```

| Layer | Implementation |
|---|---|
| Ingestion | Streaming loader, severity-graded validation, NFKC normalization |
| Claim reconstruction | Fragment classifier + dependency parser, 100% entry accounting |
| Records | summary / abstract / claim / description-passage, stable IDs |
| Index | OpenSearch, `english` analyzer + `knn_vector` (HNSW, lucene engine) |
| Retrieval | BM25, dense, hybrid via reciprocal-rank fusion |
| Reranking | Pluggable: HF cross-encoder **or** hosted LLM |
| Aggregation | Patent-level scoring: `max(record) + α·mean(next 3)` |
| Evaluation | Recall / nDCG / MRR / P@k / MAP with a generated test collection |
| Scale POC | Job queue + status store with idempotency and crash recovery |

## Repository layout

```
├── src/patsearch/
│   ├── config.py              paths, env, .env loading
│   ├── models.py              Patent, Claim, SearchRecord domain types
│   ├── pipeline.py            end-to-end wiring: corpus -> index, query -> results
│   ├── ingestion/
│   │   ├── loader.py          streaming load + severity-graded validation
│   │   ├── status_store.py    job queue / status DB  (Part 2 POC)
│   │   └── worker.py          queue-driven ingestion worker
│   ├── processing/
│   │   ├── normalize.py       conservative NFKC text normalisation
│   │   ├── reconstruct.py     claim reconstruction  ← the hard part
│   │   └── records.py         search-record generation, passage chunking
│   ├── embeddings/service.py  provider registry: sentence-transformers | OpenAI
│   ├── search/
│   │   ├── client.py          OpenSearch connection + health
│   │   ├── index.py           mapping, analyzer, bulk indexing
│   │   └── query.py           BM25 / dense / hybrid, filters, RRF, aggregation
│   ├── reranking/service.py   cross-encoder | LLM reranker registry
│   ├── evaluation/
│   │   ├── metrics.py         recall, nDCG, MRR, MAP, success@k
│   │   ├── evaluator.py       test collections, system comparison
│   │   ├── query_gen.py       LLM query paraphrasing (removes lexical bias)
│   │   ├── pooling.py         TREC-style pooling + LLM assessor
│   │   └── significance.py    paired bootstrap / t-test
│   └── api/                   FastAPI service
├── ui/                        React + Vite frontend
├── scripts/
│   ├── setup.sh               one-shot Linux bootstrap (A-Z)
│   ├── build_index.py         corpus -> OpenSearch
│   ├── search.py              CLI search
│   ├── benchmark.py           latency, filters on vs off
│   ├── evaluate.py            system comparison + significance
│   ├── build_qrels.py         pooled cross-patent relevance judgements
│   └── ingest.py              queue-driven ingestion  (Part 2 POC)
├── config/synonyms.txt        editable spelling equivalences (no code change)
├── docs/
│   ├── data_findings.md       what is actually in the data, measured
│   ├── scaling.md             Part 2: 10M-patent architecture + cost
│   └── efficiency.md          Part 1: latency with/without filters
├── docker/                    API and UI images
├── docker-compose.yml         opensearch + api + ui
├── reports/                   generated metrics (committed as evidence)
└── tests/                     281 tests, unit + integration
```

## Quickstart

### Linux, one command

Requires only **Docker** and **Python 3.11+**. OpenSearch is pulled and started as a
container — nothing to install by hand. No API key needed: the default embedder runs
locally.

```bash
# place the patent data zip in the repo root, then:
chmod +x scripts/setup.sh
./scripts/setup.sh
```

That checks prerequisites (including `vm.max_map_count`, the usual reason OpenSearch
dies on Linux), extracts the corpus, creates the venv, installs everything, downloads
the embedding model, starts OpenSearch, runs the tests, builds the index, and brings
up the API and React UI. It prints the URLs when done.

```bash
./scripts/setup.sh --embedder openai-small   # hosted embeddings instead (needs a key)
./scripts/setup.sh --docker                  # run API + UI in containers too
./scripts/setup.sh --no-ui                   # backend only
./scripts/setup.sh --clean                   # tear everything down
```

### Manual

```bash
uv venv .venv --python 3.12
uv pip install -e ".[all]" --python .venv/bin/python
cp .env.example .env                       # only for hosted models; .env is gitignored
docker compose up -d opensearch

python scripts/build_index.py                                    # BM25 only, no key
python scripts/build_index.py --embeddings --embedder minilm     # + dense, local model
python scripts/search.py --method hybrid \
  --query "flexible fibre spoke connected between a hub and a wheel rim" \
  --classification-prefix B60B
```

### Running the parts

| | Command |
|---|---|
| **Part 1** — search engine | `python scripts/build_index.py --embeddings` then `scripts/search.py` |
| **Part 1** — filter timing | `python scripts/benchmark.py` |
| **Part 2** — scaling doc | [`docs/scaling.md`](docs/scaling.md) |
| **Part 2** — POC | `python scripts/ingest.py --enqueue --run` |
| **Part 3** — evaluation | `python scripts/evaluate.py --generate --paraphrase` then `scripts/evaluate.py` |
| **Part 3** — pooled qrels | `python scripts/build_qrels.py` (cross-patent relevance) |
| **Part 3** — two-phase | `scripts/search.py --method hybrid_reranked` |
| Tests | `python -m pytest tests/ -q` |
| API | `uvicorn patsearch.api.main:app` → http://localhost:8000/docs |
| UI | `cd ui && npm install && npm run dev` → http://localhost:5173 |

### The three evaluation regimes

Retrieval quality depends heavily on how the test collection was built, so all three
are reported rather than only the flattering one:

| eval set | how built | what it measures | file |
|---|---|---|---|
| structural | patent's own abstract as query | known-item retrieval; **55% verbatim overlap** biases it toward lexical matching | `eval_set.jsonl` |
| paraphrased | queries rewritten by an LLM | same targets, **37% overlap** — a fairer lexical/semantic comparison | `eval_set_paraphrased.jsonl` |
| **pooled** | TREC pooling + LLM assessor | **cross-patent prior-art relevance** — the actual business task | `eval_set_pooled.jsonl` |

```bash
python scripts/evaluate.py --generate --paraphrase   # regimes 1 and 2
python scripts/build_qrels.py --dry-run              # pool size + call estimate
python scripts/build_qrels.py                        # regime 3 (judgements cached)
python scripts/evaluate.py --eval-path data/evaluation/eval_set_pooled.jsonl \
  --systems bm25 dense hybrid --baseline bm25
```

## Configuration

Every model is swappable by flag or env var; no calling code changes.

```bash
# embedders
--embedder minilm            # sentence-transformers/all-MiniLM-L6-v2 (local, 384d)
--embedder bge-small         # BAAI/bge-small-en-v1.5 (local, 384d)
--embedder openai-small      # text-embedding-3-small (hosted, 1536d)
--embedder st:/path/to/model # sideloaded local directory
--embedder openai-small --dimensions 512   # Matryoshka truncation

# rerankers
--reranker ce-minilm         # cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
--reranker bge-reranker      # BAAI/bge-reranker-base (local)
--reranker llm-mini          # gpt-5.4-mini (hosted)
--reranker off               # no-op baseline
```

Full lists: `PRESETS` in `embeddings/service.py`, `RERANKER_PRESETS` in
`reranking/service.py`.

## Enhancements chosen (Part 3)

**1. Two-phase search.** Hybrid retrieval returns 50 candidates; a reranker rescores
that fixed pool. Both reranker kinds see *identical* candidates, so the comparison is
controlled. The LLM reranker exists because `huggingface.co` is blocked on the
development network — the brief explicitly permits "asking a language model to output
rankings", and swapping back to a cross-encoder is a one-flag change.

**2. Evaluation framework.** Metrics are pure functions tested against hand-computed
values. The test collection is generated from corpus structure (a patent's abstract
should retrieve its own claims).

Three test collections are built and all three are reported, because the first two
produced conclusions that the third overturned — see [Results](#results). The final
one uses **TREC-style pooling**: union the top-15 from every system, judge each
candidate 0–3 against an examiner rubric, treat unjudged documents as non-relevant.
Judgements are cached on disk, so re-running costs nothing and human labels can
replace the LLM's wholesale.

Significance is tested with a **paired bootstrap** — a raw score difference on 80
queries means nothing without one.

## Results

### Headline: hybrid retrieval, measured properly

80 queries, 640 patents, 18,743 records, **pooled cross-patent relevance judgements**
(the regime that matches the actual business task).

| system | success@10 | recall@10 | recall@50 | nDCG@10 | MRR@10 | P@5 | P50 |
|---|---|---|---|---|---|---|---|
| bm25 | 1.000 | 0.205 | 0.405 | 0.527 | **0.971** | 0.858 | **4.4 ms** |
| dense | 1.000 | **0.253** | 0.463 | 0.584 | 0.951 | **0.930** | 310 ms |
| **hybrid** | 1.000 | 0.223 | **0.616** | **0.590** | 0.967 | 0.885 | 322 ms |

```
paired bootstrap, two-sided, n=80 (* = p < 0.05)
ndcg@10    dense  0.5840 vs bm25 0.5271   delta=+0.0569   p=0.0233 *
ndcg@10    hybrid 0.5898 vs bm25 0.5271   delta=+0.0627   p=0.0003 *
recall@50  dense  0.4631 vs bm25 0.4052   delta=+0.0580   p=0.0035 *
recall@50  hybrid 0.6161 vs bm25 0.4052   delta=+0.2109   p=0.0001 *
```

**Hybrid retrieves 52% more relevant prior art in the top 50 than BM25 (p = 0.0001).**
Every query surfaces relevant prior art in the top 10. Recall@10 is capped at 0.276
here because queries now average 40.8 relevant records.

### Why the evaluation had to be rebuilt twice

The first two attempts both produced *wrong conclusions*, and finding out why was most
of the engineering:

| # | eval set | how built | overlap | conclusion |
|---|---|---|---|---|
| 1 | structural | patent's own abstract as the query | **55.4%** | bm25 wins (0.367 vs 0.347 nDCG) |
| 2 | paraphrased | queries rewritten by an LLM | 36.9% | hybrid ahead, **not significant** (p=0.18) |
| 3 | **pooled** | TREC pooling + LLM assessor | — | **hybrid wins, p=0.0003** |

**Attempt 1** used each patent's own abstract as its query. Same drafter, same
vocabulary — 55% of query tokens appear verbatim in the target. That benchmark rewards
near-duplicate detection, which BM25 wins by construction.

**Attempt 2** paraphrased the queries away from that vocabulary. Hybrid moved ahead,
but at p = 0.18 it could not be claimed.

**Attempt 3** fixed the real flaw: qrels marked only the query's *own* patent as
relevant, so when hybrid correctly returned five different flexible-spoke patents,
four scored as false positives. TREC-style pooling — union the top-15 from every
system, judge each candidate 0–3 — surfaced **1,272 cross-patent relevant records**
that the earlier sets counted as errors. Mean relevant per query: 23.7 → 40.8.

All three are reproducible and all three are reported. The lesson is the deliverable:
**an unexamined benchmark will invert your conclusion**, and the first two would have
led to shipping BM25 and deleting the vector index.

> **Caveat, stated plainly:** the assessor is `gpt-5.4-mini`, not a patent examiner.
> It is adequate for *relative* comparison between systems — which is what a build/no-build
> decision needs — but it is not an absolute quality ceiling. Judgements are cached in
> `data/evaluation/judgements.jsonl` and can be replaced with human labels wholesale.

### Latency

Full analysis in [`docs/efficiency.md`](docs/efficiency.md). The headline:

```
dense total 319.8 ms
  ├── embed_query  314.9 ms  (98%)  <- hosted API round-trip
  └── retrieve       5.9 ms  ( 2%)  <- OpenSearch HNSW
```

**Dense retrieval is 5.9 ms.** It is not slow — the hosted embedding API is.
`--embedder minilm` runs locally and removes that hop entirely.

Filters cost BM25 +69% (2.6 → 4.3 ms) at 18.7K records, because there is nothing to
save at this scale. At 10M patents, `B60B` selects ~0.3% of the corpus and filtering
becomes the dominant optimisation rather than overhead.

### What these numbers still do not tell you

- The assessor is an LLM, not an examiner. Relative ordering is trustworthy; absolute
  values are not a quality ceiling.
- Pooling only judges what some system retrieved. A relevant claim that **every**
  system missed is invisible — the standard TREC limitation. Deeper pools and more
  diverse systems shrink it; they cannot eliminate it.
- 640 patents. Behaviour at 10M is argued in [`docs/scaling.md`](docs/scaling.md), not
  measured.

## Design decisions

| Decision | Rationale |
|---|---|
| Claim-level records | Whole-patent indexing destroys precision on multi-claim patents |
| Denormalise `abstract` onto every record | Filters resolve in one pass; trades index size for latency — the "make hybrid help performance" hint in the brief |
| Prefix query on `classification_raw` | Works at any depth (`B`, `B60`, `B60B`). Codes are packed (`B60B1110FI`) so group/subgroup **cannot** be parsed unambiguously — we deliberately stop at subclass |
| RRF over score addition | BM25 (tens) and cosine (0–1) are different scales; RRF is scale-free and needs no tuning |
| Pre-filter in k-NN traversal | Post-filtering returns fewer than *k* under selective filters |
| Keep the 119 blank-description patents | Still searchable via claims/abstract; excluding them loses 18.6% of the corpus |
| SQLite for the job store | Same transactional semantics as Postgres, no daemon, no Docker |

### Missing fields

The brief asks us to document the exclude-vs-handle choice. **Measured: no field is
ever absent and no field is ever empty, except descriptions.** 119 patents (18.6%) have
descriptions that are present but consist entirely of blank paragraphs. Those are
**kept and flagged**, never excluded — validation grades them `warning`, not `error`.
Nothing in this corpus is dropped.

## Limitations

- Auto-generated qrels measure known-item retrieval, not prior-art relevance.
- `build_index.py` embeds the whole corpus in memory before indexing. Fine at 18.7K
  records; the queue-driven `ingest.py` path is the one that scales.
- Some source claims are truncated upstream (`"wherein SA"`), so reconstructed text is
  occasionally odd. That is inherited damage — flagged via `status`, not repaired.
- Cross-encoder reranking is untested here because HuggingFace is network-blocked; the
  code path exists and is preset-selectable.
- Fine-tuning is not implemented. The evaluation harness it would need is.

# Demo script

Two minutes, slide by slide. Written for a technical audience — the terms are the real
ones, not simplified.

Every figure quoted here comes from `reports/`. If you re-run anything and the numbers
move, update this file too.

---

## Slide 1 — Problem · 20 sec

> Prior-art search. The failure mode is asymmetric — miss art and you get an invalid
> patent that surfaces in litigation; return noise and attorneys burn hours at $500 each.
>
> The core difficulty is lexical mismatch. A *carbon fibre spoke* gets claimed elsewhere
> as an *elongate composite tension member*. Pure BM25 has no path to that.
>
> And retrieval is claim-level, not document-level. A 46-claim patent indexed as one
> document matches anything in its field, and infringement is determined claim by claim
> anyway. 640 patents become 18,743 records.

---

## Slide 2 — Architecture · 30 sec

> Hybrid retrieval. BM25 for exact terminology — component names, materials, numeric
> limitations. Dense k-NN over embeddings for semantic equivalence. Both hit the same
> OpenSearch index.
>
> Metadata constraints — CPC prefix, title, abstract — are applied as **pre-filters** to
> both branches. For k-NN that's filtering during HNSW graph traversal, not
> post-filtering, so a selective filter still returns a full k.
>
> Fusion is **reciprocal rank fusion** — one over sixty plus rank, summed. BM25 scores are
> unbounded, cosine is zero to one; RRF uses rank only, so no score normalisation and no
> tuned weight. Agreement between retrievers is implicitly rewarded.
>
> Then a reranker over the top 50 — learning-to-rank, cross-encoder, or LLM, all behind
> one interface. Results collapse to patent level with max plus alpha times
> mean-of-top-three.

---

## Slide 3 — Claim reconstruction · 30 sec

> The dataset's claims field is a flat list with the structure destroyed by upstream XML
> parsing. Only 68.9% of entries retain a claim number. 88.6% of patents open with a
> numberless fragment. 79.7% have no claim-1 entry at all — the preamble is deleted, not
> relocated.
>
> Naive continuation-append drops claim 1, and welds stripped-preamble independent claims
> onto their predecessor.
>
> So: segment on terminal punctuation first, resolve numbering second by anchoring to the
> next explicit number.
>
> No ground truth exists, so verification is an invariant — all 10,578 source indices
> consumed exactly once, no gaps, no duplicates. That holds under schema drift;
> example-based tests don't.

---

## Slide 4 — Evaluation · 30 sec

> Three qrel regimes, because the first two inverted the conclusion.
>
> Structural known-item — patent's own abstract as query — gave 55% verbatim token overlap
> with the target. That's near-duplicate detection. BM25 won, correctly and uselessly.
>
> LLM-paraphrased queries dropped overlap to 37%. Hybrid moved ahead but at p equals 0.18
> — not claimable.
>
> The actual defect was the qrels marking only the source patent relevant, so correct
> cross-patent retrieval scored as false positives. TREC-style pooling — top-15 union
> across systems, graded zero-to-three — surfaced 1,272 cross-patent relevant records.
>
> Result: hybrid over BM25, recall-at-50 up 52%, nDCG-at-10 plus 0.063, **paired bootstrap
> p equals 0.0001**. LTR reranking adds another 0.12 nDCG on held-out patents — split by
> patent, not by row, to avoid leakage.

---

## Slide 5 — Scale · 25 sec

> Queue-driven ingestion with a job-status table. Idempotent on content hash plus parser
> and model version. Lease-based recovery for dead workers, retry accounting, quarantine
> after N failures. SQLite for the POC, same transactional claim semantics as Postgres.
>
> At 10 million patents: 293 million records. The binding constraint is vector memory —
> 1.8 terabytes float32, and HNSW wants it resident. Matryoshka truncation to 512 dims,
> int8 quantisation, and not embedding description passages takes that to 150 gigabytes.
> Roughly $2,600 a month down to $700.

---

## Close · 10 sec

> 329 tests. Every README figure derives from generated report JSON.
>
> Caveats I'd state before you ask: the assessor is an LLM, so relative ordering is sound
> but absolute scores are inflated by pooling bias. Cross-encoder reranking is wired but
> unmeasured. And 640 patents is a POC — the 10M numbers are reasoned, not measured.

---

Total ≈ 2:05. Cut the cost sentence on slide 5 if you need the margin.

---

# Live demo commands

If you are demonstrating rather than presenting slides.

```bash
# 1 — search working, with the filter the brief names
python scripts/search.py --method hybrid \
  --query "a wheel spoke made of carbon fibre bonded to the rim" \
  --classification-prefix B60B --top-k 5

# 2 — why hybrid: a deliberately non-patent-like phrasing
python scripts/search.py --method bm25   --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3
python scripts/search.py --method hybrid --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3

# 3 — claim reconstruction, including the invariant
python -m pytest tests/test_reconstruct.py -q

# 4 — latency with and without pre-filters (Part 1 deliverable)
python scripts/benchmark.py --methods bm25 hybrid --repeats 2

# 5 — ingestion is idempotent (Part 2 POC)
python scripts/ingest.py --enqueue

# 6 — full suite
python -m pytest tests/ -q
```

UI at `http://localhost:5173`, API docs at `http://localhost:8000/docs`.

---

# Likely questions

| Question | Answer |
|---|---|
| Why OpenSearch? | Lexical, vector and metadata filtering in one query with shared pre-filters. Splitting across a keyword store and a vector DB means two round-trips and two filter implementations that can disagree. |
| Alternatives considered? | Elasticsearch — near-identical, OpenSearch is the Apache-2.0 fork. pgvector — fine if you already run Postgres, but BM25 via tsvector is weaker on long technical text. Pinecone/Weaviate/Qdrant — better vector ergonomics, but you still need a lexical engine beside them, and BM25 was a strong baseline here. Vespa — best technical fit for hybrid ranking, heaviest to operate. FAISS — a library, not a search engine. |
| Why the Lucene k-NN engine? | It supports filtering during graph traversal. Every query here carries metadata constraints, so post-filtering would return fewer than k. |
| Why RRF over weighted score combination? | BM25 is unbounded, cosine is 0–1. Normalising drifts as the corpus changes. RRF uses rank only — no weight to tune. Trade-off: it discards score magnitude. |
| Why claim-level? | Claims are the legal unit; a 46-claim patent as one document matches anything in its field; and every result can name the record that matched. |
| Missing fields? | None absent. 119 patents (18.6%) have descriptions that are present but entirely blank paragraphs — kept and flagged as `warning`, not excluded, since they remain searchable via claims and abstract. |
| Why not parse the full CPC hierarchy? | Codes are packed — `B60B1110FI` is 11/10 or 1/110 with no delimiter. Section, class and subclass parse cleanly; below that we stop, because guessing produces filters that silently return the wrong patents. |
| Why LTR over cross-encoder fine-tuning? | Trains in seconds on CPU, scores in microseconds vs ~90 ms, and is what production search does. Trade-off: it only sees retrieval features, never the candidate text. |
| Are the numbers good? | The comparison is sound — identical qrels, p = 0.0001. Absolute values are optimistic: pooling only judges what some system retrieved, the assessor is an LLM and graded generously, and a corpus that is entirely wheels and tyres makes almost anything somewhat related. |
| What is unfinished? | Cross-encoder reranking is wired but unmeasured. `build_index.py` holds all embeddings in memory. No auth or rate limiting on the API. 640 patents is a POC. |

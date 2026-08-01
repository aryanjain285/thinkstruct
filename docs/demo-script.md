# Demo script

Two minutes. Read the quoted parts aloud — written to be spoken, not read.

Numbers come from `reports/`. Regenerate those and this needs updating.

---

## 1 · Problem · 15 sec

> Prior-art search. Before you file, someone checks nobody's already claimed it.
>
> The hard part is people don't use the same words. What you'd call a carbon fibre spoke,
> someone else claimed as an elongate composite tension member. BM25 will never connect
> those two.

---

## 2 · Architecture · 20 sec

> So I search twice. BM25 for the exact terminology, dense k-NN for the same idea worded
> differently. Same index, same pre-filters — and on the vector side that's filtering
> during the HNSW traversal, not after it.
>
> Then I merge with reciprocal rank fusion. Rank only, so there's nothing to normalise and
> no weight to tune.

---

## 3 · The data · 30 sec

> What actually took the time was the data, not the search.
>
> The claims field looks like a list of claims. It isn't. Only sixty-nine percent of
> entries still have their number, and in eighty percent of patents claim one's opening
> line is just deleted.
>
> The obvious fix — gluing numberless fragments onto whatever came before — loses claim one
> entirely, and welds separate independent claims together.
>
> There's no ground truth to test against, so I verify it differently: all ten and a half
> thousand source entries have to get used exactly once.

---

## 4 · Evaluation · 30 sec

> I built the answer key three times, because the first two gave me the wrong answer.
>
> The first used each patent's own abstract as the query — fifty-five percent word overlap
> with the target, same lawyer wrote both. That's duplicate detection, and BM25 wins it.
>
> The real problem was the key only counted the query's *own* patent as relevant. So when
> the system correctly found a different patent covering the same mechanism, that scored as
> a false positive. I was penalising it for working.
>
> Fixed with TREC pooling, hybrid beats BM25 on recall by fifty-two percent — p of nought
> point nought nought nought one.

---

## 5 · Scale · 15 sec

> At ten million patents you can't just run a script. Every patent gets a row tracking
> where it got to, so re-running does nothing if nothing changed, and a dead worker's jobs
> get picked back up.
>
> What actually bites is vector memory — one point eight terabytes. Quantise and truncate,
> and it's a hundred and fifty gig.

---

## Close · 10 sec

> Two caveats. The relevance grading is an LLM, not a real examiner — so comparing the
> systems is solid, absolute scores are flattering. And six hundred and forty patents is a
> proof of concept.

---

**≈ 2:00.** Over? Cut the second paragraph of section 2.

---

# Live demo commands

```bash
# search working, with the filter the brief names
python scripts/search.py --method hybrid \
  --query "a wheel spoke made of carbon fibre bonded to the rim" \
  --classification-prefix B60B --top-k 5

# why hybrid — phrased like a person, not a patent lawyer
python scripts/search.py --method bm25   --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3
python scripts/search.py --method hybrid --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3

# claim reconstruction, including the invariant
python -m pytest tests/test_reconstruct.py -q

# latency with and without pre-filters (Part 1 deliverable)
python scripts/benchmark.py --methods bm25 hybrid --repeats 2

# ingestion is idempotent (Part 2 POC)
python scripts/ingest.py --enqueue
```

UI on `http://localhost:5173`, API docs on `http://localhost:8000/docs`.

---

# Questions you'll probably get

Reference, not script.

**Why OpenSearch?**
> Lexical scoring, vector similarity and metadata filtering in one query sharing the same
> filters. Split that across a keyword store and a vector DB and you've got two round-trips
> and two filter implementations that can disagree.

**What else did you look at?**
> Elasticsearch is basically the same — OpenSearch is the Apache-2.0 fork. pgvector's fine
> if you already run Postgres, but BM25 through tsvector is weak on long technical text.
> Pinecone, Qdrant, Weaviate have better vector ergonomics but you still need a lexical
> engine next to them, and BM25 was a strong baseline here. Vespa's the best technical fit
> but a lot to operate. FAISS is a library, not a search engine.

**Why RRF rather than weighting the scores?**
> BM25 is unbounded, cosine is zero to one. Combining them means normalising and that
> drifts as the corpus changes. RRF only uses rank so there's no weight to tune. Trade-off
> is you throw away score magnitude.

**Why claims, not whole patents?**
> Claims are the legal unit, and a forty-six claim patent indexed as one blob matches
> nearly anything in its field. This way every result points at the record that matched.

**Missing fields?**
> Nothing's actually missing. Only gap is descriptions — a hundred and nineteen patents
> have descriptions that exist but are entirely blank. I keep and flag those rather than
> dropping them, since they're still searchable through claims and abstract.

**Why not parse the full classification code?**
> They're packed — B60B1110FI could be eleven-ten or one-one-ten, no delimiter. Section,
> class and subclass come out cleanly so I stop there. Guessing gives you filters that
> quietly return the wrong patents.

**Why learning-to-rank and not fine-tuning a cross-encoder?**
> Trains in seconds on CPU, scores in microseconds instead of ninety milliseconds, and it's
> what production search does. The catch is it only sees retrieval features — it never
> reads the text.

**Are the numbers actually good?**
> The comparison is solid — same answer key for every system. Absolute values I'd take with
> salt: pooling only judges what some system retrieved, the LLM assessor graded generously,
> and in a corpus that's all wheels and tyres almost anything looks related.

**What's not finished?**
> Cross-encoder reranking is wired but unmeasured. `build_index.py` holds all embeddings in
> memory — fine at nineteen thousand records, won't scale. No auth on the API.

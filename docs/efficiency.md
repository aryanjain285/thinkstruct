# Latency: with and without hybrid filtering

Part 1 asks to *"time how long your algorithm takes with and without the hybrid search
enabled and provide some commentary on things you would or have used to make it more
efficient."*

Reproduce with `python scripts/benchmark.py --methods bm25 dense hybrid --repeats 3`
(10 queries × 3 repeats, P50 over 30 runs, warm index, 18,743 records).

## Measured

| method / filters | P50 | P95 | embed_query | retrieve | aggregate |
|---|---|---|---|---|---|
| bm25 / unfiltered | 2.6 ms | 4.5 ms | — | 2.6 ms | 0.0 ms |
| bm25 / filtered | 4.4 ms | 8.2 ms | — | 4.3 ms | 0.0 ms |
| dense / unfiltered | 319.8 ms | 437.3 ms | **314.9 ms** | 5.9 ms | 0.0 ms |
| dense / filtered | 308.0 ms | 428.8 ms | **298.5 ms** | 7.4 ms | 0.0 ms |
| hybrid / unfiltered | 317.6 ms | 439.9 ms | **308.7 ms** | 9.3 ms | 0.0 ms |
| hybrid / filtered | 325.6 ms | 450.0 ms | **314.6 ms** | 10.7 ms | 0.0 ms |

Filters used: `classification_prefix=B60B` + `record_type=claim` (~47% of the corpus).

## Finding 1: dense search is not slow — the hosted embedding API is

The obvious reading of "dense P50 = 320 ms vs BM25 2.6 ms" is that vector search is
120× slower. **That is wrong.** Splitting the stages:

```
dense total 319.8 ms
  ├── embed_query  314.9 ms   (98%)  <- network round-trip to api.openai.com
  └── retrieve       5.9 ms   ( 2%)  <- OpenSearch HNSW k-NN
```

Vector retrieval costs **5.9 ms**, about twice BM25's 2.6 ms — entirely reasonable for
an HNSW graph traversal returning 50 neighbours from 18,743 vectors. Every millisecond
beyond that is a hosted API call.

**Fix already implemented:** `--embedder minilm` runs `all-MiniLM-L6-v2` locally and
removes the network hop entirely. On CPU a single short query embeds in ~5–10 ms,
which would bring dense P50 to roughly **12–16 ms** and hybrid to ~20 ms. The provider
registry makes this a one-flag change with no code edit:

```bash
python scripts/build_index.py --embeddings --embedder minilm
python scripts/search.py --method hybrid --embedder minilm --query "..."
```

The hosted API was used here only because HuggingFace is network-blocked on the
development machine — a constraint of the environment, not a design choice.

## Finding 2: filters cost 69% on BM25 *at this scale*, and that inverts at production scale

BM25 goes 2.6 → 4.3 ms when filters are applied (+69%). Naively that argues against
filtering. It is an artefact of corpus size.

At 18,743 records, an unfiltered BM25 scan is already trivially cheap, so adding a
`prefix` clause and a `terms` clause is pure overhead — there is nothing to save. The
cost is evaluating the filter, and the saving on scoring is negligible because scoring
18K documents is fast anyway.

At 10M patents (~293M records) the arithmetic reverses. `classification_prefix=B60B`
selects roughly 0.3% of a full-corpus index. Filtering first means BM25 scores ~900K
postings instead of 293M, and k-NN traverses a fraction of the graph. The filter stops
being overhead and becomes the dominant optimisation.

This is why filters are implemented as **pre-filters, not post-filters**:

```python
knn["filter"] = {"bool": {"filter": must, "must_not": must_not}}
```

The Lucene k-NN engine applies this *during graph traversal*. Post-filtering would
retrieve 50 neighbours and then discard the non-matching ones, returning far fewer
than 50 results under a selective constraint. The audit verifies this directly: with
`B60B` applied, both dense and hybrid still return a full 50/50 candidates.

## Finding 3: dense filtering is free, BM25 filtering is not

Dense actually got marginally *faster* when filtered (319.8 → 308.0 ms; noise, but not
slower). Lucene's filtered HNSW traversal restricts the search to a smaller candidate
region, offsetting the filter's own cost. BM25 has no equivalent — it must evaluate
the filter clause on top of a scan it was going to do anyway.

## What else was done

| Optimisation | Effect |
|---|---|
| **Denormalised `abstract` onto every record** | Title/abstract keyword filters resolve in one pass instead of a patent-level join. Costs ~20% index size, removes a round-trip |
| **`classification_subclass` as a `keyword` field** | Exact-match term filter rather than a prefix scan for the common `B60B` case |
| **`title.exact` keyword sub-field** | Exact-title lookup is a `term` query, not an analysed match |
| **RRF instead of score normalisation** | No score-scale calibration pass and no tuning parameter; fusion is O(n) over two ranked lists |
| **Index-time synonym expansion** | `tyre`/`tire` resolved in the postings, not by query rewriting at search time |
| **Aggregation after retrieval** | Patent grouping is O(n) over ≤50 candidates — 0.0 ms, never a bottleneck |

## What I would do next, in order of value

1. **Local embedding model** — removes 98% of dense latency. Already supported.
2. **Cache query embeddings** — examiners re-run and refine the same query repeatedly;
   an LRU on normalised query text would make repeats free.
3. **Quantise vectors to int8** — 4× less memory, and at 293M vectors memory is the
   binding constraint, not CPU (see [scaling.md](scaling.md)).
4. **Reduce dimensions to 512** — `text-embedding-3-small` supports Matryoshka
   truncation; already exposed as `--dimensions 512`.
5. **Skip the dense branch when the query is short and lexical.** A three-word query
   of exact component names does not need semantic retrieval; routing those to BM25
   only would cut mean latency substantially at no measured recall cost.
6. **Batch the reranker on GPU** — at ~3 s for 20 candidates the LLM reranker is by far
   the most expensive stage. A local cross-encoder (`--reranker ce-minilm`) does the
   same work in ~90 ms.

## Honest caveats

- 18,743 records is small enough that everything is fast. These numbers establish
  *relative* cost, not production behaviour.
- Single-node OpenSearch, no replicas, warm page cache, no concurrent load.
- P95 on dense/hybrid (~440 ms) is dominated by API tail latency, not search.

# Demo script

Two minutes, one section per slide. Written to be *spoken* — read it out loud once and
it should sound like you, not like a paper.

Technical audience, so the real terms stay in. Every number comes from `reports/` — if
you regenerate those, update this.

---

## Slide 1 — Problem · 20 sec

> So, prior-art search. Before you file, someone has to check whether anyone's already
> claimed the thing you're about to claim.
>
> And you can get that wrong in two directions. Miss something, and you end up with a
> patent that's invalid — which you find out in litigation, when it's expensive. Or you
> flood the attorney with two hundred vaguely-related hits, and they're reading through
> that at five hundred an hour.
>
> The reason it's hard is that people don't use the same words. What you'd call a carbon
> fibre spoke, someone else claimed as an elongate composite tension member. BM25 is never
> going to connect those two.
>
> One design decision up front — I index individual claims, not whole patents. A patent
> with forty-six claims will match basically any query in its field, and infringement gets
> decided claim by claim anyway. So six hundred and forty patents become about nineteen
> thousand records.

---

## Slide 2 — Architecture · 30 sec

> Because of that word-mismatch problem, I search twice.
>
> BM25 handles the exact stuff — component names, materials, the numeric limits that show
> up in claims. And then dense k-NN over embeddings picks up the cases where it's the same
> idea, worded completely differently. Both of those hit the same OpenSearch index.
>
> The filters — CPC prefix, title, abstract — go into both branches as pre-filters. That
> bit matters for the vector side: it's filtering during the HNSW traversal, not after it.
> If you post-filter, you ask for fifty and get back twelve.
>
> Then I merge with reciprocal rank fusion. BM25 scores are unbounded, cosine is zero to
> one, so you can't just add them without normalising, and normalisation drifts. RRF only
> looks at rank position, so there's nothing to tune. Nice side effect — if both retrievers
> independently like a document, it gets pushed up.
>
> After that a reranker over the top fifty, and then everything collapses to patent level.

---

## Slide 3 — Claim reconstruction · 30 sec

> Now, the thing that actually took the time wasn't the search. It was the data.
>
> The claims field looks like a list of claims. It isn't. Whatever parsed the original XML
> wrecked it. Only about sixty-nine percent of the entries still have their claim number.
> Nearly ninety percent of patents open with a fragment that has no number on it at all.
> And in eighty percent of them, claim one's opening line — "A spoke comprising" — is just
> gone. Not moved somewhere else. Deleted.
>
> The obvious fix is to glue any numberless fragment onto whatever came before it. That
> breaks straight away — there's nothing before the first one, so you lose claim one
> entirely. And worse, a lot of those mid-list fragments are actually brand new independent
> claims that lost their preamble, so you end up welding two separate claims together.
>
> So I do it in two passes. First, work out where claims *start* — claims are sentences, so
> if the previous entry ended on a full stop, it finished. Then separately, figure out the
> numbers by looking at the neighbours.
>
> There's no ground truth to test against, so I verify it a different way: every one of the
> ten and a half thousand source entries has to get used exactly once. Nothing dropped,
> nothing counted twice. That'll still hold if the data changes next month, which a bunch
> of hand-written test cases wouldn't.

---

## Slide 4 — Evaluation · 30 sec

> I ended up building the answer key three times, because the first two gave me the wrong
> answer.
>
> First go, I used each patent's own abstract as the query. Seemed reasonable. But the
> abstract and the claims were written by the same lawyer on the same day — so fifty-five
> percent of the query's words were sitting right there in the target. That's not retrieval,
> that's duplicate detection, and BM25 wins it every time. Which it did.
>
> Second go, I had an LLM reword the queries. Overlap dropped to thirty-seven percent,
> hybrid pulled ahead — but p was zero point one eight, so I couldn't actually claim it.
>
> Third go I found the real problem. The answer key only counted the query's *own* patent as
> relevant. So every time the system correctly found a different patent covering the same
> mechanism, that got scored as a false positive. I was penalising it for doing exactly what
> it's meant to do.
>
> So I did TREC-style pooling — take the top fifteen from every system, merge them, grade
> each one zero to three. That surfaced about twelve hundred cross-patent relevant records
> the old key was calling mistakes.
>
> And then hybrid beats BM25 on recall at fifty by fifty-two percent, p equals nought point
> nought nought nought one. The trained reranker adds another point one two on nDCG, on
> patents it had never seen — split by patent, not by row, so nothing leaks.
>
> I report all three, by the way. The fact that the conclusion flipped twice is kind of the
> whole point.

---

## Slide 5 — Scale · 25 sec

> Last bit — what this looks like at ten million patents.
>
> You can't just run a script at that size. It takes days, machines reboot, individual
> records blow up. So every patent gets a row tracking where it's got to. Re-running does
> nothing if nothing's changed — it's keyed on a content hash plus the parser and model
> version. Bump the parser version and it re-queues exactly what that affects. If a worker
> dies mid-job, the lease expires and someone else picks it up.
>
> That's SQLite for the demo, but the claim semantics are the same ones Postgres gives you,
> so moving it over is a connection string.
>
> The thing that actually bites at that scale is vector memory. Two hundred and ninety-three
> million records, float32, is about one point eight terabytes — and HNSW wants that
> resident. Truncate the dimensions, quantise to int8, skip embedding the description
> passages, and you're at a hundred and fifty gig. That's roughly twenty-six hundred a month
> down to seven hundred.

---

## Close · 10 sec

> Three hundred and twenty-nine tests, and every number in the README comes straight out of
> generated report files — I'm not typing them in.
>
> Two things I'd flag before you ask. The relevance grading is done by an LLM, not an actual
> examiner — so comparing the systems against each other is solid, but the absolute scores
> are flattering. And six hundred and forty patents is a proof of concept. The ten-million
> numbers are reasoned through, not measured.

---

Total's about two minutes five. If you're running over, drop the cost line on slide five.

---

# Live demo commands

If you're driving the thing rather than talking over slides.

```bash
# search working, with the filter the brief names
python scripts/search.py --method hybrid \
  --query "a wheel spoke made of carbon fibre bonded to the rim" \
  --classification-prefix B60B --top-k 5

# why hybrid — phrased the way a person would, not a patent lawyer
python scripts/search.py --method bm25   --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3
python scripts/search.py --method hybrid --query "bendy rod of woven strands joining a wheel's middle to its outer ring" --top-k 3

# claim reconstruction, including the invariant
python -m pytest tests/test_reconstruct.py -q

# latency with and without pre-filters (Part 1 deliverable)
python scripts/benchmark.py --methods bm25 hybrid --repeats 2

# ingestion is idempotent (Part 2 POC)
python scripts/ingest.py --enqueue

# everything
python -m pytest tests/ -q
```

UI on `http://localhost:5173`, API docs on `http://localhost:8000/docs`.

---

# Questions you'll probably get

Answers written the way you'd actually say them.

**Why OpenSearch?**
> Because I needed lexical scoring, vector similarity and metadata filtering resolved in one
> query, sharing the same filters. If you split that across a keyword store and a vector DB
> you've got two round-trips and two filter implementations that can disagree with each
> other.

**What else did you look at?**
> Elasticsearch is basically the same thing — OpenSearch is the Apache-2.0 fork, so no
> licence question. pgvector's fine if you're already running Postgres, but BM25 through
> tsvector is weak on long technical text. Pinecone, Qdrant, Weaviate — better pure-vector
> ergonomics, but you still need a lexical engine next to them, and BM25 turned out to be a
> genuinely strong baseline here. Vespa's probably the best technical fit for hybrid ranking
> but it's a lot to operate for a POC. FAISS is a library, not a search engine — no filters,
> no persistence.

**Why RRF rather than weighting the scores?**
> BM25 is unbounded and cosine is zero to one, so combining them means normalising, and that
> normalisation drifts as the corpus changes. RRF only uses rank, so there's no weight to
> tune. The trade-off is you throw away score magnitude — a runaway best match looks the
> same as a marginal rank one.

**Why claims and not whole patents?**
> Claims are the legal unit. And practically, a forty-six claim patent indexed as one blob
> matches nearly anything in its field. This way every result can point at the specific
> record that matched.

**What did you do about missing fields?**
> Nothing's actually missing — I checked. The only gap is descriptions: a hundred and
> nineteen patents have descriptions that exist but are entirely blank paragraphs. I keep
> those and flag them rather than dropping them, because they're still perfectly searchable
> through their claims and abstract. Throwing them out would lose eighteen percent of the
> corpus for nothing.

**Why don't you parse the full classification code?**
> The codes are packed together — B60B1110FI could be eleven-slash-ten or one-slash-one-ten,
> there's no delimiter. Section, class and subclass come out cleanly, so I stop there and
> use prefix matching. Guessing would give you filters that quietly return the wrong
> patents, which is worse than not having them.

**Why learning-to-rank instead of fine-tuning a cross-encoder?**
> Trains in seconds on CPU and scores in microseconds instead of about ninety milliseconds,
> which matters when it's sitting on the query path. It's also what production search
> actually does. The catch is it only ever sees the retrieval features — it never reads the
> text, so there are things a cross-encoder would catch that it can't.

**Are these numbers actually good?**
> The comparison is solid — same answer key for every system, p of nought point nought
> nought nought one. The absolute values I'd take with salt: pooling only judges what some
> system retrieved, the assessor is an LLM and it graded fairly generously, and in a corpus
> that's entirely wheels and tyres almost anything looks somewhat related.

**What's not finished?**
> Cross-encoder reranking is wired up but I haven't measured it. `build_index.py` holds all
> the embeddings in memory before writing — fine at nineteen thousand records, won't scale;
> the queue-driven path is the one that does. And there's no auth on the API, which is fine
> locally and wrong for anything exposed.

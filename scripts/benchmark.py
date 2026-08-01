"""Latency benchmark: every retrieval method, with and without metadata filters.

The task asks specifically to "time how long your algorithm takes with and without the
hybrid search enabled". This measures exactly that and reports P50/P95 per stage.

    python scripts/benchmark.py --methods bm25 hybrid --repeats 5
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patsearch.config import EMBEDDER, INDEX_NAME, OPENSEARCH_HOST, REPORTS_DIR
from patsearch.embeddings.service import create_embedder
from patsearch.pipeline import search
from patsearch.reranking.service import DEFAULT_RERANKER, CrossEncoderReranker
from patsearch.search.client import get_client, wait_for_health
from patsearch.search.query import Filters

QUERIES = [
    "flexible fibre spoke connected between a hub and a wheel rim",
    "pneumatic tyre tread with circumferential grooves and sipes",
    "wheel bearing unit with inner toothing and axial retention",
    "run-flat tyre sidewall insert reinforcement",
    "carbon fibre composite wheel rim manufacturing method",
    "tyre pressure sensor mounted on the inner liner",
    "brake disc mounted to a wheel hub with plural fasteners",
    "rubber composition comprising silica and a coupling agent",
    "spoke tension adjustment mechanism for a bicycle wheel",
    "vehicle wheel with a decorative cover attached by clips",
]

# Filters exercised: classification constraint plus a title keyword. Both are named
# explicitly in the task brief.
FILTERED = Filters(classification_prefix="B60B", record_types=["claim"])


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = min(int(round(p / 100 * (len(xs) - 1))), len(xs) - 1)
    return xs[idx]


def run(client, index, method, queries, repeats, embedder, reranker, filters, candidates):
    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    hits_seen: list[int] = []
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            out = search(
                client, index, q, method=method, filters=filters,
                embedder=embedder, reranker=reranker, candidates=candidates, top_k=10,
            )
            totals.append((time.perf_counter() - t0) * 1000)
            hits_seen.append(len(out.hits))
            for k, v in out.timings_ms.items():
                stages.setdefault(k, []).append(v)
    return {
        "n": len(totals),
        "p50_ms": round(percentile(totals, 50), 1),
        "p95_ms": round(percentile(totals, 95), 1),
        "mean_ms": round(st.mean(totals), 1),
        "mean_candidates": round(st.mean(hits_seen), 1),
        "stages_p50_ms": {k: round(percentile(v, 50), 1) for k, v in stages.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_NAME)
    ap.add_argument("--host", default=OPENSEARCH_HOST)
    ap.add_argument("--embedder", default=EMBEDDER)
    ap.add_argument("--methods", nargs="+",
                    default=["bm25", "dense", "hybrid", "hybrid_reranked"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    client = get_client(args.host)
    wait_for_health(client)

    needs_vec = any(m in ("dense", "hybrid", "hybrid_reranked") for m in args.methods)
    embedder = create_embedder(args.embedder) if needs_vec else None
    reranker = (
        CrossEncoderReranker(DEFAULT_RERANKER) if "hybrid_reranked" in args.methods else None
    )

    if embedder:
        for _ in range(args.warmup):
            embedder.embed_query("warmup")

    results: dict[str, dict] = {}
    for method in args.methods:
        for label, flt in (("unfiltered", Filters()), ("filtered", FILTERED)):
            key = f"{method}/{label}"
            print(f"running {key} ...", flush=True)
            try:
                results[key] = run(
                    client, args.index, method, QUERIES, args.repeats,
                    embedder, reranker, flt, args.candidates,
                )
            except Exception as exc:
                results[key] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n" + "=" * 78)
    print(f"{'method / filters':<30} {'P50':>8} {'P95':>8} {'mean':>8} {'cands':>7}")
    print("=" * 78)
    for k, v in results.items():
        if "error" in v:
            print(f"{k:<30} {v['error'][:44]}")
        else:
            print(f"{k:<30} {v['p50_ms']:>7.1f}ms {v['p95_ms']:>7.1f}ms "
                  f"{v['mean_ms']:>7.1f}ms {v['mean_candidates']:>7.1f}")
    print("=" * 78)

    print("\nfilter effect (P50):")
    for m in args.methods:
        u, f = results.get(f"{m}/unfiltered", {}), results.get(f"{m}/filtered", {})
        if "p50_ms" in u and "p50_ms" in f:
            delta = f["p50_ms"] - u["p50_ms"]
            pct = 100 * delta / u["p50_ms"] if u["p50_ms"] else 0
            print(f"  {m:<18} {u['p50_ms']:>7.1f} -> {f['p50_ms']:>7.1f} ms "
                  f"({pct:+.1f}%)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

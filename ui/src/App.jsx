import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";

const RECORD_TYPES = ["claim", "abstract", "summary", "description"];

function StatusBar({ health, capabilities }) {
  if (!health) {
    return (
      <div className="status-bar">
        <span><span className="dot bad" />API unreachable</span>
      </div>
    );
  }
  return (
    <div className="status-bar">
      <span><span className="dot ok" />OpenSearch <b>{health.opensearch}</b></span>
      <span>index <b>{health.index}</b></span>
      <span><b>{health.documents.toLocaleString()}</b> records</span>
      <span>
        vectors{" "}
        <b>{health.vector_search_available ? health.embedder : "unavailable"}</b>
      </span>
      {capabilities?.active_reranker && <span>reranker <b>{capabilities.active_reranker}</b></span>}
    </div>
  );
}

function SearchForm({ methods, onSubmit, loading }) {
  const [form, setForm] = useState({
    query: "flexible fibre spoke connected between a hub and a wheel rim",
    method: "hybrid",
    top_k: 10,
    candidates: 50,
    classification_prefix: "B60B",
    title_keyword: "",
    abstract_keyword: "",
    exact_title: "",
    record_types: [],
    independent_only: false,
  });

  useEffect(() => {
    if (methods.length && !methods.includes(form.method)) {
      setForm((f) => ({ ...f, method: methods[0] }));
    }
  }, [methods]); // eslint-disable-line react-hooks/exhaustive-deps

  const set = (k) => (e) => {
    const v = e.target.type === "checkbox" ? e.target.checked : e.target.value;
    setForm((f) => ({ ...f, [k]: v }));
  };

  const submit = (e) => {
    e.preventDefault();
    if (!form.query.trim()) return;
    onSubmit({
      ...form,
      top_k: Number(form.top_k),
      candidates: Number(form.candidates),
      classification_prefix: form.classification_prefix || null,
      title_keyword: form.title_keyword || null,
      abstract_keyword: form.abstract_keyword || null,
      exact_title: form.exact_title || null,
    });
  };

  return (
    <form className="search panel" onSubmit={submit}>
      <label className="field">
        Query — natural language, or paste a full claim
        <textarea value={form.query} onChange={set("query")} />
      </label>

      <div className="grid">
        <label className="field">
          Method
          <select value={form.method} onChange={set("method")}>
            {methods.map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <label className="field">
          Classification prefix
          <input value={form.classification_prefix} onChange={set("classification_prefix")}
                 placeholder="B60B" />
        </label>
        <label className="field">
          Title keyword
          <input value={form.title_keyword} onChange={set("title_keyword")} placeholder="wheel" />
        </label>
        <label className="field">
          Abstract keyword
          <input value={form.abstract_keyword} onChange={set("abstract_keyword")}
                 placeholder="carbon" />
        </label>
        <label className="field">
          Exact title
          <input value={form.exact_title} onChange={set("exact_title")} placeholder="SPOKE" />
        </label>
        <label className="field">
          Record type
          <select
            value={form.record_types[0] || ""}
            onChange={(e) =>
              setForm((f) => ({ ...f, record_types: e.target.value ? [e.target.value] : [] }))
            }
          >
            <option value="">any</option>
            {RECORD_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="field">
          Results
          <input type="number" min="1" max="50" value={form.top_k} onChange={set("top_k")} />
        </label>
        <label className="field">
          Candidates
          <input type="number" min="1" max="200" value={form.candidates}
                 onChange={set("candidates")} />
        </label>
      </div>

      <label style={{ fontSize: 12, color: "var(--muted)" }}>
        <input type="checkbox" style={{ width: "auto", marginRight: 6 }}
               checked={form.independent_only} onChange={set("independent_only")} />
        independent claims only
      </label>

      <button className="primary" type="submit" disabled={loading}>
        {loading ? <><span className="spinner" /> searching…</> : "Search"}
      </button>
    </form>
  );
}

function Result({ item, rank, onOpen }) {
  const b = item.best_match;
  const where = b.claim_number ? `claim ${b.claim_number}` : b.record_type;
  return (
    <div className="result">
      <div className="result-head">
        <span className="rank">{rank}.</span>
        <h3>{item.title || "(untitled)"}</h3>
        <span className="pid" onClick={() => onOpen(item.patent_id)}>{item.patent_id}</span>
        <span className="badge">{item.classification}</span>
        <span className="badge">{item.score.toFixed(4)}</span>
        {b.bm25_rank && <span className="badge">bm25 #{b.bm25_rank}</span>}
        {b.vector_rank && <span className="badge">vec #{b.vector_rank}</span>}
        {b.rerank_score != null && <span className="badge">rerank {b.rerank_score}</span>}
      </div>
      <p className="snippet"><b>{where}</b> — {b.text.slice(0, 320)}
        {b.text.length > 320 ? "…" : ""}</p>
      {item.supporting?.length > 0 && (
        <details className="supporting">
          <summary>{item.supporting.length} supporting record(s)</summary>
          <ul>
            {item.supporting.map((s) => (
              <li key={s.record_id}>
                <b>{s.claim_number ? `claim ${s.claim_number}` : s.record_type}</b>{" "}
                — {s.text.slice(0, 190)}{s.text.length > 190 ? "…" : ""}
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

function PatentDetail({ id, onBack }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setData(null);
    setError(null);
    api.patent(id).then(setData).catch((e) => setError(e.message));
  }, [id]);

  if (error) return <div className="error">{error}</div>;
  if (!data) return <div className="empty"><span className="spinner" /> loading…</div>;

  return (
    <div>
      <button className="primary" onClick={onBack} style={{ marginBottom: 14 }}>
        ← back to results
      </button>
      <div className="panel">
        <h2 style={{ marginTop: 0, fontSize: 17 }}>{data.title}</h2>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
          <span className="badge">{data.patent_id}</span>
          <span className="badge">{data.classification_raw}</span>
          <span className="badge">{data.claims.length} claims</span>
          {!data.has_description && <span className="badge">no description</span>}
        </div>
        <p style={{ color: "var(--muted)", fontSize: 13 }}>{data.abstract}</p>

        <h3 style={{ fontSize: 14, marginTop: 22 }}>Claims</h3>
        {data.claims.map((c) => (
          <div key={c.claim_number} className={`claim ${c.is_independent ? "independent" : ""}`}>
            <div className="claim-meta">
              <span>claim {c.claim_number}</span>
              <span>{c.is_independent ? "independent" : `depends on ${c.depends_on.join(", ")}`}</span>
              <span>{c.status}</span>
              {c.number_inferred && <span title="claim number was inferred during reconstruction">
                number inferred</span>}
            </div>
            <div className="claim-text">{c.text}</div>
          </div>
        ))}

        {data.description_paragraphs.length > 0 && (
          <details style={{ marginTop: 20 }}>
            <summary style={{ cursor: "pointer", color: "var(--accent)", fontSize: 13 }}>
              Description ({data.description_paragraphs.length} paragraphs)
            </summary>
            {data.description_paragraphs.map((p, i) => (
              <p key={i} style={{ color: "var(--muted)", fontSize: 13 }}>{p}</p>
            ))}
          </details>
        )}
      </div>
    </div>
  );
}

function Stats() {
  const [s, setS] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => { api.stats().then(setS).catch((e) => setError(e.message)); }, []);
  if (error) return <div className="error">{error}</div>;
  if (!s) return <div className="empty"><span className="spinner" /></div>;
  return (
    <div className="panel">
      <table className="stats">
        <tbody>
          <tr><td>Patents</td><td>{s.patents.toLocaleString()}</td></tr>
          <tr><td>Indexed records</td><td>{s.documents.toLocaleString()}</td></tr>
          {Object.entries(s.by_record_type).map(([k, v]) => (
            <tr key={k}><td style={{ paddingLeft: 26, color: "var(--muted)" }}>{k}</td>
              <td>{v.toLocaleString()}</td></tr>
          ))}
          {Object.entries(s.classifications).map(([k, v]) => (
            <tr key={k}><td>{k}</td><td>{v.toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("search");
  const [health, setHealth] = useState(null);
  const [caps, setCaps] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [openPatent, setOpenPatent] = useState(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    api.capabilities().then(setCaps).catch(() => setCaps(null));
  }, []);

  const doSearch = useCallback(async (body) => {
    setLoading(true);
    setError(null);
    setOpenPatent(null);
    try {
      setResults(await api.search(body));
    } catch (e) {
      setError(e.message);
      setResults(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const methods = caps?.methods ?? ["bm25"];

  return (
    <div className="app">
      <header>
        <h1>Patent claim search</h1>
        <span className="sub">hybrid BM25 + dense retrieval over USPTO vehicle applications</span>
      </header>
      <StatusBar health={health} capabilities={caps} />

      <div className="tabs">
        {["search", "stats"].map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {tab === "stats" && <Stats />}

      {tab === "search" && (openPatent ? (
        <PatentDetail id={openPatent} onBack={() => setOpenPatent(null)} />
      ) : (
        <>
          <SearchForm methods={methods} onSubmit={doSearch} loading={loading} />
          {error && <div className="error">{error}</div>}
          {results && (
            <>
              <div className="timings">
                {Object.entries(results.timings_ms).map(([k, v]) => (
                  <span key={k} className="chip">{k} {v.toFixed(0)}ms</span>
                ))}
                <span className="chip total">total {results.total_ms.toFixed(0)}ms</span>
                <span className="chip">{results.candidates_retrieved} candidates →{" "}
                  {results.results.length} patents</span>
              </div>
              {results.results.length === 0 && <div className="empty">No matches.</div>}
              {results.results.map((r, i) => (
                <Result key={r.patent_id} item={r} rank={i + 1} onOpen={setOpenPatent} />
              ))}
            </>
          )}
        </>
      ))}
    </div>
  );
}

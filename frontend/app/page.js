"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://dwin2020-hp-rag-backend.hf.space";

async function postJSON(path, question) {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error(`Server responded ${res.status}`);
  return res.json();
}

/* ------------------------------- HP Q&A -------------------------------- */
function FilmsPanel() {
  const [q, setQ] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function ask(e) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await postJSON("/query", q));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card">
      <h2>⚡ Films &amp; Books Q&amp;A</h2>
      <p className="sub">
        Ask about the Harry Potter films and books. Answers are grounded in Wikipedia
        with a self-correcting retrieval loop.
      </p>
      <form onSubmit={ask}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. How does the Deathly Hallows book end?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !q.trim()}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>

      {error && <p className="error">⚠️ {error}</p>}
      {result && (
        <>
          <div className="answer">{result.answer}</div>
          <p className="meta">
            search query: “{result.final_query}” · relevance: {result.relevance} ·
            attempts: {result.attempts}
          </p>
        </>
      )}
    </section>
  );
}

/* ----------------------------- Research -------------------------------- */
const STATUS = {
  answered: { mark: "✓", cls: "ok" },
  thin: { mark: "~", cls: "thin" },
  failed: { mark: "✗", cls: "fail" },
};

function TraceEvent({ e }) {
  if (e.type === "plan") {
    return (
      <div className="ev plan">
        <div className="ev-head">◆ PLAN <span className="rnd">round {e.round}</span></div>
        <ol className="subqs">
          {e.sub_questions.map((s, i) => <li key={i}>{s}</li>)}
        </ol>
      </div>
    );
  }
  if (e.type === "research") {
    const s = STATUS[e.status] || { mark: "?", cls: "" };
    return (
      <div className={`ev research ${s.cls}`}>
        <div className="ev-head">
          <span className="mark">{s.mark}</span> {e.sub_question}
          <span className="rnd">r{e.round} · {e.n_sources} src · {e.status}</span>
        </div>
        <p className="ev-body">{e.summary}</p>
      </div>
    );
  }
  if (e.type === "gap") {
    return (
      <div className="ev gap">
        <div className="ev-head">
          ◆ GAP ANALYSIS <span className="rnd">round {e.round}</span>
          <span className={`badge ${e.complete ? "ok" : "thin"}`}>
            {e.complete ? "COMPLETE" : "GAPS FOUND"}
          </span>
        </div>
        <p className="ev-body dim">{e.reasoning}</p>
        {e.gaps?.map((g, i) => <p key={i} className="missing">↳ {g}</p>)}
      </div>
    );
  }
  if (e.type === "revise") {
    return (
      <div className="ev revise">
        <div className="ev-head">↻ CHANGED DIRECTION → round {e.round} · {e.pursuing.length} new lead(s)</div>
      </div>
    );
  }
  if (e.type === "synthesize") {
    return (
      <div className="ev synth">
        <div className="ev-head">◆ SYNTHESIZE <span className="rnd">round {e.round}</span></div>
      </div>
    );
  }
  return null;
}

function ResearchPanel() {
  const [q, setQ] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function ask(e) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      setResult(await postJSON("/research", q));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="card research-card">
      <h2>🔎 Deep Research Agent</h2>
      <p className="sub">
        Ask an open research question. The agent plans sub-questions, searches the web,
        critiques its own findings, and revises — showing every decision it made.
      </p>
      <form onSubmit={ask}>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="e.g. How did the EU's AI Act influence US state-level AI legislation?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !q.trim()}>
          {loading ? "Researching…" : "Research"}
        </button>
      </form>

      {loading && (
        <p className="meta pulse">Planning, searching and self-revising — this can take up to a minute…</p>
      )}
      {error && <p className="error">⚠️ {error}</p>}

      {result && (
        <>
          <div className="trace">
            <div className="trace-title">Reasoning trace</div>
            {result.trace.map((e, i) => <TraceEvent key={i} e={e} />)}
          </div>
          <div className="answer">{result.answer}</div>
        </>
      )}
    </section>
  );
}

export default function Home() {
  return (
    <main className="wrap">
      <header className="hero">
        <h1>Harry Potter RAG <span className="amp">×</span> Research</h1>
        <p className="tagline">A self-correcting retrieval agent and a self-improving research agent.</p>
      </header>
      <FilmsPanel />
      <ResearchPanel />
      <footer className="foot">
        LangGraph · Chroma · Groq · DuckDuckGo · deployed on Hugging Face + Vercel
      </footer>
    </main>
  );
}

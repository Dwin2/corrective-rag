"use client";

import { useState } from "react";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "https://dwin2020-hp-rag-backend.hf.space";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function ask(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const res = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      if (!res.ok) throw new Error(`Server responded ${res.status}`);
      setResult(await res.json());
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="wrap">
      <h1>⚡ Harry Potter Films Q&A</h1>
      <p className="sub">
        Ask anything about the Harry Potter films. Answers are grounded in Wikipedia.
      </p>

      <form onSubmit={ask}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g. Who directed Prisoner of Azkaban?"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !question.trim()}>
          {loading ? "Asking…" : "Ask"}
        </button>
      </form>

      {error && <p className="error">⚠️ {error}</p>}

      {result && (
        <>
          <div className="answer">{result.answer}</div>
          <p className="meta">
            Final search query: “{result.final_query}” · relevance:{" "}
            {result.relevance} · attempts: {result.attempts}
          </p>
        </>
      )}
    </main>
  );
}

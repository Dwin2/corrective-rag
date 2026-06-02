---
title: Harry Potter Films RAG API
emoji: ⚡
colorFrom: indigo
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Harry Potter Films RAG API

A self-correcting RAG agent (LangGraph) that answers questions about the Harry
Potter films, grounded in Wikipedia. Built with FastAPI + Groq + Chroma.

## Endpoints

- `GET /` — health check
- `POST /query` — body `{"question": "..."}` → `{answer, final_query, relevance, attempts}`

## Secrets

Set `GROQ_API_KEY` in the Space settings (Settings → Variables and secrets).

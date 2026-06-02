# Corrective RAG — Harry Potter Films Q&A

A self-correcting (corrective) RAG agent that answers questions about the Harry
Potter films, grounded in Wikipedia. Built with LangGraph, retrieves passages,
**grades their relevance**, and **rewrites the query and retries** when the
retrieval isn't good enough before generating a cited answer.

## Architecture

- **Backend** — FastAPI on a Hugging Face Space (Docker). The LangGraph agent:
  `retrieve → grade → (rewrite → retrieve)* → generate`. The Chroma vector index
  is pre-built locally and shipped with the Space, so it never calls Wikipedia at
  runtime. LLM: Groq (`llama-3.3-70b-versatile` for answers, `llama-3.1-8b-instant`
  for grading/rewriting, with fallback). Code in [`hf_space/`](hf_space/).
- **Frontend** — Next.js app on Vercel with a single input box. Code in
  [`frontend/`](frontend/).

The `phase0`–`phase3` scripts are the incremental build-up: a LangGraph counter,
basic retrieval, a retrieve→generate graph, and finally the corrective agent.

## Run locally

```bash
# backend
cp .env.example .env          # add your GROQ_API_KEY
python hf_space/build_index.py   # build the Chroma index (run from hf_space/)
uvicorn app:app --reload      # from hf_space/, serves POST /query

# frontend
cd frontend && npm install && npm run dev
```

Set `NEXT_PUBLIC_API_URL` for the frontend to point at your backend.

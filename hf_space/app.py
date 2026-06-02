import os
from typing_extensions import TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from groq import RateLimitError

from prompt import template
from grader import grader
from rewrite import rewrite_template
from research import build_research_agent

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION = "hp_films"

# Shared models. Big model only for user-facing answers; cheap model for the
# internal grade/rewrite/summary steps to conserve the 70b daily token quota.
big = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=2048)
small = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=1024)

_embeddings = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings


# ---------------------------------------------------------------------------
# Phase 3: corrective RAG over the pre-built Harry Potter index
# ---------------------------------------------------------------------------
class Phase3(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    relevance: str
    count: int


def build_query_agent():
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve(state):
        return {"documents": retriever.invoke(state["question"])}

    def generate(state):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = template.format(context=context, question=state["question"])
        try:
            return {"answer": big.invoke(prompt).content}
        except RateLimitError:
            return {"answer": small.invoke(prompt).content}

    def grade(state):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = grader.format(context=context, question=state["question"])
        return {"relevance": small.invoke(prompt).content}

    def rewrite(state):
        prompt = rewrite_template.format(question=state["question"])
        return {"question": small.invoke(prompt).content, "count": state["count"] + 1}

    def router(state):
        if state["count"] >= 2:
            return "finish"
        if state["relevance"] == "not_relevant":
            return "retry"
        return "finish"

    build = StateGraph(Phase3)
    build.add_node("retrieve", retrieve)
    build.add_node("generate", generate)
    build.add_node("grader", grade)
    build.add_node("rewrite", rewrite)
    build.add_edge(START, "retrieve")
    build.add_edge("retrieve", "grader")
    build.add_conditional_edges("grader", router, {"retry": "rewrite", "finish": "generate"})
    build.add_edge("rewrite", "retrieve")
    build.add_edge("generate", END)
    return build.compile()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Harry Potter RAG + Research API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_query_agent = None
_research_run = None


def get_query_agent():
    # Build lazily on first request so the server binds its port immediately.
    global _query_agent
    if _query_agent is None:
        _query_agent = build_query_agent()
    return _query_agent


def get_research_run():
    global _research_run
    if _research_run is None:
        _research_run = build_research_agent(big, small, get_embeddings())
    return _research_run


@app.get("/")
def health():
    return {"status": "ok", "query_ready": _query_agent is not None,
            "research_ready": _research_run is not None}


class Query(BaseModel):
    question: str


@app.post("/query")
def query(q: Query):
    result = get_query_agent().invoke({"question": q.question, "count": 0})
    return {
        "answer": result["answer"],
        "final_query": result["question"],
        "relevance": result["relevance"],
        "attempts": result["count"] + 1,
    }


@app.post("/research")
def research(q: Query):
    return get_research_run()(q.question)

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

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION = "hp_films"


class Phase3(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    relevance: str
    count: int


def build_agent():
    # Big model only for the final user-facing answer; cheap model for the
    # internal grade/rewrite steps to conserve the 70b daily token quota.
    model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=2048)
    helper = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=512)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the index that was pre-built and shipped with the Space (see build_index.py).
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    def retrieve(state):
        return {"documents": retriever.invoke(state["question"])}

    def generate(state):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = template.format(context=context, question=state["question"])
        try:
            return {"answer": model.invoke(prompt).content}
        except RateLimitError:
            # 70b daily quota exhausted — fall back to the smaller model so the
            # app still answers (lower quality) instead of erroring out.
            return {"answer": helper.invoke(prompt).content}

    def grade(state):
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = grader.format(context=context, question=state["question"])
        return {"relevance": helper.invoke(prompt).content}

    def rewrite(state):
        prompt = rewrite_template.format(question=state["question"])
        return {"question": helper.invoke(prompt).content, "count": state["count"] + 1}

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


app = FastAPI(title="Harry Potter Films RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_AGENT = None


def get_agent():
    # Build lazily on first request so the server binds its port immediately
    # (HF Spaces health-checks the port; heavy work at import/startup risks a timeout).
    global _AGENT
    if _AGENT is None:
        _AGENT = build_agent()
    return _AGENT


@app.get("/")
def health():
    return {"status": "ok", "ready": _AGENT is not None}


class Query(BaseModel):
    question: str


@app.post("/query")
def query(q: Query):
    result = get_agent().invoke({"question": q.question, "count": 0})
    return {
        "answer": result["answer"],
        "final_query": result["question"],
        "relevance": result["relevance"],
        "attempts": result["count"] + 1,
    }

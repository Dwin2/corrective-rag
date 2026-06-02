from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from prompt import template
from grader import grader
from rewrite import rewrite_template
import wikipedia
from dotenv import load_dotenv
load_dotenv()

wikipedia.set_user_agent("hp-rag-hackathon/1.0 (your-email@example.com)")

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=2048)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

titles = [
    # Films — your original set
    "Harry Potter and the Philosopher's Stone (film)",
    "Harry Potter and the Chamber of Secrets (film)",
    "Harry Potter and the Prisoner of Azkaban (film)",
    "Harry Potter and the Goblet of Fire (film)",
    "Harry Potter and the Order of the Phoenix (film)",
    "Harry Potter and the Half-Blood Prince (film)",
    "Harry Potter and the Deathly Hallows – Part 1",
    "Harry Potter and the Deathly Hallows – Part 2",
    "Harry Potter (film series)",

    # Main characters — only the three protagonists
    "Harry Potter (character)",
    "Hermione Granger",
    "Ron Weasley",

    # Comprehensive lore pages — broad coverage instead of per-character
    "List of Harry Potter characters",
    "Magical objects in Harry Potter",
    "Music of the Harry Potter films",
]

docs = []

for title in titles:
    loader = WikipediaLoader(query=title, load_max_docs=1)
    docs.extend(loader.load())

#print(f"Loaded {len(docs)} documents from Wikipedia.")
#print(f"First document content:\n{docs[0].page_content[:500]}...")
#print(f"First document metadata:\n{docs[0].metadata}")

chunks = splitter.split_documents(docs)

print(len(embeddings.embed_query("hello world")))

print(f"\nLoaded {len(docs)} documents:")
for d in docs:
    print(f"  - {d.metadata.get('title')}")

vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="hp_films")
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

class Phase2(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    relevance: str
    count: int

def retrieve(state):
    result = retriever.invoke(state["question"])
    return {
        "documents": result
    }

def generate(state):
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    sources = "\n".join(set(doc.metadata["source"] for doc in state["documents"]))
    prompt = template.format(context=context, question=state["question"])
    answer = model.invoke(prompt)
    return {
        "answer": answer.content
    }

def grade(state):
    context = "\n\n".join([doc.page_content for doc in state["documents"]])
    prompt = grader.format(context=context, question=state["question"])
    relevance = model.invoke(prompt)
    return {
        "relevance": relevance.content
    }

def rewrite(state):
    prompt = rewrite_template.format(question=state["question"])
    new_query = model.invoke(prompt)
    return {
        "question": new_query.content,
        "count": state["count"]+1
    }

def router(state):
    if state["count"] >= 2: return "finish"
    
    if state["relevance"] == "not_relevant": return "retry"
    else: return "finish"

build = StateGraph(Phase2)
build.add_node("retrieve", retrieve)
build.add_node("generate", generate)
build.add_node("grader", grade)
build.add_node("rewrite", rewrite)

build.add_edge(START, "retrieve")
build.add_edge("retrieve", "grader")

build.add_conditional_edges("grader", router, {"retry" : "rewrite", "finish": "generate"})
build.add_edge("rewrite", "retrieve")
build.add_edge("generate", END)

agent = build.compile()

for q in [
    "who composed harry potter music",
    "the one who dies",
    "the snake guy?",
    "the guy with silver hand",
    "when the troll gets in bathroom"
]:
    result = agent.invoke({"question": q, "count": 0})
    print(f"\nORIGINAL:   {q}")
    print(f"Final Q:    {result['question']}")
    print(f"Verdict:    {result['relevance']}")
    print(f"Attempts:   {result['count']}")
    print(f"Answer:\n{result['answer']}")
    print("-" * 60)
"""Build the Chroma vector index locally and persist it into ./chroma_db.

Run this from a machine whose IP isn't blocked by the Wikipedia API
(i.e. your laptop, not the HF Space). The resulting chroma_db/ folder is
shipped with the Space so the backend never has to call Wikipedia at runtime.
"""
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import datetime
import wikipedia
from wikipedia import wikipedia as _w

# The old `wikipedia` lib defaults to an http endpoint + a shared User-Agent
# that Wikimedia now rate-limits (HTTP 429). Use https + a descriptive UA with
# contact info, and enable polite rate limiting.
_w.API_URL = "https://en.wikipedia.org/w/api.php"
_w.USER_AGENT = (
    "hp-rag-hackathon/1.0 "
    "(https://huggingface.co/spaces/dwin2020/hp-rag-backend; darwinli@college.harvard.edu)"
)
wikipedia.set_rate_limiting(True, min_wait=datetime.timedelta(milliseconds=200))

TITLES = [
    "Harry Potter and the Philosopher's Stone (film)",
    "Harry Potter and the Chamber of Secrets (film)",
    "Harry Potter and the Prisoner of Azkaban (film)",
    "Harry Potter and the Goblet of Fire (film)",
    "Harry Potter and the Order of the Phoenix (film)",
    "Harry Potter and the Half-Blood Prince (film)",
    "Harry Potter and the Deathly Hallows – Part 1",
    "Harry Potter and the Deathly Hallows – Part 2",
    "Harry Potter (film series)",
    "Harry Potter (character)",
    "Hermione Granger",
    "Ron Weasley",
    "List of Harry Potter characters",
    "Magical objects in Harry Potter",
    "Music of the Harry Potter films",
]

PERSIST_DIR = "chroma_db"
COLLECTION = "hp_films"

if __name__ == "__main__":
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    docs = []
    for title in TITLES:
        loaded = WikipediaLoader(query=title, load_max_docs=1).load()
        docs.extend(loaded)
        print(f"  loaded {len(loaded):>2} doc(s) for: {title}")

    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
    )
    print(f"Persisted index to ./{PERSIST_DIR}")

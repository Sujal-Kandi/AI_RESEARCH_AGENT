import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import FakeEmbeddings
from dotenv import load_dotenv



load_dotenv(dotenv_path=".env") or load_dotenv(dotenv_path="..env")

CHROMA_PATH = "./vector_store"
COLLECTION_NAME = "research_memory"

# Using FakeEmbeddings - no torch/transformers needed.
# Good enough for keyword-based memory retrieval on research topics.
embeddings = FakeEmbeddings(size=384)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

def query_memory(query: str, k: int = 4) -> str:
    """Search past research for relevant context."""
    try:
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            return ""
        results = []
        for doc in docs:
            topic = doc.metadata.get("topic", "unknown")
            results.append(f"[PAST RESEARCH on '{topic}']\n{doc.page_content}")
        return "\n\n---\n\n".join(results)
    except Exception as e:
        print(f"  RAG query failed: {e}")
        return ""

def save_to_memory(topic: str, dossier_text: str, sources: list):
    """Persist completed research into vector store for future reuse."""
    try:
        # Split into chunks so retrieval is more precise
        chunks = [dossier_text[i:i+1000] for i in range(0, len(dossier_text), 1000)]
        metadatas = [{"topic": topic, "sources": str(sources[:5])} for _ in chunks]
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        print(f"  Saved {len(chunks)} chunks to memory for: '{topic}'")
    except Exception as e:
        print(f"  RAG save failed: {e}")

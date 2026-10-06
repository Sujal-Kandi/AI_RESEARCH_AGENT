import os
import json
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env") or load_dotenv(dotenv_path="..env")

CHROMA_PATH = "./vector_store"
COLLECTION_NAME = "research_memory"

# Lazy-loaded — models are only initialized when RAG is actually called.
# This saves ~90MB of RAM at startup for every request that doesn't use memory.
_embeddings = None
_vector_store = None
_llm = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings


def _get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_PATH,
        )
    return _vector_store


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    return _llm


def retrieve_past_research(query: str, k: int = 5, similarity_threshold: float = 0.6) -> dict:
    """
    Retrieve past research with similarity scores.
    Only returns results above the threshold to avoid irrelevant matches.

    Returns:
    {
        "relevant_research": [{"topic": str, "content": str, "score": float}],
        "has_relevant_research": bool
    }
    """
    try:
        results = _get_vector_store().similarity_search_with_scores(query, k=k)

        relevant = []
        for doc, score in results:
            if score >= similarity_threshold:
                relevant.append({
                    "topic": doc.metadata.get("topic", "unknown"),
                    "content": doc.page_content,
                    "score": float(score)
                })

        return {
            "relevant_research": relevant,
            "has_relevant_research": len(relevant) > 0
        }
    except Exception as e:
        print(f"  RAG retrieval failed: {e}")
        return {
            "relevant_research": [],
            "has_relevant_research": False
        }


def analyze_knowledge_gaps(user_query: str, past_research: list) -> dict:
    """
    Use LLM to analyze what we know and what gaps exist.

    Returns:
    {
        "known_areas": [str],
        "knowledge_gaps": [str],
        "research_summary": str
    }
    """
    try:
        if not past_research:
            gap_analysis_prompt = f"""
Query: {user_query}

No previous research found on this topic.

You must start completely fresh and comprehensive research.

Generate a brief analysis of:
1. What should be researched for this query (key areas to cover)
2. Priority of information gathering

Format as JSON:
{{
    "known_areas": [],
    "knowledge_gaps": ["area1", "area2", ...],
    "research_summary": "We need comprehensive research from scratch..."
}}
"""
        else:
            past_research_text = "\n".join([
                f"- Topic: {r['topic']}\n  Content: {r['content'][:300]}...\n  Relevance Score: {r['score']:.2f}"
                for r in past_research
            ])

            gap_analysis_prompt = f"""
Query: {user_query}

Previous Research Found:
{past_research_text}

Analyze this query against previous research:
1. What areas have we ALREADY researched? (known_areas)
2. What information is STILL MISSING? (knowledge_gaps)
3. Provide a summary of what we know vs what we need

Format as JSON:
{{
    "known_areas": ["area1: specific details", "area2: specific details", ...],
    "knowledge_gaps": ["missing area 1", "missing area 2", ...],
    "research_summary": "Summary of what we have and what we need..."
}}
"""

        response = _get_llm().invoke(gap_analysis_prompt)

        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            analysis = {
                "known_areas": ["Could not parse previous research"],
                "knowledge_gaps": ["Comprehensive research needed"],
                "research_summary": response.content
            }

        return analysis

    except Exception as e:
        print(f"  Gap analysis failed: {e}")
        return {
            "known_areas": [],
            "knowledge_gaps": ["Unable to analyze gaps"],
            "research_summary": f"Error during analysis: {str(e)}"
        }


def generate_targeted_queries(user_query: str, knowledge_gaps: list, num_queries: int = 5) -> list:
    """
    Generate specific research queries to fill identified knowledge gaps.

    Returns:
    ["query1", "query2", ...]
    """
    try:
        gaps_text = "\n".join([f"- {gap}" for gap in knowledge_gaps])

        query_generation_prompt = f"""
Original Query: {user_query}

Knowledge Gaps to Fill:
{gaps_text}

Generate {num_queries} specific, targeted research queries that:
1. Address the identified knowledge gaps
2. Are specific enough to get relevant information
3. Are complementary (not repetitive)
4. Follow naturally from the original query

Return ONLY a JSON array of strings (no markdown, no extra text):
["query1", "query2", "query3", ...]
"""

        response = _get_llm().invoke(query_generation_prompt)

        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                return queries[:num_queries]
        except json.JSONDecodeError:
            pass

        return [f"{user_query} - {gap}" for gap in knowledge_gaps[:num_queries]]

    except Exception as e:
        print(f"  Query generation failed: {e}")
        return [user_query]


def strategist_workflow(user_query: str) -> dict:
    """
    Complete Strategist workflow:
    1. Retrieve past research
    2. Analyze knowledge gaps
    3. Generate targeted queries
    """
    print(f"\n{'='*60}")
    print(f"STRATEGIST WORKFLOW: {user_query}")
    print(f"{'='*60}\n")

    print("[STEP 1] Retrieving past research...")
    past_research_result = retrieve_past_research(user_query)
    past_research = past_research_result["relevant_research"]

    if past_research_result["has_relevant_research"]:
        print(f"✓ Found {len(past_research)} relevant past research items")
        for item in past_research:
            print(f"  - {item['topic']} (relevance: {item['score']:.2f})")
    else:
        print("✗ No relevant past research found - starting fresh")

    print("\n[STEP 2] Analyzing knowledge gaps...")
    gap_analysis = analyze_knowledge_gaps(user_query, past_research)

    print(f"\n KNOWN AREAS:")
    for area in gap_analysis.get("known_areas", []):
        print(f"  ✓ {area}")

    print(f"\n KNOWLEDGE GAPS:")
    for gap in gap_analysis.get("knowledge_gaps", []):
        print(f"  • {gap}")

    print(f"\nSUMMARY: {gap_analysis.get('research_summary', 'N/A')}")

    print("\n[STEP 3] Generating targeted research queries...")
    targeted_queries = generate_targeted_queries(
        user_query,
        gap_analysis.get("knowledge_gaps", [])
    )

    print(f"\n TARGETED QUERIES FOR CRAWLER:")
    for i, query in enumerate(targeted_queries, 1):
        print(f"  {i}. {query}")

    return {
        "user_query": user_query,
        "past_research": past_research,
        "known_areas": gap_analysis.get("known_areas", []),
        "knowledge_gaps": gap_analysis.get("knowledge_gaps", []),
        "research_summary": gap_analysis.get("research_summary", ""),
        "targeted_queries": targeted_queries
    }


def save_to_memory(topic: str, dossier_text: str, sources: list, person: str = "default"):
    """Persist completed research into vector store for future reuse."""
    try:
        chunk_size = 1500
        overlap = 200
        chunks = []

        for i in range(0, len(dossier_text), chunk_size - overlap):
            chunk = dossier_text[i:i + chunk_size]
            if len(chunk.strip()) > 100:
                chunks.append(chunk)

        metadatas = [
            {
                "topic": topic,
                "person": person,
                "sources": str(sources[:5]),
                "chunk_index": i
            }
            for i, _ in enumerate(chunks)
        ]

        _get_vector_store().add_texts(texts=chunks, metadatas=metadatas)

        print(f"\n✓ Saved {len(chunks)} chunks to memory for: '{topic}' (person: {person})")
        print(f"  Sources: {', '.join(sources[:3])}")

    except Exception as e:
        print(f"  RAG save failed: {e}")


def clear_memory():
    """Clear all stored research (use with caution)."""
    global _vector_store
    try:
        _get_vector_store().delete_collection()
        _vector_store = None
        print("✓ Memory cleared successfully")
    except Exception as e:
        print(f"  Failed to clear memory: {e}")


def query_memory(topic: str) -> str:
    """
    Used by strategist_node to check if this topic was researched before.
    Returns a short text summary or empty string if nothing found.
    """
    result = retrieve_past_research(topic)
    if not result["has_relevant_research"]:
        return ""
    lines = [f"- {r['topic']}: {r['content'][:200]}" for r in result["relevant_research"][:3]]
    return "\n".join(lines)

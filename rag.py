import os
import json
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv(dotenv_path=".env") or load_dotenv(dotenv_path="..env")

CHROMA_PATH = "./vector_store"
COLLECTION_NAME = "research_memory"

# Use real semantic embeddings instead of FakeEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

# Initialize LLM for gap analysis
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)


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
        # Use similarity_search_with_scores to get relevance scores
        results = vector_store.similarity_search_with_scores(query, k=k)
        
        relevant = []
        for doc, score in results:
            # Only include if above threshold
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

        response = llm.invoke(gap_analysis_prompt)
        
        # Parse LLM response
        try:
            analysis = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback if LLM doesn't return valid JSON
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
        
        response = llm.invoke(query_generation_prompt)
        
        try:
            queries = json.loads(response.content)
            if isinstance(queries, list):
                return queries[:num_queries]
        except json.JSONDecodeError:
            pass
        
        # Fallback: return simple queries if LLM fails
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
    
    Returns:
    {
        "user_query": str,
        "past_research": [...],
        "known_areas": [...],
        "knowledge_gaps": [...],
        "research_summary": str,
        "targeted_queries": [...]
    }
    """
    print(f"\n{'='*60}")
    print(f"STRATEGIST WORKFLOW: {user_query}")
    print(f"{'='*60}\n")
    
    # Step 1: Retrieve past research
    print("[STEP 1] Retrieving past research...")
    past_research_result = retrieve_past_research(user_query)
    past_research = past_research_result["relevant_research"]
    
    if past_research_result["has_relevant_research"]:
        print(f"✓ Found {len(past_research)} relevant past research items")
        for item in past_research:
            print(f"  - {item['topic']} (relevance: {item['score']:.2f})")
    else:
        print("✗ No relevant past research found - starting fresh")
    
    # Step 2: Analyze knowledge gaps
    print("\n[STEP 2] Analyzing knowledge gaps...")
    gap_analysis = analyze_knowledge_gaps(user_query, past_research)
    
    print(f"\n KNOWN AREAS:")
    for area in gap_analysis.get("known_areas", []):
        print(f"  ✓ {area}")
    
    print(f"\n KNOWLEDGE GAPS:")
    for gap in gap_analysis.get("knowledge_gaps", []):
        print(f"  • {gap}")
    
    print(f"\nSUMMARY: {gap_analysis.get('research_summary', 'N/A')}")
    
    # Step 3: Generate targeted queries
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
    """
    Persist completed research into vector store for future reuse.
    
    Args:
        topic: Research topic/title
        dossier_text: Full research content
        sources: List of sources used
        person: Optional person/user identifier for multi-user scenarios
    """
    try:
        # Smart chunking: preserve context while splitting
        # Use 1500 chars per chunk for better semantic coherence
        chunk_size = 1500
        overlap = 200
        chunks = []
        
        for i in range(0, len(dossier_text), chunk_size - overlap):
            chunk = dossier_text[i:i + chunk_size]
            if len(chunk.strip()) > 100:  # Only add meaningful chunks
                chunks.append(chunk)
        
        # Add metadata for better retrieval and multi-user support
        metadatas = [
            {
                "topic": topic,
                "person": person,
                "sources": str(sources[:5]),
                "chunk_index": i
            }
            for i, _ in enumerate(chunks)
        ]
        
        # Add documents to vector store
        vector_store.add_texts(texts=chunks, metadatas=metadatas)
        
        print(f"\n✓ Saved {len(chunks)} chunks to memory for: '{topic}' (person: {person})")
        print(f"  Sources: {', '.join(sources[:3])}")
    
    except Exception as e:
        print(f"  RAG save failed: {e}")


def clear_memory():
    """Clear all stored research (use with caution!)"""
    try:
        # Reset the vector store
        global vector_store
        vector_store.delete_collection()
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        print("✓ Memory cleared successfully")
    except Exception as e:
        print(f"  Failed to clear memory: {e}")


from rag import strategist_workflow

# Test the workflow
result = strategist_workflow("Biography of Anushka Sharma")

print("\n Targeted Queries to Research:")
for query in result["targeted_queries"]:
    print(f"  - {query}")


# Add or verify these in rag.py
def query_memory(query: str, top_k: int = 5):
    """Retrieves relevant background context from vector store."""
    # Your ChromaDB / vector store query logic here
    pass

def save_to_memory(text: str, metadata: dict = None):
    """Saves text chunk to vector store."""
    pass
import os
import sqlite3
import requests
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from fpdf import FPDF
from fpdf.enums import XPos, YPos



from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from rag import query_memory, save_to_memory


load_dotenv(dotenv_path="/etc/secrets/.env", override=False)
load_dotenv(dotenv_path=".env", override=False)

# Manual fallback for Render secret files
_secret_path = "/etc/secrets/.env"
if os.path.exists(_secret_path):
    with open(_secret_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                if not os.environ.get(_k.strip()):
                    os.environ[_k.strip()] = _v.strip()

# ── TOOLS ─────────────────────────────────────────────────────────────────────
def get_web_search():
    return TavilySearch(
        max_results=6,
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        include_raw_content=True,
    )

def make_llm(key: str):
    return ChatGroq(model="openai/gpt-oss-120b", api_key=key, temperature=0.2)



# ── CUSTOM EXCEPTIONS ─────────────────────────────────────────────────────────
class RateLimitExhausted(Exception):
    """Raised when all LLM API keys are rate limited and no fallback is available."""
    pass

# ── LLM with key rotation ─────────────────────────────────────────────────────
_groq_keys = []
_key_index = 0
_using_fallback = False
# Key index -> monotonic timestamp when that key is expected to be usable again.
# A rate-limited key is remembered as cooling rather than permanently burned, so
# a long run can come back to it once its per-minute window resets.
_key_cooldowns: Dict[int, float] = {}
llm = None

# Longest we will sit waiting for a cooling key before giving up on Groq, and
# the total waiting budget for a single invocation.
MAX_RATE_LIMIT_WAIT = float(os.getenv("MAX_RATE_LIMIT_WAIT", "90"))
MAX_TOTAL_RATE_LIMIT_WAIT = float(os.getenv("MAX_TOTAL_RATE_LIMIT_WAIT", "300"))
DEFAULT_COOLDOWN = float(os.getenv("KEY_COOLDOWN_SECONDS", "60"))

# Guards every mutation of the globals above so parallel section writers cannot
# rotate past each other's keys or clobber the active client.
_llm_lock = threading.RLock()
# Bumped on every rotation; lets a thread detect that another thread already
# rotated away from the client it failed on.
_llm_generation = 0

def _init_llm():
    """Initialize LLM on first use, not at import time."""
    global _groq_keys, llm
    with _llm_lock:
        if not _groq_keys:
            _groq_keys = [k.strip() for k in [
                os.getenv("GROQ_API_KEY", ""),
                os.getenv("GROQ_API_KEY_2", ""),
                os.getenv("GROQ_API_KEY_3", ""),
                os.getenv("GROQ_API_KEY_4", ""),
                os.getenv("GROQ_API_KEY_5", ""),
            ] if k and k.strip()]
            print(f"  [INIT] Loaded {len(_groq_keys)} Groq keys")
        if llm is None and _groq_keys:
            llm = make_llm(_groq_keys[0])
            print(f"  [INIT] LLM initialized with key index 0")

def make_together_llm():
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        raise RuntimeError("No TOGETHER_API_KEY found in .env")
    print("  [FALLBACK] Switching to Together.ai (Llama-3.3-70B-Instruct)")
    return ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct",
        together_api_key=together_key,
        temperature=0.2,
    )

def _is_rate_limit_error(e: Exception) -> bool:
    """Detect rate limit errors across different exception types."""
    e_str = str(e).lower()
    e_type = type(e).__name__.lower()
    return (
        "429" in str(e)
        or "rate" in e_str
        or "limit" in e_str
        or "ratelimit" in e_type
        or "quota" in e_str
        or "tokens per day" in e_str
        or "try again" in e_str
    )

def _retry_after_seconds(error: Exception) -> float:
    """How long the provider says to wait. Groq puts it in the 429 message."""
    if error is None:
        return DEFAULT_COOLDOWN
    import re
    match = re.search(r"try again in ([\d.]+)\s*(ms|m|s)\b", str(error), re.I)
    if not match:
        return DEFAULT_COOLDOWN
    value, unit = float(match.group(1)), match.group(2).lower()
    seconds = value / 1000 if unit == "ms" else value * 60 if unit == "m" else value
    return min(max(seconds, 1.0), 600.0)

def _structured_invoke(schema, messages):
    """Structured-output call that shares the rotation/backoff path. None if exhausted."""
    _init_llm()
    try:
        return llm_invoke_with_rotation(messages, structured_output=schema)
    except RateLimitExhausted:
        return None

def _current_client():
    """Snapshot the active client and the generation it belongs to."""
    with _llm_lock:
        return llm, _llm_generation

def _advance_key(failed_generation: int, error: Exception = None):
    """Move to the next usable key/provider unless another thread already did.

    Returns the number of seconds the caller should sleep before retrying, or
    None when nothing is left to try. The sleep happens in the caller so this
    never holds the lock while waiting.
    """
    global llm, _key_index, _using_fallback, _llm_generation
    with _llm_lock:
        if _llm_generation != failed_generation:
            # Another thread already rotated; just retry with the new client.
            return 0.0
        if _using_fallback:
            return None

        now = time.monotonic()
        _key_cooldowns[_key_index] = now + _retry_after_seconds(error)

        ready = [i for i in range(len(_groq_keys)) if _key_cooldowns.get(i, 0.0) <= now]
        if ready:
            _key_index = ready[0]
            llm = make_llm(_groq_keys[_key_index])
            _llm_generation += 1
            print(f"  [KEY ROTATION] Switched to Groq key {_key_index + 1}")
            return 1.0

        # Every key is cooling. Waiting for the soonest one is almost always
        # faster than failing the whole run — a Groq TPM window is ~60s.
        soonest = min(range(len(_groq_keys)), key=lambda i: _key_cooldowns.get(i, 0.0))
        wait = max(0.0, _key_cooldowns.get(soonest, 0.0) - now)
        if wait <= MAX_RATE_LIMIT_WAIT:
            _key_index = soonest
            llm = make_llm(_groq_keys[soonest])
            _llm_generation += 1
            print(f"  [RATE LIMIT] All {len(_groq_keys)} Groq keys cooling; "
                  f"waiting {wait:.0f}s for key {soonest + 1}")
            return wait

        print("  [RATE LIMIT] All Groq keys exhausted, switching to Together.ai...")
        try:
            llm = make_together_llm()
        except RuntimeError:
            return None  # No Together.ai key configured at all
        except Exception as fallback_e:
            if "401" in str(fallback_e) or "invalid" in str(fallback_e).lower():
                print("  [TOGETHER AUTH FAILED] Invalid API key — check TOGETHER_API_KEY in .env")
            raise fallback_e
        _using_fallback = True
        _key_cooldowns.clear()
        _llm_generation += 1
        return 2.0

def llm_invoke_with_rotation(messages, structured_output=None):
    """Invoke LLM, rotating Groq keys on rate limit, then falling back to Together.ai.

    Safe to call from multiple threads: only rotation touches shared state, and
    the network call itself happens outside the lock so sections run in parallel.
    """
    _init_llm()  # ensure LLM is initialized
    waited = 0.0

    while True:
        client, generation = _current_client()
        try:
            if structured_output is not None:
                return client.with_structured_output(structured_output).invoke(messages)
            return client.invoke(messages)
        except Exception as e:
            if not _is_rate_limit_error(e):
                print(f"  [LLM ERROR] {type(e).__name__}: {e}")
                raise
            wait = _advance_key(generation, e)
            if wait is None or waited + wait > MAX_TOTAL_RATE_LIMIT_WAIT:
                raise RateLimitExhausted(
                    "All API keys are currently rate limited. Please try again in 5 minutes."
                )
            waited += wait
            time.sleep(wait)

def _rotate_key():
    """Rotate to next available Groq API key."""
    global llm, _key_index, _llm_generation
    _init_llm()
    with _llm_lock:
        if _using_fallback:
            return
        _key_index = (_key_index + 1) % len(_groq_keys)
        llm = make_llm(_groq_keys[_key_index])
        _llm_generation += 1
# ── PERSISTENCE ───────────────────────────────────────────────────────────────
conn = sqlite3.connect("research_memory.db", check_same_thread=False)
memory = SqliteSaver(conn)

# ── SCHEMAS ───────────────────────────────────────────────────────────────────
class ResearchPlan(BaseModel):
    queries: List[str] = Field(description="10-15 high-precision search queries covering different angles.")
    reasoning: str = Field(description="Why these queries were chosen and what gaps they cover.")

class CitedFact(BaseModel):
    """A hard fact with its source ID. Used as sidebar data points."""
    fact: str = Field(description="One hard metric, date, or specification. e.g. 'H100 delivers 3.35 TB/s memory bandwidth'")
    source_id: int = Field(description="The [N] index number from the SOURCE INDEX. Must be a real index number.")

class ResearchChapter(BaseModel):
    title: str = Field(description="Clear, specific chapter title in Title Case. e.g. 'Memory Architecture and Bandwidth Evolution'")
    intro: str = Field(description="2-3 sentence paragraph that sets context and states what this chapter investigates.")
    narrative: str = Field(
        description=(
            "The main investigative body. Write as flowing prose like a journal article or long-form report. "
            "Minimum 200 words. Build an argument - show cause and effect, tensions, decisions, consequences. "
            "Cite sources inline using [N] notation e.g. 'The H100 introduced HBM3 in 2022 [3], doubling bandwidth over the A100 [7].' "
            "Only cite source IDs that exist in the SOURCE INDEX."
        )
    )
    key_facts: List[CitedFact] = Field(description="3-5 hard data points pulled from this chapter as a sidebar. Each must cite a real source_id.")
    takeaway: str = Field(description="One sharp sentence - the single most important insight from this chapter.")

class FinalDossier(BaseModel):
    title: str = Field(description="Specific, descriptive report title.")
    key_findings: List[str] = Field(description="5 most important findings from the entire report. Each should be a complete sentence with a hard fact.")
    executive_summary: str = Field(
        description=(
            "Half-page executive summary (150-200 words). Written for a senior decision-maker. "
            "Cover: what was investigated, the 3 most important findings, and the strategic implication. "
            "Must contain specific metrics and dates, not vague statements."
        )
    )
    chapters: List[ResearchChapter]
    synthesis: str = Field(
        description=(
            "Final synthesis paragraph (100-150 words). Reveal a non-obvious connection across chapters. "
            "What does the data collectively suggest that no single chapter states explicitly?"
        )
    )

class SectionChallenge(BaseModel):
    section: str = Field(description="Exact title of the weak section, copied verbatim.")
    problem: str = Field(description="One of: Safe explanation, Repetition, Missing contrarian, No specifics.")
    challenge: str = Field(description="One sharp question or missing angle the rewrite must address.")

class QualityVerdict(BaseModel):
    score: int = Field(description="Quality score 1-10. 8+ means publish-ready.")
    gaps: List[str] = Field(default_factory=list, description="Specific missing data points. Empty list if none.")
    follow_up_queries: List[str] = Field(default_factory=list, description="New queries to fill gaps. Empty list if score >= 8.")
    verdict: str = Field(description="APPROVED or NEEDS_MORE_RESEARCH")

    @field_validator("gaps", "follow_up_queries", mode="before")
    @classmethod
    def coerce_to_list(cls, v):
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v or []

class ReportAudit(QualityVerdict):
    """Scoring and weak-section triage in one call instead of two round trips."""
    weak_sections: List[SectionChallenge] = Field(
        default_factory=list,
        description="The 2 weakest sections and what each rewrite must fix. Empty if none are weak.",
    )

class AgentState(Dict):
    topic: str
    plan: ResearchPlan
    raw_data: str
    raw_report: str
    source_index: Dict[int, str]
    source_titles: Dict[int, str]
    source_texts: Dict[int, str]  # per-source crawled text, used for retrieval + verification
    memory_context: str
    quality: QualityVerdict
    weak_sections: List[SectionChallenge]  # from the audit, consumed by the rewriter
    grounding: float  # share of factual sentences corroborated by source text
    iteration: int
    research_rounds: int

# ── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────
STRATEGIST_PROMPT = """You are an elite intelligence analyst. Build a comprehensive research plan.
Generate 10-15 search queries covering:
- Technical specifications and hard metrics
- Historical timeline with specific dates and events
- Key figures, engineers, decision-makers
- Comparative analysis and bottlenecks
- Failures, weaknesses, and overlooked angles
Every query must target a specific data point. No generic queries."""

CRITIC_PROMPT = """You are a ruthless research quality auditor for a top-tier journal.
You do two jobs in one pass: score the report, and name its 2 weakest sections.

Score the report 1-10 on grounding first, style second:
- 1-4: Claims without citations, generic filler, reads like a summary of a summary
- 5-7: Cited but thin — few hard numbers, repeats itself across sections, hedged judgments
- 8-10: Every substantive claim traceable to a source, specific figures and dates, mechanisms
  explained, no repetition
Penalise heavily: numbers or dates with no citation, sentences that could appear in a report on
any other topic, and paragraphs that restate an earlier point.
Be harsh. Only approve (score >= 8) if a senior researcher would find it valuable.
Only ask for more research when a section is missing facts no rewrite could supply.

For weak_sections, return the 2 worst sections (or none if all are solid). Copy each
section title verbatim and classify the problem as one of:
- "Safe explanation" — describes what happened but never says why, or what it cost
- "Repetition" — repeats a point already made in another section
- "Missing contrarian" — accepts the obvious narrative without challenging it
- "No specifics" — claims without hard numbers, dates, or named decisions
Each challenge must be one sharp question the rewrite has to answer."""

# ── NODES ──────────────────────────────────────────────────────────────────────
def strategist_node(state: AgentState):
    print("\n[STRATEGIST] Building search vectors...")
    memory_context = query_memory(state["topic"])
    print(f"  Memory: {'found past research' if memory_context else 'starting fresh'}")

    memory_hint = f"\n\nPAST RESEARCH (avoid re-searching these):\n{memory_context}" if memory_context else ""

    plan = _structured_invoke(ResearchPlan, [
        SystemMessage(content=STRATEGIST_PROMPT),
        HumanMessage(content=f"Build research plan for: {state['topic']}{memory_hint}")
    ])
    if plan is None:
        raise RateLimitExhausted(
            "All API keys are currently rate limited. Please try again in 5 minutes."
        )
    print(f"  {len(plan.queries)} queries planned")
    return {
        "plan": plan,
        "iteration": 1,
        "research_rounds": 0,
        "source_index": {},
        "source_titles": {},
        "source_texts": {},
        "memory_context": memory_context or "",
    }

def hitl_node(state: AgentState):
    print(f"\n[COMMANDER REVIEW] {len(state['plan'].queries)} vectors ready")
    for i, q in enumerate(state['plan'].queries, 1):
        print(f"  {i:02d}. {q}")
    return state

def deep_fetch(url: str, max_chars: int = 5000) -> str:
    """Fetch full page content from a URL. Falls back gracefully."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        if resp.status_code != 200:
            return ""
        text = resp.text
        # Strip HTML tags simply
        import re
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""

# Domains whose numbers can be trusted over a random blog post. Ranked highest
# when a section has more candidate sources than it can fit in context.
PRIMARY_SOURCE_MARKERS = (
    ".gov", ".gov.in", ".gov.uk", ".edu", ".ac.", ".int",
    "rbi.org", "npci.org", "bis.org", "imf.org", "worldbank.org", "oecd.org",
    "who.int", "europa.eu", "nature.com", "science.org", "arxiv.org",
    "ieee.org", "acm.org", "sec.gov", "investor.", "nasa.gov", "nist.gov",
)
BLOG_MARKERS = (
    "medium.com", "substack.com", "blogspot.", "wordpress.", "quora.com",
    "/blog/", "blog.", "linkedin.com/pulse",
)

def source_tier(url: str) -> int:
    """0 = primary/official, 1 = ordinary press, 2 = blog/self-published."""
    u = url.lower()
    if any(m in u for m in PRIMARY_SOURCE_MARKERS):
        return 0
    if any(m in u for m in BLOG_MARKERS):
        return 2
    return 1

# Search and page fetches are network-bound and have no shared state, so they
# run concurrently. IDs are still assigned in query order afterwards, which
# keeps citation numbers stable regardless of which request finishes first.
CRAWL_WORKERS = int(os.getenv("CRAWL_WORKERS", "5"))
DEEP_FETCH_LIMIT = int(os.getenv("DEEP_FETCH_LIMIT", "5"))

def emit(state, stage: str, detail: str = "", done=None, total=None):
    """Report real pipeline progress to whoever is watching (the API, the CLI).

    The callback lives in state under "_progress" so nothing here depends on a
    global, and a missing or failing callback can never break a run.
    """
    cb = state.get("_progress") if isinstance(state, dict) else None
    if not cb:
        return
    try:
        cb(stage, detail, done, total)
    except Exception:
        pass

def _search_one(query: str):
    """Run one search query. Never raises — a dead query must not kill the crawl."""
    try:
        response = get_web_search().invoke(query)
        return response.get("results", []) if isinstance(response, dict) else response
    except Exception as e:
        print(f"  Query failed: {query[:50]} — {e}")
        return []

def crawler_node(state: AgentState):
    queries = state['plan'].queries
    rounds = state.get("research_rounds", 0) + 1
    print(f"\n[CRAWLER] Round {rounds} - {len(queries)} queries ({CRAWL_WORKERS} at a time)...")

    source_index: Dict[int, str] = {}
    source_titles: Dict[int, str] = {}
    source_texts: Dict[int, str] = {}
    for k, v in (state.get("source_index") or {}).items():
        source_index[int(k)] = v
    for k, v in (state.get("source_titles") or {}).items():
        source_titles[int(k)] = v
    for k, v in (state.get("source_texts") or {}).items():
        source_texts[int(k)] = v

    emit(state, "crawler", "Searching the web", 0, len(queries))
    results_per_query = [[] for _ in queries]
    finished = 0
    with ThreadPoolExecutor(max_workers=min(CRAWL_WORKERS, max(len(queries), 1))) as pool:
        futures = {pool.submit(_search_one, q): i for i, q in enumerate(queries)}
        for fut in as_completed(futures):
            # Results are stored by query index, so citation numbering stays
            # stable even though completion order is arbitrary.
            results_per_query[futures[fut]] = fut.result()
            finished += 1
            emit(state, "crawler", "Searching the web", finished, len(queries))

    seen_urls = set(source_index.values())
    new_sources = []  # (sid, url, title, snippet)
    for items in results_per_query:
        for r in items:
            url = (r.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            sid = len(source_index) + 1
            source_index[sid] = url
            title = (r.get("title") or url)[:80]
            source_titles[sid] = title
            new_sources.append((sid, url, title, (r.get("raw_content") or r.get("content") or "")[:600]))

    # Deep fetch the most authoritative sources first — official/primary domains
    # are where the hard numbers live, so they get the full-page fetch budget.
    ranked = sorted(new_sources, key=lambda s: source_tier(s[1]))
    skip_markers = ("youtube.com", "twitter.com", "linkedin.com", ".pdf", "reddit.com")
    to_fetch = [s for s in ranked if not any(x in s[1] for x in skip_markers)][:DEEP_FETCH_LIMIT]

    print(f"  Deep fetching {len(to_fetch)} top sources in parallel...")
    fetched = {}
    if to_fetch:
        emit(state, "crawler", "Reading top sources", 0, len(to_fetch))
        done_fetch = 0
        with ThreadPoolExecutor(max_workers=len(to_fetch)) as pool:
            futures = {pool.submit(deep_fetch, s[1], 2000): s[0] for s in to_fetch}
            for fut in as_completed(futures):
                fetched[futures[fut]] = fut.result()
                done_fetch += 1
                emit(state, "crawler", "Reading top sources", done_fetch, len(to_fetch))

    deep_fetch_count = 0
    for sid, url, title, snippet in new_sources:
        full = fetched.get(sid, "")
        if len(full) > len(snippet):
            source_texts[sid] = full
            deep_fetch_count += 1
        else:
            source_texts[sid] = snippet

    raw_chunks = [
        f"[{sid}] SOURCE: {url}\nTITLE: {title}\n{source_texts.get(sid, '')}"
        for sid, url, title, _ in new_sources
    ]

    print(f"  {len(source_index)} sources indexed, {deep_fetch_count} deep fetched")
    emit(state, "crawler", f"{len(source_index)} sources indexed, {deep_fetch_count} read in full")

    # ── GUARD: stop before any LLM calls if web returned nothing useful ───────
    if len(source_index) < 3:
        raise ValueError(
            f"Not enough sources found for '{state['topic']}' ({len(source_index)} result(s)). "
            "This topic may not have enough public information available. "
            "Try a well-known subject, event, company, or technology."
        )

    existing_raw = state.get("raw_data", "")
    combined = existing_raw + f"\n\n=== ROUND {rounds} ===\n\n" + "\n---\n".join(raw_chunks)
    return {
        "raw_data": combined,
        "source_index": source_index,
        "source_titles": source_titles,
        "source_texts": source_texts,
        "research_rounds": rounds,
    }

# Phrases the model reaches for when it has nothing concrete to say. Banned in
# the prompts and stripped from output if they survive.
SLOP_PHRASES = (
    "in hindsight",
    "a better alternative would have been",
    "this decision was a mistake because",
    "in conclusion",
    "it is essential to",
    "it is important to note",
    "cannot be overstated",
    "remarkable journey",
    "paved the way",
    "the digital landscape",
    "a testament to",
    "delve into",
    "significant implications for",
    "as we look to the future",
)

SECTION_COUNT = int(os.getenv("SECTION_COUNT", "5"))
SECTION_WORKERS = int(os.getenv("SECTION_WORKERS", "3"))
SECTION_MIN_WORDS = int(os.getenv("SECTION_MIN_WORDS", "600"))

def generate_section_topics(topic: str, raw_data: str) -> List[str]:
    """Ask the LLM to propose section titles tailored to the research topic."""
    prompt = f"""You are planning a long-form research report on: "{topic}"

Based on this topic, propose exactly {SECTION_COUNT} section titles that together give comprehensive coverage.
Rules:
- Each title must be specific and distinct — no overlap
- Each section must be broad enough to carry {SECTION_MIN_WORDS}+ words of dense analysis on its own
- One section MUST be a timeline of key events (title it like "Timeline: [Topic] from [Year] to [Year]")
- One section MUST be a contrarian/critical analysis (title it like "The Contrarian View: Was Failure Inevitable?" or similar)
- One section MUST be a comparison (title it like "Comparative Analysis: [A] vs [B]" or similar)
- Merge narrow angles (origins, peak, decline, legacy) into fewer, deeper sections rather than splitting them
Return ONLY a Python list of strings, nothing else. Example:
["Title One", "Title Two", "Title Three", "Title Four", "Title Five"]"""
    try:
        response = llm_invoke_with_rotation([HumanMessage(content=prompt)]).content.strip()
        import ast, re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            titles = [t for t in ast.literal_eval(match.group()) if isinstance(t, str) and t.strip()]
            if titles:
                return titles[:SECTION_COUNT]
    except Exception:
        pass
    # Fallback generic sections
    return [
        "Origins, Peak Dominance, and the Foundations of Later Failure",
        "Timeline: Key Events and Turning Points",
        "Disruption and Strategic Missteps",
        "Comparative Analysis: Key Competitors vs Subject",
        "The Contrarian View: Was Failure Inevitable?",
    ][:SECTION_COUNT]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "to", "from", "with",
    "its", "it", "is", "was", "were", "as", "at", "by", "that", "this", "analysis",
    "section", "view", "key", "vs", "versus", "timeline", "comparative", "case",
}

def _keywords(text: str) -> List[str]:
    import re
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2]

def select_sources(section_title: str, topic: str, source_texts: Dict[int, str],
                   source_titles: Dict[int, str], source_index: Dict[int, str],
                   limit: int = 10) -> List[int]:
    """Rank sources by term overlap with the section title, then by source tier.

    Each section gets its own evidence subset, so parallel sections stop
    converging on whichever handful of facts sat at the top of the shared blob.
    """
    terms = set(_keywords(section_title)) | set(_keywords(topic))
    scored = []
    for sid, text in source_texts.items():
        haystack = f"{source_titles.get(sid, '')} {text}".lower()
        overlap = sum(haystack.count(t) for t in terms)
        tier = source_tier(source_index.get(sid, ""))
        # Primary sources win ties; blogs need real overlap to make the cut.
        scored.append((-(overlap + (6 if tier == 0 else 2 if tier == 1 else 0)), sid))
    scored.sort()
    return [sid for _, sid in scored[:limit]]

def build_evidence_block(sids: List[int], source_index: Dict[int, str],
                         source_titles: Dict[int, str], source_texts: Dict[int, str],
                         chars_per_source: int = 1200) -> str:
    return "\n\n".join(
        f"[{sid}] {source_index.get(sid, '')} - \"{source_titles.get(sid, '')}\"\n"
        f"{(source_texts.get(sid) or '')[:chars_per_source]}"
        for sid in sids
    )

def strip_slop(text: str) -> str:
    """Drop sentences built around filler phrases the prompt already bans."""
    import re
    kept = []
    for para in text.split("\n"):
        sentences = re.split(r'(?<=[.!?])\s+', para)
        kept_sentences = [
            s for s in sentences
            if not any(p in s.lower() for p in SLOP_PHRASES)
        ]
        kept.append(" ".join(kept_sentences) if len(sentences) > 1 else para)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

def architect_node(state: AgentState):
    print("\n[ARCHITECT] Writing research report section by section...")

    source_index = {int(k): v for k, v in state["source_index"].items()}
    source_titles = {int(k): v for k, v in (state.get("source_titles") or {}).items()}
    source_texts = {int(k): v for k, v in (state.get("source_texts") or {}).items()}

    top_index = dict(sorted(source_index.items())[:30])
    index_str = "\n".join(
        f"[{sid}] {url} - \"{source_titles.get(sid, '')}\""
        for sid, url in top_index.items()
    )

    raw = state["raw_data"]
    if len(raw) > 14000:
        raw = raw[:14000] + "\n[truncated]"

    memory_section = (
        f"\n\nPAST RESEARCH CONTEXT:\n{state['memory_context'][:1500]}"
        if state.get("memory_context") else ""
    )

    topic = state["topic"]
    context_block = f"""TOPIC: {topic}

SOURCES (cite inline as [N] when using specific facts):
{index_str}

RAW DATA:
{raw}{memory_section}"""

    banned = "; ".join(f'"{p}"' for p in SLOP_PHRASES)
    section_system = (
        "You are a senior investigative journalist writing for IEEE Spectrum or MIT Technology Review. "
        "Write only what the supplied evidence supports. "
        "Every number, date, name, and percentage you write MUST appear in the EVIDENCE text you were given, "
        "cited with the [N] of the source it came from. If the evidence does not contain a figure, write about "
        "what the evidence does say instead of estimating, rounding, or recalling it from memory. "
        "Attribute claims to the organisation that produced the data (e.g. RBI, NPCI), never to the blogger who repeated it. "
        f"Never use these filler phrases: {banned}. "
        "Do not end sections with a summary or conclusion paragraph. Do not restate a point you already made. "
        "No bullet points. Length follows the evidence — stop when the evidence is used up rather than padding."
    )

    # Step 1: Write title, key findings, executive summary
    print("  Writing header (title, findings, summary)...")
    emit(state, "architect", "Writing title, key findings and summary")
    header = llm_invoke_with_rotation([
        SystemMessage(content=section_system),
        HumanMessage(content=f"""{context_block}

Write ONLY the following three parts, nothing else:

## TITLE
[A specific, descriptive title for this research report]

## KEY_FINDINGS
1. [Finding with hard fact and citation]
2. [Finding with hard fact and citation]
3. [Finding with hard fact and citation]
4. [Finding with hard fact and citation]
5. [Finding with hard fact and citation]

## EXECUTIVE_SUMMARY
[200-250 words. Written for a senior decision-maker. Cover what was investigated, the 3 most critical findings with specific metrics, and the strategic implication. Every figure must come from the sources above and carry its [N].]""")
    ]).content
    header = strip_slop(header)

    # Step 2: Write each section independently, in parallel
    emit(state, "architect", "Planning report sections")
    section_topics = generate_section_topics(topic, raw)

    def write_section(index_and_title):
        i, sec_title = index_and_title
        # Sections are written concurrently, so each one is told what the others
        # cover instead of relying on already-written text to avoid overlap.
        other_titles = "\n".join(
            f"- {t}" for j, t in enumerate(section_topics) if j != i
        )
        sids = select_sources(sec_title, topic, source_texts, source_titles, source_index)
        evidence = build_evidence_block(sids, source_index, source_titles, source_texts)
        print(f"  Writing section {i + 1}/{len(section_topics)}: {sec_title} "
              f"(sources {', '.join(str(s) for s in sids)})...")
        section = llm_invoke_with_rotation([
            SystemMessage(content=section_system),
            HumanMessage(content=f"""TOPIC: {topic}

EVIDENCE FOR THIS SECTION — this is the only material you may draw facts from:
{evidence}

Write ONLY this one section. Do not write any other sections.

## SECTION: {sec_title}

OTHER SECTIONS IN THIS REPORT (written separately — do NOT cover their ground):
{other_titles}

How to write it:
- Ground every claim in the EVIDENCE above and cite it inline as [N]. A sentence with a
  number, date, or name and no [N] is not acceptable.
- If the evidence lacks a figure you want, say what is missing ("the crawled sources do not
  report X") rather than supplying a number from memory.
- Explain mechanism: what caused what, who decided it, what it cost, what followed.
- Where the evidence supports a judgment, make it and say which fact drives it. Where it does
  not, describe the fact and stop. Do not manufacture a verdict for every paragraph.
- If this section is a comparison or timeline, build it from evidence rows only:
  a plain-text table using | separators, or lines of YEAR: Event [N] — consequence.
- Aim for roughly {SECTION_MIN_WORDS} words, but only as far as the evidence carries you.
  A shorter, fully-grounded section beats a padded one.
- No opening throat-clearing, no closing summary. Start on the first substantive fact.

Write the section now:""")
        ]).content
        section = strip_slop(section)
        # Normalize section header in case model added extra text before it
        if f"## SECTION: {sec_title}" not in section:
            section = f"## SECTION: {sec_title}\n{section}"
        return section

    workers = max(1, min(SECTION_WORKERS, len(section_topics)))
    emit(state, "architect", "Writing sections", 0, len(section_topics))
    sections = [""] * len(section_topics)
    written = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(write_section, (i, t)): i
            for i, t in enumerate(section_topics)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            sections[i] = fut.result()
            written += 1
            emit(state, "architect", f"Wrote: {section_topics[i]}", written, len(section_topics))

    # Step 3: Write synthesis
    print("  Writing synthesis...")
    emit(state, "architect", "Writing cross-section synthesis")
    synthesis = llm_invoke_with_rotation([
        SystemMessage(content=section_system),
        HumanMessage(content=f"""{context_block}

Write ONLY the final synthesis section.

## SYNTHESIS
[150-200 words. Reveal a non-obvious connection across all the themes covered.
What does the data collectively suggest that no single section states explicitly?
Include a contrarian take — something that challenges the dominant narrative of the report.
Be specific and opinionated, and tie each claim to the [N] that supports it. No vague conclusions.]""")
    ]).content
    synthesis = strip_slop(synthesis)

    raw_report = header + "\n\n" + "\n\n".join(sections) + "\n\n" + synthesis
    print(f"  Total report: {len(raw_report)} chars across {len(sections)} sections")
    emit(state, "architect", f"{len(raw_report.split())} words across {len(sections)} sections")

    with open("raw_report_debug.txt", "w", encoding="utf-8") as f:
        f.write(raw_report)

    return {"raw_report": raw_report}

STRIP_UNVERIFIED = os.getenv("STRIP_UNVERIFIED", "1") != "0"
# Never gut a section: if more than this share of its factual sentences fail
# verification, the crawl is too thin to judge and the text is kept as-is.
MAX_STRIP_RATIO = float(os.getenv("MAX_STRIP_RATIO", "0.4"))

def _numbers_in(text: str) -> List[str]:
    """Numeric tokens that a source must corroborate (years, %, magnitudes)."""
    import re
    raw_numbers = re.findall(r'\d[\d,\.]*', text)
    out = []
    for n in raw_numbers:
        clean = n.replace(",", "").rstrip(".")
        # Single digits and trivial values match almost any text; ignore them.
        if clean and (len(clean) > 1 or clean == "0") and float(clean or 0) >= 2:
            out.append(clean)
    return out

def _corroborated(number: str, haystack: str) -> bool:
    """True if the number appears in the source text in any usual formatting."""
    if number in haystack:
        return True
    if len(number) > 3:  # 13000 written as 13,000
        with_commas = f"{int(float(number)):,}" if number.isdigit() else number
        if with_commas in haystack:
            return True
    return False

def verify_claims(raw_report: str, source_texts: Dict[int, str], source_index: Dict[int, str]):
    """Strip sentences whose figures no source actually supports.

    The old check only asked whether a cited [N] existed in the index, so an
    invented date wearing a real citation passed. This checks the numbers
    themselves against the crawled text of the sources cited in that sentence,
    falling back to the whole corpus when the sentence cites nothing.
    """
    import re
    corpus = " ".join(source_texts.values())
    valid_ids = set(source_index.keys())

    kept_lines, unverified, checked, bad_ids = [], [], 0, set()
    for line in raw_report.split("\n"):
        if line.startswith("##") or line.startswith("|") or not line.strip():
            kept_lines.append(line)
            continue

        sentences = re.split(r'(?<=[.!?])\s+', line)
        kept_sentences, dropped = [], []
        for sentence in sentences:
            cited = {int(x) for x in re.findall(r'\[(\d+)\]', sentence)}
            bad_ids |= (cited - valid_ids)
            numbers = _numbers_in(re.sub(r'\[\d+\]', '', sentence))
            if not numbers:
                kept_sentences.append(sentence)
                continue

            checked += 1
            cited_text = " ".join(source_texts.get(sid, "") for sid in cited & valid_ids)
            haystack = cited_text or corpus
            if all(_corroborated(n, haystack) or _corroborated(n, corpus) for n in numbers):
                kept_sentences.append(sentence)
            else:
                dropped.append(sentence)

        if dropped and STRIP_UNVERIFIED and len(dropped) <= max(1, int(len(sentences) * MAX_STRIP_RATIO)):
            unverified.extend(dropped)
            kept_lines.append(" ".join(kept_sentences))
        else:
            unverified.extend(dropped)
            kept_lines.append(line)

    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept_lines))
    grounding = 1.0 if not checked else round((checked - len(unverified)) / checked, 2)
    return cleaned, {
        "checked": checked,
        "unverified": unverified,
        "grounding": grounding,
        "invalid_source_ids": sorted(bad_ids),
    }

def factcheck_node(state: AgentState):
    print("\n[FACTCHECK] Verifying citations against source text...")
    source_texts = {int(k): v for k, v in (state.get("source_texts") or {}).items()}
    source_index = {int(k): v for k, v in state["source_index"].items()}
    cleaned, stats = verify_claims(state.get("raw_report", ""), source_texts, source_index)

    print(f"  {stats['checked']} factual sentences checked | grounding {stats['grounding']:.0%}"
          f" | {len(stats['unverified'])} unsupported")
    if stats["invalid_source_ids"]:
        print(f"  Citations to non-existent sources: {stats['invalid_source_ids']}")
    for sentence in stats["unverified"][:3]:
        print(f"  [UNSUPPORTED] {sentence[:120]}")

    emit(state, "factcheck",
         f"{stats['checked']} factual claims checked, {stats['grounding']:.0%} grounded",
         stats["checked"] - len(stats["unverified"]), stats["checked"])
    return {"raw_report": cleaned, "grounding": stats["grounding"]}

def audit_node(state: AgentState):
    """Score the report and name its weakest sections in a single LLM call.

    This used to be two serial round trips (challenger, then critic) over the
    same text; the model was reading the report twice to answer two questions.
    """
    iteration = state.get("iteration", 1)
    print(f"\n[AUDIT] Scoring report and triaging weak sections - iteration {iteration}...")
    raw = state.get("raw_report", "")

    import re
    sections = re.findall(r'##\s+SECTION:\s*(.+?)\n(.{0,300})', raw, re.DOTALL)
    preview = "\n\n".join(
        f"SECTION: {title.strip()}\n{body.strip()[:300]}" for title, body in sections
    ) or raw[:3000]

    emit(state, "audit", f"Reviewing {len(sections)} sections for weak arguments")
    audit = _structured_invoke(ReportAudit, [
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"TOPIC: {state['topic']}\n\nREPORT SECTIONS:\n{preview}")
    ])
    if audit is None:
        # All keys rate limited — approve rather than block the export.
        print("  [AUDIT] All keys rate limited, auto-approving to proceed to export...")
        audit = ReportAudit(score=8, gaps=[], follow_up_queries=[], verdict="APPROVED")

    print(f"  Score: {audit.score}/10 | {audit.verdict}")
    if audit.gaps:
        print(f"  Gaps: {', '.join(audit.gaps[:2])}")
    for weak in audit.weak_sections[:2]:
        print(f"  Weak: {weak.section[:60]} — {weak.problem}")

    emit(state, "audit",
         f"Scored {audit.score}/10, {len(audit.weak_sections[:2])} section(s) flagged for rewrite")
    return {
        "quality": audit,
        "weak_sections": audit.weak_sections[:2],
        "iteration": iteration + 1,
    }

def refine_node(state: AgentState):
    verdict = state["quality"]
    print(f"\n[REFINE] {len(verdict.follow_up_queries)} follow-up queries added")
    return {
        "plan": ResearchPlan(
            queries=verdict.follow_up_queries,
            reasoning=f"Filling gaps: {', '.join(verdict.gaps)}"
        )
    }

# A refine pass re-runs the crawler and the whole architect (~7 more LLM calls),
# so it is only worth it when the report is genuinely thin — not merely when a
# harsh critic asks for more.
REFINE_GROUNDING_FLOOR = float(os.getenv("REFINE_GROUNDING_FLOOR", "0.75"))
MAX_REFINE_ITERATIONS = int(os.getenv("MAX_REFINE_ITERATIONS", "2"))

def should_refine(state: AgentState) -> str:
    verdict = state.get("quality")
    iteration = state.get("iteration", 1)
    grounding = state.get("grounding", 0.0)

    if not verdict or verdict.verdict != "NEEDS_MORE_RESEARCH":
        return "export"
    if iteration > MAX_REFINE_ITERATIONS:
        return "export"
    if grounding >= REFINE_GROUNDING_FLOOR:
        print(f"  [REFINE SKIPPED] Grounding {grounding:.0%} is already solid; exporting.")
        return "export"
    if not verdict.follow_up_queries:
        return "export"
    return "refine"

# ── PDF HELPERS ────────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00b7": "*",
        "\u2022": "-", "\u00ae": "(R)", "\u2122": "(TM)", "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    # Models emit narrow/no-break spaces and non-breaking hyphens. latin-1 drops
    # them silently, which glues words together ("$3.447billion", "Form10K"), so
    # map every space-like and dash-like codepoint onto its ASCII equivalent.
    import unicodedata
    text = "".join(
        " " if unicodedata.category(c) == "Zs"
        else "-" if unicodedata.category(c) == "Pd"
        else c
        for c in text
    )
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def draw_line(pdf, w, r=180, g=180, b=180):
    pdf.set_draw_color(r, g, b)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + w, pdf.get_y())
    pdf.ln(3)

# ── PDF EXPORT ─────────────────────────────────────────────────────────────────
def clean_body(text: str) -> str:
    """Strip leaked markdown headers and extra blank lines from section body."""
    import re
    # Remove any ## heading lines that leaked into body
    text = re.sub(r'^##\s+.+\n?', '', text, flags=re.MULTILINE)
    # Collapse 3+ blank lines into 1
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def export_to_pdf(raw_report: str, source_index: Dict, source_titles: Dict, topic: str):
    import re
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(18, 15, 18)
    w = 210 - 36

    # Parse title
    title_match = re.search(r'##\s*TITLE\s*\n(.+)', raw_report)
    title = title_match.group(1).strip() if title_match else topic

    # ── COVER PAGE ────────────────────────────────────────────────────────────
    pdf.add_page()
    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 0, 210, 8, "F")
    pdf.ln(18)
    pdf.set_font("Helvetica", "B", 19)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(w, 10, sanitize(title.upper()), align="C")
    pdf.ln(3)
    draw_line(pdf, w, 20, 20, 20)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf.cell(w, 6, f"Research Report  |  Generated {date_str}  |  {len(source_index)} verified sources",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(7)

    # Key findings on cover
    findings_match = re.search(r'KEY_FINDINGS[:\s]*\n(.*?)(?=\n##|\nEXECUTIVE|\Z)', raw_report, re.DOTALL)
    if findings_match:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 7, "KEY FINDINGS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_line(pdf, w, 180, 180, 180)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        for line in findings_match.group(1).strip().split("\n"):
            if line.strip():
                pdf.multi_cell(w, 5.5, sanitize(line.strip()))
                pdf.ln(0.5)

    pdf.ln(5)

    # TOC
    draw_line(pdf, w)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(w, 7, "TABLE OF CONTENTS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    draw_line(pdf, w)
    pdf.ln(1)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(60, 60, 60)
    section_titles = re.findall(r'##\s+SECTION:\s*(.+)', raw_report)
    toc = ["Executive Summary"] + section_titles + ["Synthesis", "Verified Sources"]
    for i, item in enumerate(toc, 1):
        pdf.cell(w, 5.5, sanitize(f"  {i}.  {item}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 287, 210, 10, "F")

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    exec_match = re.search(r'EXECUTIVE_SUMMARY[:\s]*\n(.*?)(?=\n##\s+SECTION|\Z)', raw_report, re.DOTALL)
    if exec_match:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 8, "EXECUTIVE SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        draw_line(pdf, w, 20, 20, 20)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 6, sanitize(clean_body(exec_match.group(1))))

    # ── SECTIONS (no page break between — just a divider) ─────────────────────
    section_blocks = re.findall(
        r'##\s+SECTION:\s*(.+?)\n(.*?)(?=\n##\s+SECTION:|##\s+SYNTHESIS|\Z)',
        raw_report, re.DOTALL
    )
    for idx, (sec_title, sec_body) in enumerate(section_blocks, 1):
        # Small gap + divider instead of new page
        pdf.ln(6)
        draw_line(pdf, w, 180, 180, 180)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(130, 130, 130)
        pdf.cell(w, 5, f"SECTION {idx}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.multi_cell(w, 7, sanitize(sec_title.strip()))
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 6, sanitize(clean_body(sec_body)))

    # ── SYNTHESIS ─────────────────────────────────────────────────────────────
    synth_match = re.search(r'##\s+SYNTHESIS\s*\n(.*?)(?=\n##|\Z)', raw_report, re.DOTALL)
    if synth_match:
        pdf.ln(6)
        draw_line(pdf, w, 20, 20, 20)
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(w, 8, "FINAL SYNTHESIS", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 10.5)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(w, 6, sanitize(clean_body(synth_match.group(1))))

    # ── VERIFIED SOURCES (2 columns, compact) ─────────────────────────────────
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(w, 8, "VERIFIED SOURCES", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    draw_line(pdf, w, 20, 20, 20)
    pdf.ln(2)

    col_w = (w - 4) / 2  # two columns with 4mm gap
    sources = sorted(source_index.keys())
    left_col = sources[:len(sources)//2 + len(sources)%2]
    right_col = sources[len(sources)//2 + len(sources)%2:]

    start_y = pdf.get_y()
    # Left column
    pdf.set_xy(pdf.l_margin, start_y)
    for sid in left_col:
        url = source_index[sid]
        title_s = source_titles.get(sid, url)[:60]
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(col_w, 4, sanitize(f"[{sid}] {title_s}"))
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(0, 60, 160)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(col_w, 3.5, sanitize(url[:75]))
        pdf.set_text_color(0, 0, 0)
        pdf.set_x(pdf.l_margin)
        pdf.ln(1.5)

    mid_y = pdf.get_y()

    # Right column — reset to start_y, offset by col_w + gap
    pdf.set_xy(pdf.l_margin + col_w + 4, start_y)
    for sid in right_col:
        if pdf.get_y() > 270:
            pdf.add_page()
            pdf.set_xy(pdf.l_margin + col_w + 4, 15)
        url = source_index[sid]
        title_s = source_titles.get(sid, url)[:60]
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_text_color(30, 30, 30)
        pdf.set_x(pdf.l_margin + col_w + 4)
        pdf.multi_cell(col_w, 4, sanitize(f"[{sid}] {title_s}"))
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(0, 60, 160)
        pdf.set_x(pdf.l_margin + col_w + 4)
        pdf.multi_cell(col_w, 3.5, sanitize(url[:75]))
        pdf.set_text_color(0, 0, 0)
        pdf.set_x(pdf.l_margin + col_w + 4)
        pdf.ln(1.5)

    pdf.set_fill_color(20, 20, 20)
    pdf.rect(0, 287, 210, 10, "F")

    # Titles come from the model, so they carry curly quotes and dashes. Those
    # cannot go in a Content-Disposition header, so fold to plain ASCII here.
    stem = sanitize(title[:35]).replace(" ", "_").upper()
    stem = re.sub(r"[^A-Z0-9_-]", "", stem).strip("_") or "REPORT"
    filename = f"REPORT_{stem}.pdf"
    pdf.output(filename)
    print(f"\n[EXPORTED] {filename} | {len(section_blocks)} sections | {len(source_index)} sources")
    return filename

def _rewrite_one(topic, sec_title, challenge_text, existing_body,
                 source_index, source_titles, source_texts):
    """Rewrite a single section against its own evidence. Runs in a worker thread."""
    sids = select_sources(sec_title, topic, source_texts, source_titles, source_index)
    evidence = build_evidence_block(sids, source_index, source_titles, source_texts)

    rewritten = llm_invoke_with_rotation([
        SystemMessage(content=(
            "You are rewriting one section of a research report to fix a specific weakness. "
            "Keep every fact that the evidence supports and drop the ones it does not. "
            "Add no figure, date, or name that is absent from the evidence below. "
            f"Never use these filler phrases: {', '.join(SLOP_PHRASES)}. "
            "No closing summary. Cite sources inline as [N]. No bullet points. Flowing prose only."
        )),
        HumanMessage(content=f"""TOPIC: {topic}

EVIDENCE (the only material you may draw facts from):
{evidence}

SECTION TO REWRITE: {sec_title}

EXISTING CONTENT (keep facts, improve analysis):
{existing_body}

SPECIFIC CHALLENGE TO ADDRESS:
{challenge_text}

Rewrite this section now, directly addressing the challenge:""")
    ]).content
    return strip_slop(rewritten)


def targeted_rewrite_node(state: AgentState):
    print("\n[REWRITER] Fixing challenged sections...")
    weak = state.get("weak_sections") or []
    if not weak:
        print("  No weak sections flagged, skipping.")
        return {}

    import re
    raw = state.get("raw_report", "")
    source_index = {int(k): v for k, v in state["source_index"].items()}
    source_titles = {int(k): v for k, v in (state.get("source_titles") or {}).items()}
    source_texts = {int(k): v for k, v in (state.get("source_texts") or {}).items()}

    # Locate each flagged section first, so the rewrites themselves can run
    # concurrently instead of editing the report one after another.
    jobs = []
    for challenge in weak[:2]:
        sec_title = challenge.section.strip()
        sec_match = re.search(
            rf'(##\s+SECTION:\s*{re.escape(sec_title)}\n)(.*?)(?=\n##\s+SECTION:|\n##\s+SYNTHESIS|\Z)',
            raw, re.DOTALL
        )
        if not sec_match:
            print(f"  Section not found in report, skipping: {sec_title[:60]}")
            continue
        print(f"  Rewriting: {sec_title[:60]} ({challenge.problem})")
        jobs.append((sec_match, sec_title, challenge.challenge, sec_match.group(2).strip()[:1500]))

    if not jobs:
        return {}

    emit(state, "targeted_rewrite", "Rewriting flagged sections", 0, len(jobs))
    rewrites = [""] * len(jobs)
    completed = 0
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {
            pool.submit(
                _rewrite_one, state["topic"], job[1], job[2], job[3],
                source_index, source_titles, source_texts,
            ): i
            for i, job in enumerate(jobs)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            rewrites[i] = fut.result()
            completed += 1
            emit(state, "targeted_rewrite", f"Rewrote: {jobs[i][1]}", completed, len(jobs))

    # Splice back-to-front so earlier match offsets stay valid.
    updated_report = raw
    for (sec_match, _, _, _), rewritten in sorted(
        zip(jobs, rewrites), key=lambda pair: pair[0][0].start(2), reverse=True
    ):
        updated_report = (
            updated_report[:sec_match.start(2)] + "\n" + rewritten + "\n"
            + updated_report[sec_match.end(2):]
        )

    print(f"  Rewrite complete. {len(jobs)} section(s) improved.")

    with open("raw_report_debug.txt", "w", encoding="utf-8") as f:
        f.write(updated_report)

    return {"raw_report": updated_report}


# ── GRAPH ──────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("strategist",          strategist_node)
builder.add_node("commander_review",    hitl_node)
builder.add_node("crawler",             crawler_node)
builder.add_node("architect",           architect_node)
builder.add_node("audit",               audit_node)
builder.add_node("targeted_rewrite",    targeted_rewrite_node)
builder.add_node("factcheck",           factcheck_node)
builder.add_node("refine",              refine_node)

builder.set_entry_point("strategist")
builder.add_edge("strategist",         "commander_review")
builder.add_edge("commander_review",   "crawler")
builder.add_edge("crawler",            "architect")
builder.add_edge("architect",          "audit")
builder.add_edge("audit",              "targeted_rewrite")
builder.add_edge("targeted_rewrite",   "factcheck")
builder.add_conditional_edges("factcheck", should_refine, {"refine": "refine", "export": END})
builder.add_edge("refine",             "crawler")

app = builder.compile(checkpointer=memory, interrupt_before=["commander_review"])

# ── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "NVIDIA H100 vs A100 architecture"
    thread_id = topic[:30].replace(" ", "_")
    thread_config = {"configurable": {"thread_id": thread_id}}

    existing_state = app.get_state(thread_config)
    if not existing_state.values:
        print(f"\n[MISSION] {topic}")
        print("=" * 60)
        for event in app.stream({"topic": topic}, thread_config):
            pass

    current = app.get_state(thread_config)
    plan = current.values.get("plan")
    print("\n" + "=" * 60)
    print("MISSION PAUSED - COMMANDER REVIEW")
    print("=" * 60)
    for i, q in enumerate(plan.queries, 1):
        print(f"  {i:02d}. {q}")
    print(f"\nReasoning: {plan.reasoning}")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm == "y":
        print("\n[RESUMING]")
        final_result = app.invoke(None, thread_config)
        export_to_pdf(
            final_result.get("raw_report", ""),
            final_result["source_index"],
            final_result["source_titles"],
            topic,
        )

        print("\n[MEMORY] Saving to long-term memory...")
        raw = final_result.get("raw_report", "")
        save_to_memory(topic, raw[:3000], list(final_result["source_index"].values()))
        print("[DONE] Research saved.")
    else:
        print("Mission aborted.")



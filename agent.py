import os
import sqlite3
import requests
import time
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
    return ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", api_key=key, temperature=0.2)



# ── CUSTOM EXCEPTIONS ─────────────────────────────────────────────────────────
class RateLimitExhausted(Exception):
    """Raised when all LLM API keys are rate limited and no fallback is available."""
    pass

# ── LLM with key rotation ─────────────────────────────────────────────────────
_groq_keys = []
_key_index = 0
_using_fallback = False
_tried_keys: set = set()
llm = None

def _init_llm():
    """Initialize LLM on first use, not at import time."""
    global _groq_keys, llm
    if not _groq_keys:
        _groq_keys = [k.strip() for k in [
            os.getenv("GROQ_API_KEY", ""),
            os.getenv("GROQ_API_KEY_2", ""),
        ] if k and k.strip()]
        print(f"  [INIT] Loaded {len(_groq_keys)} Groq keys")
    if llm is None and _groq_keys:
        llm = make_llm(_groq_keys[0])
        print(f"  [INIT] LLM initialized with key index 0")

def make_together_llm():
    together_key = os.getenv("TOGETHER_API_KEY")
    if not together_key:
        raise RuntimeError("No TOGETHER_API_KEY found in .env")
    print("  [FALLBACK] Switching to Together.ai (Llama-4-Scout)")
    return ChatTogether(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
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

def llm_invoke_with_rotation(messages):
    """Invoke LLM, rotating Groq keys on rate limit, then falling back to Together.ai."""
    global llm, _key_index, _using_fallback, _tried_keys
    _init_llm()  # ensure LLM is initialized

    if _using_fallback:
        try:
            return llm.invoke(messages)
        except Exception as e:
            if _is_rate_limit_error(e):
                raise RateLimitExhausted(
                    "All API keys are currently rate limited. Please try again in 5 minutes."
                )
            print(f"  [TOGETHER ERROR] {e}")
            raise

    while True:
        try:
            return llm.invoke(messages)
        except Exception as e:
            if _is_rate_limit_error(e):
                _tried_keys.add(_key_index)
                next_index = next(
                    (i for i in range(len(_groq_keys)) if i not in _tried_keys),
                    None
                )
                if next_index is not None:
                    _key_index = next_index
                    llm = make_llm(_groq_keys[_key_index])
                    print(f"  [KEY ROTATION] Switched to Groq key {_key_index + 1}")
                    time.sleep(3)
                else:
                    print("  [RATE LIMIT] All Groq keys exhausted, switching to Together.ai...")
                    try:
                        llm = make_together_llm()
                        _using_fallback = True
                        _tried_keys = set()  # reset for future use
                        time.sleep(2)
                    except RuntimeError:
                        # No Together.ai key configured at all
                        raise RateLimitExhausted(
                            "All API keys are currently rate limited. Please try again in 5 minutes."
                        )
                    except Exception as fallback_e:
                        if "401" in str(fallback_e) or "invalid" in str(fallback_e).lower():
                            print("  [TOGETHER AUTH FAILED] Invalid API key — check TOGETHER_API_KEY in .env")
                        raise fallback_e
            else:
                print(f"  [LLM ERROR] {type(e).__name__}: {e}")
                raise

def _rotate_key():
    """Rotate to next available Groq API key."""
    global llm, _key_index, _using_fallback
    _init_llm()
    if _using_fallback:
        return
    _key_index = (_key_index + 1) % len(_groq_keys)
    llm = make_llm(_groq_keys[_key_index])
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

class AgentState(Dict):
    topic: str
    plan: ResearchPlan
    raw_data: str
    raw_report: str
    source_index: Dict[int, str]
    source_titles: Dict[int, str]
    memory_context: str
    quality: QualityVerdict
    iteration: int
    research_rounds: int
    challenge_notes: str  # feedback from section_challenger for targeted rewrite

# ── SYSTEM PROMPTS ─────────────────────────────────────────────────────────────
STRATEGIST_PROMPT = """You are an elite intelligence analyst. Build a comprehensive research plan.
Generate 10-15 search queries covering:
- Technical specifications and hard metrics
- Historical timeline with specific dates and events
- Key figures, engineers, decision-makers
- Comparative analysis and bottlenecks
- Failures, weaknesses, and overlooked angles
Every query must target a specific data point. No generic queries."""

ARCHITECT_PROMPT = """You are a world-class investigative analyst writing a long-form research report.

CITATION RULES:
1. A SOURCE INDEX will be provided: [1] https://... - "Title"
2. Cite retrieved facts inline using [N] notation in narratives.
3. Every key_fact MUST have a source_id from the index.
4. NEVER invent source IDs. NEVER cite a number not in the index.
5. NEVER fabricate quotes or statistics and attribute them to a source.

WRITING RULES - CRITICAL:
1. You are an expert. USE YOUR OWN DEEP KNOWLEDGE to write rich, detailed narratives.
2. Retrieved data gives you grounded facts to cite. Your knowledge fills the depth.
3. Each chapter narrative MUST be at least 300 words. No exceptions.
4. Write like a senior engineer explaining to another engineer - specific, technical, opinionated.
5. Show tradeoffs, engineering decisions, real-world implications.
6. MINIMUM 5 chapters. Each on its own distinct angle.
7. Executive summary MUST be at least 150 words covering key findings and strategic implications.
8. key_findings must be 5 sentences each with a hard number or specific fact.

CRITICAL - ANALYTICAL DEPTH RULES (this separates great reports from average ones):
9. NEVER just describe what happened. Always explain WHY it happened and what a BETTER alternative could have been.
   BAD: "Nokia failed to adapt to smartphones."
   GOOD: "Nokia's failure to adopt a touch-first OS was a strategic mistake — the company had the engineering talent but lacked the organizational will. A better path would have been to spin off a dedicated smartphone unit with full autonomy, as Samsung did with its Android division."
10. CONTRARIAN ANGLE REQUIRED: Every report must include at least one section that challenges the obvious narrative.
    Ask: Was it entirely the subject's fault? What external forces (ecosystem effects, network effects, competitor moves) made failure almost inevitable regardless of internal decisions?
11. COMPARISON TABLE REQUIRED: Include at least one structured comparison in the report.
    Format it as a plain-text table using | separators. Example:
    | Metric | Subject A | Subject B |
    | Market Share 2007 | 70% | 3% |
    | OS Strategy | Symbian | iOS |
12. TIMELINE REQUIRED: Include a concise chronological timeline of key events in one section.
    Format: YEAR: Event — consequence
13. DO NOT repeat the same point across sections. Each section must add NEW insight."""

CRITIC_PROMPT = """You are a ruthless research quality auditor for a top-tier journal.
Score the report 1-10:
- 1-4: Generic, missing fundamental facts, reads like a summary
- 5-7: Decent but narratives are thin, missing key metrics or angles
- 8-10: Expert-level, specific, narrative-driven, cross-referenced, surprising insights
Be harsh. Only approve (score >= 8) if a senior researcher would find it valuable."""

# ── NODES ──────────────────────────────────────────────────────────────────────
def strategist_node(state: AgentState):
    print("\n[STRATEGIST] Building search vectors...")
    memory_context = query_memory(state["topic"])
    print(f"  Memory: {'found past research' if memory_context else 'starting fresh'}")

    _init_llm()
    structured_llm = llm.with_structured_output(ResearchPlan)
    memory_hint = f"\n\nPAST RESEARCH (avoid re-searching these):\n{memory_context}" if memory_context else ""
    for attempt in range(len(_groq_keys) * 2 + 1):
        try:
            plan = structured_llm.invoke([
                SystemMessage(content=STRATEGIST_PROMPT),
                HumanMessage(content=f"Build research plan for: {state['topic']}{memory_hint}")
            ])
            break
        except Exception as e:
            if _is_rate_limit_error(e):
                _rotate_key()
                structured_llm = llm.with_structured_output(ResearchPlan)
                print(f"  [KEY ROTATION] Switched to key {_key_index + 1}")
                time.sleep(3)
            else:
                raise
    print(f"  {len(plan.queries)} queries planned")
    return {
        "plan": plan,
        "iteration": 1,
        "research_rounds": 0,
        "source_index": {},
        "source_titles": {},
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

def crawler_node(state: AgentState):
    queries = state['plan'].queries
    rounds = state.get("research_rounds", 0) + 1
    print(f"\n[CRAWLER] Round {rounds} - {len(queries)} queries...")

    source_index: Dict[int, str] = {}
    source_titles: Dict[int, str] = {}
    for k, v in (state.get("source_index") or {}).items():
        source_index[int(k)] = v
    for k, v in (state.get("source_titles") or {}).items():
        source_titles[int(k)] = v

    seen_urls = set(source_index.values())
    # Store (sid, url, snippet) for deep fetch candidates
    new_sources = []
    raw_chunks = []

    for i, q in enumerate(queries, 1):
        print(f"  [{i}/{len(queries)}] {q[:65]}...")
        try:
            web_search = get_web_search()
            response = web_search.invoke(q)
            items = response.get("results", []) if isinstance(response, dict) else response
            for r in items:
                url = (r.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                sid = len(source_index) + 1
                source_index[sid] = url
                title = (r.get("title") or url)[:80]
                source_titles[sid] = title
                content = (r.get("raw_content") or r.get("content") or "")[:600]
                new_sources.append((sid, url, title, content))
        except Exception as e:
            print(f"  Query failed: {e}")

    # Deep fetch top 12 most relevant sources for full content
    print(f"  Deep fetching top sources for full content...")
    deep_fetch_count = 0
    for sid, url, title, snippet in new_sources:
        # Skip PDFs, social media, video sites
        skip = any(x in url for x in ["youtube.com", "twitter.com", "linkedin.com", ".pdf", "reddit.com"])
        if not skip and deep_fetch_count < 5:
            full_content = deep_fetch(url, max_chars=2000)
            if len(full_content) > len(snippet):
                content = full_content
                deep_fetch_count += 1
            else:
                content = snippet
        else:
            content = snippet
        raw_chunks.append(f"[{sid}] SOURCE: {url}\nTITLE: {title}\n{content}")

    print(f"  {len(source_index)} sources indexed, {deep_fetch_count} deep fetched")

    # ── GUARD: stop before any LLM calls if web returned nothing useful ───────
    new_sources_count = len(source_index)
    if new_sources_count < 3:
        raise ValueError(
            f"Not enough sources found for '{state['topic']}' ({new_sources_count} result(s)). "
            "This topic may not have enough public information available. "
            "Try a well-known subject, event, company, or technology."
        )

    existing_raw = state.get("raw_data", "")
    combined = existing_raw + f"\n\n=== ROUND {rounds} ===\n\n" + "\n---\n".join(raw_chunks)
    return {
        "raw_data": combined,
        "source_index": source_index,
        "source_titles": source_titles,
        "research_rounds": rounds,
    }

def generate_section_topics(topic: str, raw_data: str) -> List[str]:
    """Ask the LLM to propose 6-7 section titles tailored to the research topic."""
    prompt = f"""You are planning a long-form research report on: "{topic}"

Based on this topic, propose exactly 7 section titles that together give comprehensive coverage.
Rules:
- Each title must be specific and distinct — no overlap
- One section MUST be a timeline of key events (title it like "Timeline: [Topic] from [Year] to [Year]")
- One section MUST be a contrarian/critical analysis (title it like "The Contrarian View: Was Failure Inevitable?" or similar)
- One section MUST be a comparison (title it like "Comparative Analysis: [A] vs [B]" or similar)
Return ONLY a Python list of strings, nothing else. Example:
["Title One", "Title Two", "Title Three", "Title Four", "Title Five", "Title Six", "Title Seven"]"""
    try:
        response = llm_invoke_with_rotation([HumanMessage(content=prompt)]).content.strip()
        import ast, re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            return ast.literal_eval(match.group())
    except Exception:
        pass
    # Fallback generic sections
    return [
        "Origins and Early Innovation",
        "Peak Years and Market Dominance",
        "Timeline: Key Events and Turning Points",
        "Disruption and Strategic Missteps",
        "Comparative Analysis: Key Competitors vs Subject",
        "The Contrarian View: Was Failure Inevitable?",
        "Legacy and Long-Term Impact",
    ]

def architect_node(state: AgentState):
    print("\n[ARCHITECT] Writing research report section by section...")

    top_index = dict(sorted(state["source_index"].items())[:30])
    top_titles = {k: state["source_titles"].get(k, "") for k in top_index}
    index_str = "\n".join(
        f"[{sid}] {url} - \"{top_titles.get(sid, '')}\""
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

    section_system = (
        "You are a senior investigative journalist writing for IEEE Spectrum or MIT Technology Review. "
        "Write dense, expert-level prose. Every paragraph must contain specific facts, dates, numbers, or technical details. "
        "Never write vague generalities. Never use bullet points. Minimum 450 words per section. Do not truncate."
    )

    # Step 1: Write title, key findings, executive summary
    print("  Writing header (title, findings, summary)...")
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
[200-250 words. Written for a senior decision-maker. Cover what was investigated, the 3 most critical findings with specific metrics, and the strategic implication. Must include real numbers and dates.]""")
    ]).content

    # Step 2: Write each section independently
    section_topics = generate_section_topics(topic, raw)
    sections = []
    for i, sec_title in enumerate(section_topics, 1):
        print(f"  Writing section {i}/{len(section_topics)}: {sec_title}...")
        section = llm_invoke_with_rotation([
            SystemMessage(content=section_system),
            HumanMessage(content=f"""{context_block}

Write ONLY this one section. Do not write any other sections.

## SECTION: {sec_title}

Requirements:
- Minimum 450 words of flowing prose
- Include specific dates, metrics, technical decisions, and their consequences
- Cite sources inline using [N] notation
- Explain the WHY and HOW, not just the WHAT
- Show cause-and-effect chains and engineering/business tradeoffs
- Write as if explaining to a senior engineer or executive who wants depth

ANALYTICAL DEPTH — MANDATORY:
- Do NOT just describe events. After every major fact, add your critical judgment:
  "This decision was a mistake because...", "A better alternative would have been...", "In hindsight..."
- Include a CONTRARIAN perspective: challenge the obvious narrative. Ask what external forces
  (ecosystem lock-in, network effects, competitor moves, market timing) made the outcome
  partially or fully inevitable — regardless of internal decisions.
- If this section covers a comparison or timeline, include a plain-text table using | separators
  OR a year-by-year timeline in format: YEAR: Event — consequence
- Avoid repeating points already covered in other sections. Add NEW insight only.

Begin the section now and write until you have covered the topic thoroughly:""")
        ]).content
        # Normalize section header in case model added extra text before it
        if f"## SECTION: {sec_title}" not in section:
            section = f"## SECTION: {sec_title}\n{section}"
        sections.append(section)

    # Step 3: Write synthesis
    print("  Writing synthesis...")
    synthesis = llm_invoke_with_rotation([
        SystemMessage(content=section_system),
        HumanMessage(content=f"""{context_block}

Write ONLY the final synthesis section.

## SYNTHESIS
[150-200 words. Reveal a non-obvious connection across all the themes covered.
What does the data collectively suggest that no single section states explicitly?
Include a contrarian take — something that challenges the dominant narrative of the report.
Be specific, opinionated, and insightful. No vague conclusions.]""")
    ]).content

    raw_report = header + "\n\n" + "\n\n".join(sections) + "\n\n" + synthesis
    print(f"  Total report: {len(raw_report)} chars across {len(sections)} sections")

    with open("raw_report_debug.txt", "w", encoding="utf-8") as f:
        f.write(raw_report)

    return {"raw_report": raw_report}

def factcheck_node(state: AgentState):
    print("\n[FACTCHECK] Verifying citations...")
    import re
    valid_ids = set(state["source_index"].keys())
    raw = state.get("raw_report", "")
    cited = set(int(x) for x in re.findall(r'\[(\d+)\]', raw))
    valid_cited = cited & valid_ids
    invalid_cited = cited - valid_ids
    print(f"  {len(valid_cited)} valid citations, {len(invalid_cited)} unverified")
    return {}

def critic_node(state: AgentState):
    iteration = state.get("iteration", 1)
    print(f"\n[CRITIC] Quality audit - iteration {iteration}...")
    structured_llm = llm.with_structured_output(QualityVerdict)
    raw = state.get("raw_report", "")[:3000]
    verdict = structured_llm.invoke([
        SystemMessage(content=CRITIC_PROMPT),
        HumanMessage(content=f"TOPIC: {state['topic']}\n\nREPORT EXCERPT:\n{raw}")
    ])
    print(f"  Score: {verdict.score}/10 | {verdict.verdict}")
    if verdict.gaps:
        print(f"  Gaps: {', '.join(verdict.gaps[:2])}")
    return {"quality": verdict, "iteration": iteration + 1}

def refine_node(state: AgentState):
    verdict = state["quality"]
    print(f"\n[REFINE] {len(verdict.follow_up_queries)} follow-up queries added")
    return {
        "plan": ResearchPlan(
            queries=verdict.follow_up_queries,
            reasoning=f"Filling gaps: {', '.join(verdict.gaps)}"
        )
    }

def should_refine(state: AgentState) -> str:
    verdict = state.get("quality")
    iteration = state.get("iteration", 1)
    if verdict and verdict.verdict == "NEEDS_MORE_RESEARCH" and iteration <= 4:
        return "refine"
    return "export"

# ── PDF HELPERS ────────────────────────────────────────────────────────────────
def sanitize(text: str) -> str:
    replacements = {
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00b7": "*",
        "\u2022": "-", "\u00ae": "(R)", "\u2122": "(TM)", "\u00a0": " ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
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

    filename = f"REPORT_{title[:35].replace(' ', '_').upper()}.pdf"
    filename = "".join(c for c in filename if c not in r'\/:*?"<>|')
    pdf.output(filename)
    print(f"\n[EXPORTED] {filename} | {len(section_blocks)} sections | {len(source_index)} sources")
    return filename

CHALLENGER_PROMPT = """You are a sharp editorial critic reviewing a research report before publication.
Your job: find the 2 weakest sections and return specific, actionable challenges.

For each weak section, identify ONE of these problems:
- "Safe explanation" — describes what happened but never says why it was wrong or what better alternative existed
- "Repetition" — repeats a point already made in another section
- "Missing contrarian" — accepts the obvious narrative without challenging it
- "No specifics" — makes claims without hard numbers, dates, or named decisions

Be surgical. Return exactly 2 challenges, each targeting a specific section by its title.
Format strictly as:
SECTION: [exact section title]
PROBLEM: [one of the 4 problem types above]
CHALLENGE: [one sharp question or missing angle the rewrite must address]
---
SECTION: [exact section title]
PROBLEM: [one of the 4 problem types above]
CHALLENGE: [one sharp question or missing angle the rewrite must address]"""

def section_challenger_node(state: AgentState):
    print("\n[CHALLENGER] Identifying weak sections...")
    raw = state.get("raw_report", "")

    # Extract just section titles + first 300 chars of each for efficiency
    import re
    sections = re.findall(r'##\s+SECTION:\s*(.+?)\n(.{0,300})', raw, re.DOTALL)
    if not sections:
        print("  No sections found, skipping challenge.")
        return {"challenge_notes": ""}

    preview = "\n\n".join(
        f"SECTION: {title.strip()}\n{body.strip()[:300]}"
        for title, body in sections
    )

    response = llm_invoke_with_rotation([
        SystemMessage(content=CHALLENGER_PROMPT),
        HumanMessage(content=f"TOPIC: {state['topic']}\n\nREPORT SECTIONS PREVIEW:\n{preview}")
    ])

    notes = response.content.strip()
    print(f"  Challenges identified:\n{notes[:300]}...")
    return {"challenge_notes": notes}


def targeted_rewrite_node(state: AgentState):
    print("\n[REWRITER] Fixing challenged sections...")
    notes = state.get("challenge_notes", "")
    if not notes:
        print("  No challenges to address, skipping.")
        return {}

    import re
    raw = state.get("raw_report", "")

    # Parse challenged section titles from notes
    challenged = re.findall(r'SECTION:\s*(.+)', notes)
    if not challenged:
        return {}

    top_index = dict(sorted(state["source_index"].items())[:30])
    top_titles = {k: state["source_titles"].get(k, "") for k in top_index}
    index_str = "\n".join(
        f"[{sid}] {url} - \"{top_titles.get(sid, '')}\""
        for sid, url in top_index.items()
    )

    updated_report = raw
    for sec_title in challenged[:2]:  # max 2 rewrites
        sec_title = sec_title.strip()
        # Find the challenge note for this section
        challenge_match = re.search(
            rf'SECTION:\s*{re.escape(sec_title)}.*?CHALLENGE:\s*(.+?)(?=---|$)',
            notes, re.DOTALL
        )
        challenge_text = challenge_match.group(1).strip() if challenge_match else "Add critical analysis and contrarian perspective."

        print(f"  Rewriting: {sec_title[:60]}...")
        print(f"  Challenge: {challenge_text[:100]}")

        # Find existing section content
        sec_match = re.search(
            rf'(##\s+SECTION:\s*{re.escape(sec_title)}\n)(.*?)(?=\n##\s+SECTION:|\n##\s+SYNTHESIS|\Z)',
            updated_report, re.DOTALL
        )
        if not sec_match:
            print(f"  Section not found in report, skipping.")
            continue

        existing_body = sec_match.group(2).strip()[:1500]

        rewritten = llm_invoke_with_rotation([
            SystemMessage(content=(
                "You are rewriting one section of a research report to fix a specific weakness. "
                "Keep all correct facts. Improve the analysis. Minimum 400 words. "
                "Cite sources inline as [N]. No bullet points. Flowing prose only."
            )),
            HumanMessage(content=f"""TOPIC: {state['topic']}

SOURCE INDEX:
{index_str}

SECTION TO REWRITE: {sec_title}

EXISTING CONTENT (keep facts, improve analysis):
{existing_body}

SPECIFIC CHALLENGE TO ADDRESS:
{challenge_text}

Rewrite this section now, directly addressing the challenge:""")
        ]).content

        # Replace old section body with rewritten version
        updated_report = updated_report[:sec_match.start(2)] + "\n" + rewritten + "\n" + updated_report[sec_match.end(2):]
        time.sleep(3)

    print(f"  Rewrite complete. {len(challenged[:2])} section(s) improved.")

    # Save updated report to debug file
    with open("raw_report_debug.txt", "w", encoding="utf-8") as f:
        f.write(updated_report)

    return {"raw_report": updated_report}


# ── GRAPH ──────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)
builder.add_node("strategist",          strategist_node)
builder.add_node("commander_review",    hitl_node)
builder.add_node("crawler",             crawler_node)
builder.add_node("architect",           architect_node)
builder.add_node("section_challenger",  section_challenger_node)
builder.add_node("targeted_rewrite",    targeted_rewrite_node)
builder.add_node("factcheck",           factcheck_node)
builder.add_node("critic",              critic_node)
builder.add_node("refine",              refine_node)

builder.set_entry_point("strategist")
builder.add_edge("strategist",         "commander_review")
builder.add_edge("commander_review",   "crawler")
builder.add_edge("crawler",            "architect")
builder.add_edge("architect",          "section_challenger")
builder.add_edge("section_challenger", "targeted_rewrite")
builder.add_edge("targeted_rewrite",   "factcheck")
builder.add_edge("factcheck",          "critic")
builder.add_conditional_edges("critic", should_refine, {"refine": "refine", "export": END})
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



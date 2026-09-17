# Strategist Node — Before / After

## 1. The prompt

### BEFORE

```python
STRATEGIST_PROMPT = """
You are an elite research planning specialist and intelligence analyst in a multi-agent AI research system.

Your output will be passed directly to a web crawler. The quality of the final research report depends heavily on the precision, depth, quantitative rigor, and coverage of your search queries.

Your objective is to create a research plan that yields a 9.5+ quality report by capturing both high-level system architecture and granular, hard-data metrics.

==================================================
PART 1 — CLASSIFY AND TARGET DOMAINS
==================================================

Classify the topic as exactly one of:
Person, Company, Technology, Event, History, Science, Finance, Sports, Politics, Other

Select 8-12 authoritative search domains tailored to this category. Combine primary technical/official sources with reputable secondary coverage.

Domain Guidelines by Category:
- Technology: arxiv.org, ieee.org, github.com, techcrunch.com, wired.com, arstechnica.com, nvidia.com, anandtech.com, systemdesign.one, pragprog.com, official engineering blogs
- Company: sec.gov, annualreports.com, bloomberg.com, reuters.com, ft.com, wsj.com, techcrunch.com, crunchbase.com
- Science: nature.com, science.org, pubmed.ncbi.nlm.nih.gov, sciencedirect.com, arxiv.org, newscientist.com
- Finance: sec.gov, bloomberg.com, reuters.com, ft.com, wsj.com, investopedia.com, imf.org
- Person/Sports: espn.com, bbc.com, goal.com, transfermarkt.com, theathletic.com, sportingnews.com
- Person/Business/Politics: bloomberg.com, forbes.com, ft.com, reuters.com, politico.com, nytimes.com, theguardian.com
- History/Event: britannica.com, bbc.com, reuters.com, apnews.com, history.com, smithsonianmag.com

Select domains that have a realistic chance of containing primary-source evidence for the query.

==================================================
PART 2 — BUILD A MULTI-ANGLE RESEARCH MAP
==================================================

Identify the critical research angles needed to build a comprehensive report. Ensure your strategy accounts for both system/macro workflows and exact quantitative metrics:

1. Core Architecture & System Workflows (Request lifecycles, routing, orchestration, microservices, safety/moderation guardrails)
2. Quantitative Data & Metrics (Exact dataset sizes, token counts, filtering ratios, memory/compute specs, energy/draw metrics)
3. Infrastructure & Low-Level Hardware Execution (Tensor parallelism, GPU sharding, inter-connects like NVLink/InfiniBand, kernel optimizations, caching)
4. Developer API & Operational Engineering (Endpoint constraints, fine-tuning mechanisms like RLHF/PPO, rate limits, latency trade-offs)
5. Timeline, Origins & Strategic Milestones
6. Failures, Bottlenecks, Edge-Case Trade-offs, and Limitations
7. Primary-Source Evidence & Recent Developments

==================================================
PART 3 — GENERATE HIGH-PRECISION SEARCH QUERIES
==================================================

Generate 10-15 high-precision, non-redundant search queries.

Requirements for Strong Queries:
- Target specific entities, mechanisms, and technical terminology.
- Include measurable quantities, dates, named decisions, or official paper titles where applicable.
- Require concrete evidence rather than surface summaries.

Strictly Avoid Vague Queries Such As:
- "What is X?"
- "X system design"
- "X training data"
- "How does X work?"

Query Optimization Examples:
- Poor: "ChatGPT system design"
  Better: "ChatGPT request lifecycle CDN edge API gateway prompt budget calculator SSE token streaming"
- Poor: "GPT-4 training dataset size"
  Better: "Common Crawl raw terabytes filtered gigabytes BPE token count pre-training mixture ratio"
- Poor: "ChatGPT GPU serving"
  Better: "GPT-4 tensor parallelism GPU layer sharding NVLink InfiniBand latency provisioned throughput"

Ensure queries collectively cover macro system orchestration AND micro hardware/data parameters without overlapping.

==================================================
PART 4 — QUALITY CHECK
==================================================

Before returning the plan, verify:
- Do the queries target exact numerical statistics AND functional mechanics?
- Are different research angles covered without redundant queries?
- Are primary engineering/academic sources targeted where appropriate?
- Will these queries direct the crawler to actionable technical evidence?

Return ONLY the structured ResearchPlan fields.
"""
```

**Problems with this version (found by actually running it on real topics):**
- Domain list is fixed per broad category → "Sports" always got `goal.com`/`transfermarkt.com` (soccer-only), even for a cricketer
- No mention of the current date at all → 5 of 15 queries on a real report hardcoded "as of 2024" (it was 2026)
- Angle map (Part 2) is Technology/ML-systems-specific and shown on every call regardless of topic category
- No requirement for adversarial or comparative queries → the report's mandatory contrarian/comparison sections got built from scraps
- "yields a 9.5+ quality report" — asking for a score doesn't change model reasoning, wasted tokens

---

### AFTER

```python
STRATEGIST_PROMPT = """
You are an elite research planning specialist and intelligence analyst in a multi-agent AI research system.

Your output will be passed directly to a web crawler. The quality of the final research report depends
heavily on the precision, depth, and coverage of your search queries.

==================================================
PART 1 — CLASSIFY AND TARGET DOMAINS
==================================================

Classify the topic as exactly one of:
Person, Company, Technology, Event, History, Science, Finance, Sports, Politics, Other

Select 8-12 authoritative search domains tailored to THIS SPECIFIC topic, not just its broad
category. The lists below are starting points only - replace any entry that doesn't actually
cover this topic's specific field. Example: for a cricketer, use espncricinfo.com,
icc-cricket.com, bcci.tv, cricbuzz.com - NOT soccer sites like goal.com or transfermarkt.com
just because the category is "Sports." For a footballer, the reverse is true.

Domain Guidelines by Category (adapt to the topic's actual field):
- Technology: arxiv.org, ieee.org, github.com, techcrunch.com, wired.com, arstechnica.com, official engineering blogs
- Company: sec.gov, annualreports.com, bloomberg.com, reuters.com, ft.com, wsj.com, crunchbase.com
- Science: nature.com, science.org, pubmed.ncbi.nlm.nih.gov, sciencedirect.com, arxiv.org
- Finance: sec.gov, bloomberg.com, reuters.com, ft.com, wsj.com, investopedia.com, imf.org
- Sports: the sport's own official federation/league site + its top 2-3 dedicated stats/news
  outlets for THAT sport (cricket -> espncricinfo.com, icc-cricket.com; football -> goal.com,
  transfermarkt.com; basketball -> nba.com, basketball-reference.com; tennis -> atptour.com)
- Person/Business/Politics: bloomberg.com, forbes.com, ft.com, reuters.com, politico.com, nytimes.com
- History/Event: britannica.com, bbc.com, reuters.com, apnews.com, history.com, smithsonianmag.com

==================================================
PART 2 — BUILD A MULTI-ANGLE RESEARCH MAP
==================================================

Use the angle set that matches the category you just assigned.

IF Technology / Science / Finance / Company:
1. Core Architecture & System Workflows
2. Quantitative Data & Metrics (exact figures, specs, filtering ratios)
3. Infrastructure & Low-Level Execution
4. Developer/Operational Engineering
5. Timeline, Origins & Strategic Milestones
6. Failures, Bottlenecks & Limitations
7. Primary-Source Evidence & Recent Developments
8. A direct comparison against the closest named competitor/alternative on shared metrics

IF Person / Sports / History / Politics / Event:
1. Origins & Early Milestones (verifiable dates, named events)
2. Career/Life Statistical Record (exact figures - no vague category summaries)
3. Leadership/Decision-Making Record (if applicable)
4. Recognition & Institutional Record (awards, rankings, official honors, with dates)
5. Commercial/Financial Dimension (contracts, earnings, valuations - dates and figures)
6. Timeline of Key Turning Points
7. Named controversies, criticism, failures, or disputed decisions
8. A direct comparison against a named peer/rival on shared metrics

Both tracks end in an adversarial angle (7) and a comparative angle (8) for a reason: a report
with no adversarial material in its evidence cannot honestly write a critical section later, and
a report with no comparison evidence cannot honestly write a comparison section later - both
end up manufactured from scraps instead of built from real sources found up front.

==================================================
PART 3 — GENERATE HIGH-PRECISION SEARCH QUERIES
==================================================

Generate 10-15 high-precision, non-redundant queries. Each should map to a distinct angle above -
if two queries would return the same fact area, merge or drop one.

Requirements for Strong Queries:
- Target specific entities, mechanisms, or named events.
- Include measurable quantities, dates, or named decisions where applicable.
- You have been given today's date above - use "current" or "latest" instead of a specific year,
  and never hardcode a year like "as of 2024" unless the topic itself is historical. Writing a
  stale year sends the crawler after outdated results even when current ones exist.
- Include at least one query explicitly targeting controversy, criticism, or failure, and at
  least one explicitly targeting a comparison against a named peer - phrase these directly
  ("[topic] criticism", "[topic] vs [rival] comparison") rather than hoping a general query
  surfaces them incidentally.

Strictly Avoid Vague Queries Such As:
- "What is X?"  /  "X career"  /  "X training data"  /  "How does X work?"

Query Optimization Examples (Technology):
- Poor: "ChatGPT system design"
  Better: "ChatGPT request lifecycle CDN edge API gateway prompt budget calculator SSE token streaming"

Query Optimization Examples (Person/Sports):
- Poor: "Rohit Sharma career"
  Better: "Rohit Sharma ODI double century record dates venues opponent runs scored"
- Poor: "Rohit Sharma achievements"
  Better: "Rohit Sharma captaincy criticism win percentage overseas scrutiny"
- Poor: "Rohit Sharma vs other players"
  Better: "Rohit Sharma vs Virat Kohli ODI captaincy win percentage comparison"

==================================================
PART 4 — QUALITY CHECK
==================================================

Before returning the plan, verify:
- Does every angle in Part 2 have at least one query, including adversarial and comparative?
- Are domains specific to the topic's actual field, not just its broad category?
- Is any query hardcoding a stale year unnecessarily?
- Do any two queries cover the same fact area?

Return ONLY the structured ResearchPlan fields.
"""
```

---

## 2. `strategist_node` — the code

### BEFORE

```python
def strategist_node(state: AgentState):
    print("\n[STRATEGIST] Building search vectors...")
    _t = _node_start("strategist")
    memory_context = query_memory(state["topic"])
    print(f"  Memory: {'found past research' if memory_context else 'starting fresh'}")

    memory_hint = f"\n\nPAST RESEARCH (avoid re-searching these):\n{memory_context}" if memory_context else ""

    plan = _structured_invoke(ResearchPlan, [
        SystemMessage(content=STRATEGIST_PROMPT),
        HumanMessage(content=f"Build research plan for: {state['topic']}."
                     f"Today's date is {datetime.now().strftime(' %B %d, %Y')}."
                     f"{memory_hint}")
    ], stage="strategist")

    print(f"  {len(plan.queries)} queries planned | category: {plan.topic_category} | domains: {len(plan.search_domains)}")
    _node_end("strategist", _t)
```

**Problem:** the date was already being injected here — but nothing in the prompt told the model to actually use it instead of defaulting to a hardcoded year. And nothing checked whether the plan the model returned actually covered the adversarial/comparative angles the prompt asked for — pure prompt trust, no verification.

---

### AFTER

```python
def _missing_angles(queries: List[str]) -> Tuple[bool, bool]:
    """Cheap keyword check on the plan the strategist actually returned.
    A prompt instruction to include these already failed once before (the
    duplicate-query bug slipped past "non-redundant queries" as plain text),
    so this doesn't trust the prompt alone - it verifies, and if either angle
    is missing, strategist_node appends a deterministic fallback query below
    rather than re-asking the LLM. Bare "vs" is deliberately NOT used as the
    comparative signal - "ODI debut date vs which opponent" matched it during
    testing and isn't a comparison at all. Requiring the word "compar" itself
    trades a few missed legitimate comparisons for zero false confirmations -
    the safe side, since a missed one just costs one harmless extra query,
    while a false confirmation would let the guarantee silently do nothing.
    """
    text = " ".join(queries).lower()
    has_adversarial = any(w in text for w in
        ("criticism", "controvers", "failure", "failed", "limitation", "scrutiny", "backlash", "scandal"))
    has_comparative = "compar" in text
    return not has_adversarial, not has_comparative


def strategist_node(state: AgentState):
    print("\n[STRATEGIST] Building search vectors...")
    _t = _node_start("strategist")
    memory_context = query_memory(state["topic"])
    print(f"  Memory: {'found past research' if memory_context else 'starting fresh'}")

    memory_hint = f"\n\nPAST RESEARCH (avoid re-searching these):\n{memory_context}" if memory_context else ""

    plan = _structured_invoke(ResearchPlan, [
        SystemMessage(content=STRATEGIST_PROMPT),
        HumanMessage(content=f"Build research plan for: {state['topic']}."
                     f"Today's date is {datetime.now().strftime(' %B %d, %Y')}."
                     f"{memory_hint}")
    ], stage="strategist")

    missing_adversarial, missing_comparative = _missing_angles(plan.queries)
    if missing_adversarial:
        plan.queries.append(f"{state['topic']} criticism controversy limitations failure")
        print(f"  [ANGLE GUARANTEE] No adversarial query found - added one")
    if missing_comparative:
        plan.queries.append(f"{state['topic']} comparison versus closest rival or alternative")
        print(f"  [ANGLE GUARANTEE] No comparative query found - added one")

    print(f"  {len(plan.queries)} queries planned | category: {plan.topic_category} | domains: {len(plan.search_domains)}")
    _node_end("strategist", _t)
```

---

## 3. What was actually verified (not just claimed)

- Ran on SS Rajamouli (Person/Sports-adjacent track) → real film-industry domains (`ssrajamouli.com`, `boxofficemojo.com`, `rottentomatoes.com`), zero stale years, adversarial + comparative present unprompted
- Ran on "The Architecture of Claude" (Technology track, first test of that branch) → same result, plus surfaced a genuinely new, unfixed problem: the strategist has no concept of "this fact was never going to be public" vs "this fact is just hard to find" (undisclosed model specs)
- `_missing_angles` fallback logic unit-tested in isolation against 3 cases (both angles missing, both present, only one missing) — all 3 correct
- **Not yet observed:** the fallback firing for real inside a live pipeline run — both real tests had the model get it right on its own, so the safety net has never actually had to catch anything yet

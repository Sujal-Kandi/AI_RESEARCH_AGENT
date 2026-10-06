# AI Research Agent

Most AI research tools are just a search + summarize wrapper. You ask a question, it returns a paragraph with citations that may or may not exist.

This is different. It's a full multi-agent pipeline that plans, crawls, writes, audits, rewrites, and fact-checks — automatically — before producing a publication-ready PDF report. Every number in the output is checked against the actual crawled source text before the report ships.

---

## The Problem It Solves

When you ask a single LLM to research a topic, you get:
- Training data regurgitation, not live web research
- Citations that point to real URLs but fake the specific fact
- No quality control — the first draft is the final draft
- Flat summaries with no argument, no critical angle, no comparison

This agent solves all four. It searches the live web, builds a numbered source index, writes each section from real crawled evidence, scores its own work with a measured rubric, rewrites the weak sections, and then verifies every number in the output against the text of the source it cited.

---

## What It Produces

A PDF report with:
- Cover page with key findings
- Table of contents
- Executive summary
- 4-8 research sections written as long-form investigative prose
- A contrarian/critical section built in by design
- A comparative analysis section built in by design
- A final synthesis
- Two-column verified sources list with URLs

Sample reports in this repo: Nokia, Kodak, Dot-Com Bubble, NVIDIA H100 vs A100, Tesla FSD v12.

---

## Pipeline

```
Topic input
    │
    ▼
STRATEGIST
Classifies topic. Plans 7-8 targeted search queries covering
different angles — origins, metrics, failures, comparisons.
Checks long-term memory for past research on related topics.
    │
    ▼
HUMAN-IN-THE-LOOP PAUSE
User reviews the planned queries, can edit or remove them,
then approves. This is the only manual step.
    │
    ▼
CRAWLER
Runs all queries in parallel via Tavily.
Deep-fetches full page content from the top results,
prioritizing official domains (SEC filings, .edu, .gov,
official league/company sites) over blogs.
Builds a numbered source index [1], [2], [3]...
Relevance-filters results — a Salman Khan search won't
pull Shah Rukh Khan articles just because both match "Khan".
    │
    ▼
ARCHITECT
Plans topic-specific section titles.
Allocates different source subsets to each section
so they don't all write from the same 3 URLs.
Writes all sections simultaneously in parallel LLM calls
— not one after another, all at once.
Writes header (title, key findings, exec summary) last,
from the finished sections, not from raw crawl data.
Runs a prose dedup pass to remove near-identical sentences
that appeared in multiple parallel sections.
Runs a stat reconciliation pass to normalize conflicting
numbers across sections to the most-cited value.
Detects and removes invented technical entity names
that carry a real citation but never appear in any source.
    │
    ▼
AUDIT
Measures each section against the crawled sources —
citation coverage, corroboration rate, figures per 100 words,
figures recycled from earlier sections.
Computes a 1-10 score from that measurement, not from
asking the LLM to guess. Prints the deduction behind
every lost point.
Asks the LLM only what measurement can't answer:
which sections have a reasoning problem and why.
    │
    ▼
TARGETED REWRITER
Rewrites only the flagged sections, in parallel,
against their specific evidence subsets.
Skipped entirely if no sections were flagged.
    │
    ▼
FACTCHECK
For every sentence with a number, date, or named figure:
checks that the figure appears in the text of the cited source,
AND that multiple figures from the same sentence appear within
400 characters of each other in that source.
This catches the case where a real Q3 revenue figure gets
attached to the wrong year — both strings exist in the document
but sit paragraphs apart, so they were never stated together.
Drops sentences that fail. Prints grounding score.
    │
    ▼
REFINE (conditional)
Only runs if the audit verdict was NEEDS_MORE_RESEARCH
AND grounding is below 75%.
A report that is already well-sourced exports immediately.
When it runs, it adds follow-up queries and loops back
through crawler → architect → factcheck.
    │
    ▼
PDF EXPORT
Cover page, TOC, executive summary, sections,
synthesis, two-column verified sources list.
    │
    ▼
RAG MEMORY
Chunks the report and saves to ChromaDB.
Next run on a related topic retrieves this
so the strategist avoids re-searching known ground.
```

---

## How It's Different From a Simple RAG Pipeline

A basic RAG pipeline does: retrieve docs → generate answer → return.

This pipeline adds layers that change the output quality:

**Source targeting** — queries are written to hunt primary sources, not commentary. An SEC filing beats a Medium article about the same filing. The crawler prioritizes official domains.

**Co-occurrence verification** — most fact-checkers just confirm a number appears somewhere on the cited page. This one also checks that the multiple numbers in one claim appear within 400 characters of each other. "Revenue grew 18% in Q3 2023" passes only if 18%, Q3, and 2023 all appear together in the source — not scattered across the page in different contexts.

**Measured audit** — the quality score is computed from counts (citation rate, corroboration rate, figure density, repetition rate) before the LLM ever sees the report. The LLM is then asked only what counting can't tell you. Previously the critic invented faults to fill a "name 2 weak sections" quota. Now it's free to flag nothing.

**Parallel section writing with dedup** — sections write simultaneously so the report takes 2 minutes instead of 7. Because parallel writers can't see each other, a dedup pass afterward removes near-identical sentences that appeared in multiple sections.

**Key rotation with real backoff** — tracks when each key cools, waits for the soonest available one, has a configurable total wait budget. A run with one key still completes instead of crashing on a 429.

---

## Latency

A full research run takes 2-4 minutes depending on topic complexity and API response times.

Breakdown:
- Strategist + crawler: 20-40 seconds (parallel queries)
- Architect: 60-90 seconds (parallel section writing)
- Audit + rewrite: 30-60 seconds
- Factcheck + export: 15-30 seconds

The pipeline is designed around parallelism — search queries run concurrently, sections write concurrently, rewrites run concurrently. Sequential execution would take 10-15 minutes. Parallel brings it to 2-4.

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
TAVILY_API_KEY=your_key
GROQ_API_KEY=your_key
GROQ_API_KEY_2=your_second_key
TOGETHER_API_KEY=your_key
```

Get free keys:
- Tavily: tavily.com
- Groq: console.groq.com
- Together.ai: api.together.ai

---

## Running It

Web UI:

```bash
python api.py
```

Open http://localhost:8000. Enter a topic, review the planned queries, approve, watch it run, download the PDF.

Command line:

```bash
python agent.py "The Rise and Fall of Nokia"
python agent.py "NVIDIA H100 vs A100 Architecture"
python agent.py "The Dot-Com Bubble"
```

---

## API

| Method | Endpoint | What it does |
|---|---|---|
| POST | /research/plan | Runs strategist, returns queries for review |
| POST | /research/start | Approves plan, starts pipeline |
| GET | /research/status/{id} | Poll progress |
| GET | /research/result/{id} | View PDF inline |
| GET | /research/download/{id} | Download PDF |

---

## Configuration

All tunable via environment variables:

```
SECTION_WORKERS=8          # sections written in parallel
DEEP_FETCH_LIMIT=5         # full page fetches per crawl round
SECTION_MIN=3              # minimum sections in a report
SECTION_MAX=9              # maximum sections in a report
CLAIM_WINDOW=400           # character window for co-occurrence check
REFINE_GROUNDING_FLOOR=0.75  # skip refine if grounding is above this
MAX_REFINE_ITERATIONS=2    # hard cap on refine loops
MAX_WEAK_SECTIONS=3        # max sections sent for rewrite per audit
KEY_COOLDOWN_SECONDS=60    # how long a rate-limited key is skipped
MAX_TOTAL_RATE_LIMIT_WAIT=300  # total wait budget per LLM call
```

---

## Project Structure

```
agent.py          all pipeline nodes, graph definition, PDF export
api.py            FastAPI server, REST endpoints, session management
ui.html           single-page web interface
rag.py            ChromaDB memory — save and retrieve past research
tools.py          web search and Arxiv tools
clear_memory.py   reset checkpoints and vector store
requirements.txt  dependencies
```

---

## Stack

Python, LangGraph, LangChain, FastAPI, Groq, Together.ai, Tavily, ChromaDB, fpdf2

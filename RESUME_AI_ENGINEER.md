# SUJAL KANDI
**AI Engineer — Agentic AI & GenAI**

kandisujal@gmail.com | 9346766534 | Hyderabad
GitHub: [github.com/Bryan-eng-lng](https://github.com/Bryan-eng-lng) | LinkedIn: [linkedin.com/in/sujal-kandi-914974372](https://www.linkedin.com/in/sujal-kandi-914974372/)

---

## SUMMARY

AI Engineer specialized in building production-grade agentic systems and multi-agent pipelines. Hands-on experience with LangGraph, RAG, LLM orchestration, and full-stack AI deployment. Built autonomous agents that plan, self-critique, and self-correct — all deployed live on the web.

---

## SKILLS

| Category | Skills |
|---|---|
| **Agentic AI & LLM Orchestration** | LangGraph, LangChain, Multi-Agent Pipelines, RAG, ChromaDB, Prompt Engineering, API Key Rotation, Human-in-the-Loop |
| **Generative AI** | Groq, Together.ai, Cerebras, Ollama, Hugging Face |
| **NLP** | Transformers, BERT, RoBERTa, Sentiment Analysis, ABSA, Embeddings |
| **Tools & Frameworks** | Python, FastAPI, Flask, Streamlit, Pandas, NumPy, Git, Vercel, Render |

---

## PROJECTS

### AI Research Agent *(Agentic AI, Multi-Agent, End-to-End)*
**Tech:** LangGraph, LangChain, FastAPI, Groq, Together.ai, Tavily, ChromaDB, fpdf2
[Live App](https://research-agent-zxhy.onrender.com) | [Code](https://github.com/Bryan-eng-lng)

- Built a 7-node autonomous research pipeline: Strategist → Crawler → Architect → Section Challenger → Targeted Rewriter → Factcheck → Critic — each node is a specialized LLM agent
- Agent decides its own report structure, writes each section independently, identifies its 2 weakest sections, rewrites them, and scores its own quality — looping until publish-ready
- Implemented hallucination firewall: every source gets a numbered ID at crawl time; factcheck node verifies all citations against the real index
- Built automatic API key rotation across multiple Groq keys with Together.ai fallback — zero crashes under free-tier rate limits
- Deployed full-stack: FastAPI backend + vanilla JS UI with real-time progress tracking and PDF download

---

### Blog Writer Agent *(Agentic AI, Multi-Agent, End-to-End)*
**Tech:** LangChain, FastAPI, Groq, Cerebras, Tavily, ChromaDB, Vercel, Render
[Live App](https://blog-agent-rho.vercel.app) | [Code](https://github.com/Bryan-eng-lng/Blog_Agent)

- Built a 9-step autonomous blog writing pipeline: Plan → Research → Fact Extraction → Write → Critique & Rewrite → Cliché Detection → SEO Metadata → Extras → Quality Score
- Implemented temperature strategy per node (0.0 for scorer, 0.6 for writer) — different LLM behavior for different tasks
- Cliché detector scans 30+ known phrases and forces topic-specific replacements; quality scored across 5 dimensions
- Deployed split architecture: FastAPI on Render, frontend on Vercel

---

### Smart Chatbot *(RAG, Agentic Routing, End-to-End)*
**Tech:** LangChain, ChromaDB, Ollama (Qwen2.5:7b), FastAPI, Streamlit, DuckDuckGo
[Code](https://github.com/Bryan-eng-lng/Smart-Chatbot)

- Built an intelligent document chatbot with an LLM-powered router that automatically decides whether to answer from uploaded documents (RAG) or live web search — no manual switching
- Supports PDF, DOCX, TXT, CSV, Markdown uploads; chunks and indexes into ChromaDB vector store
- Persistent multi-session chat history (like ChatGPT) with streaming responses; fully local — no data leaves the machine

---

## EDUCATION

**IIT Patna + Masai School** — Certified Program, Full-Stack ML & GenAI *(2025–2026)*

Class 12th — 93.7% | Class 10th — 87%

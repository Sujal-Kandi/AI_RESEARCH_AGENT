import io
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from auth import create_access_token, create_user, get_current_user, get_user_by_email, verify_password
from schemas import Token, UserData, UserLogin, UserRegister

# ── App & Router ───────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load secrets from Render secret file first, then fall back to local .env
    load_dotenv(dotenv_path="/etc/secrets/.env", override=False)
    load_dotenv(dotenv_path=".env", override=False)

    keys = {
        "TAVILY_API_KEY":   bool(os.getenv("TAVILY_API_KEY")),
        "GROQ_API_KEY":     bool(os.getenv("GROQ_API_KEY")),
        "GROQ_API_KEY_2":   bool(os.getenv("GROQ_API_KEY_2")),
        "TOGETHER_API_KEY": bool(os.getenv("TOGETHER_API_KEY")),
    }
    print(f"[STARTUP] Environment keys loaded: {keys}")
    yield

app = FastAPI(title="AI Research Agent", lifespan=lifespan)
router = APIRouter(prefix="/api")


# ── AUTH ENDPOINTS ─────────────────────────────────────────────────────────────

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(body: UserRegister):
    """Create a new account and return a JWT so the user is immediately logged in."""
    user = create_user(
        username=body.username,
        email=body.email,
        plain_password=body.password,
    )
    token = create_access_token(data={
        "sub": user["username"],
        "email": user["email"],
        "tenant_id": user["tenant_id"],
    })
    return Token(access_token=token, token_type="bearer")


@router.post("/login", response_model=Token)
async def login(body: UserLogin):
    """Authenticate with email + password and return a JWT."""
    user = get_user_by_email(body.email)
    if not user or not verify_password(body.password, user["hashed_pw"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(data={
        "sub": user["username"],
        "email": user["email"],
        "tenant_id": user["tenant_id"],
    })
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserData)
async def get_me(current_user: UserData = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


# ── Include auth router ────────────────────────────────────────────────────────
app.include_router(router)


# ── Startup check ─────────────────────────────────────────────────────────────



# ── SESSION STORE ──────────────────────────────────────────────────────────────
sessions: dict = {}


# ── REQUEST MODELS ─────────────────────────────────────────────────────────────
class PlanRequest(BaseModel):
    topic: str

class StartRequest(BaseModel):
    session_id: str
    queries: list = []


# ── STEP 1: PLAN ──────────────────────────────────────────────────────────────
@app.post("/research/plan")
def plan_research(
    req: PlanRequest,
    current_user: UserData = Depends(get_current_user),
):
    """Run strategist node, return queries for user approval."""
    import traceback
    try:
        from agent import AgentState, strategist_node
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[IMPORT ERROR] agent.py failed to import:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Agent import failed: {str(e)}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "status": "planning",
        "topic": req.topic,
        "queries": [],
        "reasoning": "",
        "current_node": "strategist",
        "progress": 5,
        "pdf_bytes": None,
        "pdf_filename": None,
        "error": None,
        "owner": current_user.username,
    }

    try:
        state = strategist_node({"topic": req.topic})
        plan = state["plan"]
        sessions[session_id].update({
            "status": "awaiting_approval",
            "queries": plan.queries,
            "reasoning": plan.reasoning,
            "current_node": "commander_review",
            "progress": 10,
            "_state": {
                "topic": req.topic,
                "plan": plan,
                "iteration": state["iteration"],
                "research_rounds": state["research_rounds"],
                "source_index": state["source_index"],
                "source_titles": state["source_titles"],
                "memory_context": state["memory_context"],
            },
        })
    except Exception as e:
        sessions[session_id].update({"status": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "session_id": session_id,
        "topic": req.topic,
        "queries": plan.queries,
        "reasoning": plan.reasoning,
    }


# ── STEP 2: START ─────────────────────────────────────────────────────────────
@app.post("/research/start")
def start_research(
    req: StartRequest,
    current_user: UserData = Depends(get_current_user),
):
    """User approved the plan — kick off the full pipeline in a background thread."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Session is in state: {session['status']}")

    session["status"] = "running"
    if req.queries:
        session["_state"]["plan"].queries = req.queries

    thread = threading.Thread(target=_run_pipeline, args=(req.session_id,), daemon=True)
    thread.start()

    return {"session_id": req.session_id, "status": "running"}


def _run_pipeline(session_id: str):
    """Background thread: runs the full multi-agent pipeline."""
    session = sessions[session_id]
    state = session["_state"].copy()

    def update(node: str, progress: int):
        session["current_node"] = node
        session["progress"] = progress

    try:
        from agent import (
            architect_node,
            crawler_node,
            critic_node,
            export_to_pdf,
            factcheck_node,
            refine_node,
            section_challenger_node,
            should_refine,
            targeted_rewrite_node,
        )

        update("crawler", 15)
        state.update(crawler_node(state))

        update("architect", 35)
        state.update(architect_node(state))

        update("section_challenger", 60)
        state["challenge_notes"] = ""
        state.update(section_challenger_node(state))

        update("targeted_rewrite", 70)
        result = targeted_rewrite_node(state)
        if result:
            state.update(result)

        update("factcheck", 80)
        factcheck_node(state)

        update("critic", 88)
        state.update(critic_node(state))

        # One refine loop if needed
        if should_refine(state) == "refine":
            update("refine", 90)
            state.update(refine_node(state))
            update("crawler_2", 92)
            state.update(crawler_node(state))
            update("architect_2", 94)
            state.update(architect_node(state))
            update("factcheck_2", 96)
            factcheck_node(state)
            update("critic_2", 97)
            state.update(critic_node(state))

        update("exporting", 98)
        pdf_path = export_to_pdf(
            state["raw_report"],
            state["source_index"],
            state["source_titles"],
            state["topic"],
        )

        pdf_bytes = Path(pdf_path).read_bytes()
        try:
            Path(pdf_path).unlink()
        except Exception:
            pass

        session.update({
            "status": "done",
            "current_node": "done",
            "progress": 100,
            "pdf_bytes": pdf_bytes,
            "pdf_filename": Path(pdf_path).name,
        })

    except Exception as e:
        session.update({
            "status": "error",
            "error": str(e),
            "current_node": "error",
        })
        print(f"[PIPELINE ERROR] {e}")


# ── STATUS ─────────────────────────────────────────────────────────────────────
@app.get("/research/status/{session_id}")
def get_status(
    session_id: str,
    current_user: UserData = Depends(get_current_user),
):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "status": session["status"],
        "current_node": session["current_node"],
        "progress": session["progress"],
        "error": session.get("error"),
    }


# ── RESULT: PDF ────────────────────────────────────────────────────────────────
@app.get("/research/result/{session_id}")
def get_result(
    session_id: str,
    current_user: UserData = Depends(get_current_user),
):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {session['status']}")
    pdf_bytes = session.get("pdf_bytes")
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not found")
    filename = session.get("pdf_filename", "report.pdf")
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{filename}\""},
    )


@app.get("/research/download/{session_id}")
def download_result(
    session_id: str,
    current_user: UserData = Depends(get_current_user),
):
    """Force-download the PDF."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Not ready. Status: {session['status']}")
    pdf_bytes = session.get("pdf_bytes")
    if not pdf_bytes:
        raise HTTPException(status_code=404, detail="PDF not found")
    filename = session.get("pdf_filename", "report.pdf")
    if not filename.endswith(".pdf"):
        filename += ".pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )


# ── DEBUG ──────────────────────────────────────────────────────────────────────
@app.get("/debug/env")
def debug_env():
    return {
        "TAVILY_API_KEY":   (os.getenv("TAVILY_API_KEY", "")[:8] + "...") if os.getenv("TAVILY_API_KEY") else "MISSING",
        "GROQ_API_KEY":     (os.getenv("GROQ_API_KEY", "")[:8] + "...") if os.getenv("GROQ_API_KEY") else "MISSING",
        "GROQ_API_KEY_2":   (os.getenv("GROQ_API_KEY_2", "")[:8] + "...") if os.getenv("GROQ_API_KEY_2") else "MISSING",
        "TOGETHER_API_KEY": (os.getenv("TOGETHER_API_KEY", "")[:8] + "...") if os.getenv("TOGETHER_API_KEY") else "MISSING",
        "secret_file_exists": os.path.exists("/etc/secrets/.env"),
    }


# ── UI ─────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

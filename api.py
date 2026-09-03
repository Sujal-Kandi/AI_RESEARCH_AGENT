import io
import os
import re
import threading
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

from auth import create_access_token, create_user, get_current_user, get_user_by_email, verify_password
from schemas import Token, UserData, UserLogin, UserRegister
from database import get_conn, init_db

# rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv(dotenv_path="/etc/secrets/.env", override=False)
    load_dotenv(dotenv_path=".env", override=False)

    # init PostgreSQL tables
    try:
        init_db()
    except Exception as e:
        print(f"[DB INIT ERROR] {e}")

    keys = {
        "TAVILY_API_KEY":   bool(os.getenv("TAVILY_API_KEY")),
        "GROQ_API_KEY":     bool(os.getenv("GROQ_API_KEY")),
        "GROQ_API_KEY_2":   bool(os.getenv("GROQ_API_KEY_2")),
        "TOGETHER_API_KEY": bool(os.getenv("TOGETHER_API_KEY")),
        "DATABASE_URL":     bool(os.getenv("DATABASE_URL")),
    }
    print(f"[STARTUP] Keys loaded: {keys}")
    yield

app = FastAPI(title="AI Research Agent", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

router = APIRouter(prefix="/api")


# auth endpoints

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def register(request: Request, body: UserRegister):
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
@limiter.limit("10/minute")
async def login(request: Request, body: UserLogin):
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
    return current_user


@router.get("/history")
async def get_history(current_user: UserData = Depends(get_current_user)):
    """Return list of past reports for the logged-in user."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, topic, filename, created_at
                    FROM reports
                    WHERE user_id = (SELECT id FROM users WHERE LOWER(username) = LOWER(%s))
                    ORDER BY created_at DESC
                    LIMIT 50
                    """,
                    (current_user.username,)
                )
                rows = cur.fetchall()
        return {"reports": [dict(r) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{report_id}/download")
async def download_history_report(
    report_id: str,
    current_user: UserData = Depends(get_current_user),
):
    """Download a past report PDF by its ID."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.pdf_bytes, r.filename
                    FROM reports r
                    JOIN users u ON r.user_id = u.id
                    WHERE r.id = %s AND LOWER(u.username) = LOWER(%s)
                    """,
                    (report_id, current_user.username)
                )
                row = cur.fetchone()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not row:
        raise HTTPException(status_code=404, detail="Report not found")

    return StreamingResponse(
        io.BytesIO(bytes(row["pdf_bytes"])),
        media_type="application/pdf",
        headers=pdf_headers(row["filename"]),
    )


def pdf_headers(filename: str, disposition: str = "attachment") -> dict:
    """Content-Disposition that survives non-ASCII filenames (RFC 5987)."""
    ascii_name = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode()
    ascii_name = re.sub(r'[^A-Za-z0-9._-]', "_", ascii_name) or "report.pdf"
    quoted = quote(filename)
    return {
        "Content-Disposition": (
            f"{disposition}; filename=\"{ascii_name}\"; filename*=UTF-8''{quoted}"
        )
    }


app.include_router(router)

# in-memory session store (pipeline state lives here during processing)
sessions: dict = {}


class PlanRequest(BaseModel):
    topic: str

class StartRequest(BaseModel):
    session_id: str
    queries: list = []


# step 1: plan
@app.post("/research/plan")
@limiter.limit("20/minute")
def plan_research(
    request: Request,
    req: PlanRequest,
    current_user: UserData = Depends(get_current_user),
):
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
        "detail": "Planning search queries",
        "step_done": None,
        "step_total": None,
        "started_at": None,
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
                "source_texts": state["source_texts"],
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


# step 2: start
@app.post("/research/start")
def start_research(
    req: StartRequest,
    current_user: UserData = Depends(get_current_user),
):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Session is in state: {session['status']}")

    session["status"] = "running"
    session["started_at"] = time.time()
    if req.queries:
        session["_state"]["plan"].queries = req.queries

    thread = threading.Thread(
        target=_run_pipeline,
        args=(req.session_id, current_user.username),
        daemon=True
    )
    thread.start()

    return {"session_id": req.session_id, "status": "running"}


def _save_report_to_db(username: str, topic: str, pdf_bytes: bytes, filename: str):
    """Save completed report to PostgreSQL for history."""
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM users WHERE LOWER(username) = LOWER(%s)", (username,))
                user_row = cur.fetchone()
                if not user_row:
                    print(f"[DB] User {username} not found, skipping report save")
                    return
                cur.execute(
                    """
                    INSERT INTO reports (id, user_id, topic, filename, pdf_bytes)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), user_row["id"], topic, filename, psycopg2_bytes(pdf_bytes))
                )
            conn.commit()
        print(f"[DB] Report saved for {username}: {filename}")
    except Exception as e:
        print(f"[DB] Failed to save report: {e}")


def psycopg2_bytes(data: bytes):
    """Wrap bytes for psycopg2 binary insert."""
    from psycopg2 import Binary
    return Binary(data)


# Each stage owns a slice of the bar. Progress inside a slice comes from the
# pipeline's own events (queries searched, sections written), so the number
# moves because real work finished, not because time passed.
STAGE_SPANS = {
    "crawler":          (12, 32),
    "architect":        (32, 70),
    "audit":            (70, 76),
    "targeted_rewrite": (76, 88),
    "factcheck":        (88, 93),
    "refine":           (93, 95),
    "exporting":        (95, 99),
}


def _run_pipeline(session_id: str, username: str):
    session = sessions[session_id]
    state = session["_state"].copy()

    def update(node: str, progress: int, detail: str = ""):
        session["current_node"] = node
        session["progress"] = progress
        session["detail"] = detail
        session["step_done"] = None
        session["step_total"] = None

    def on_progress(stage: str, detail: str, done, total):
        """Turn a pipeline event into a bar position inside that stage's slice."""
        low, high = STAGE_SPANS.get(stage, (session["progress"], session["progress"]))
        fraction = (done / total) if (done is not None and total) else 0
        session["current_node"] = stage
        session["detail"] = detail
        session["step_done"] = done
        session["step_total"] = total
        # Never let the bar walk backwards between stages.
        session["progress"] = max(session["progress"], int(low + (high - low) * fraction))

    state["_progress"] = on_progress

    try:
        from agent import (
            architect_node,
            audit_node,
            crawler_node,
            export_to_pdf,
            factcheck_node,
            refine_node,
            should_refine,
            targeted_rewrite_node,
        )

        update("crawler", 12, "Starting web research")
        state.update(crawler_node(state))

        update("architect", 32, "Drafting the report")
        state.update(architect_node(state))

        update("audit", 70, "Auditing the draft")
        state.update(audit_node(state))

        update("targeted_rewrite", 76, "Improving weak sections")
        result = targeted_rewrite_node(state)
        if result:
            state.update(result)

        update("factcheck", 88, "Verifying claims against sources")
        state.update(factcheck_node(state))

        if should_refine(state) == "refine":
            update("refine", 93, "Report too thin, researching further")
            state.update(refine_node(state))
            state.update(crawler_node(state))
            state.update(architect_node(state))
            state.update(factcheck_node(state))

        update("exporting", 95, "Building the PDF")
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

        pdf_filename = Path(pdf_path).name

        # print token summary to Render logs
        from agent import print_token_summary
        print_token_summary()

        # save to PostgreSQL history
        _save_report_to_db(username, state["topic"], pdf_bytes, pdf_filename)

        session.update({
            "status": "done",
            "current_node": "done",
            "progress": 100,
            "detail": f"{len(state.get('source_index') or {})} sources cited",
            "step_done": None,
            "step_total": None,
            # Audit and factcheck finish faster than the UI polls, so their
            # numbers are carried to the result card instead of only flashing.
            "stats": {
                "sources": len(state.get("source_index") or {}),
                "grounding": state.get("grounding"),
                "score": getattr(state.get("quality"), "score", None),
            },
            "pdf_bytes": pdf_bytes,
            "pdf_filename": pdf_filename,
        })

    except Exception as e:
        session.update({
            "status": "error",
            "error": str(e),
            "current_node": "error",
        })
        print(f"[PIPELINE ERROR] {e}")


# status
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
        "detail": session.get("detail", ""),
        "step_done": session.get("step_done"),
        "step_total": session.get("step_total"),
        "started_at": session.get("started_at"),
        "stats": session.get("stats"),
        "error": session.get("error"),
    }


# view PDF inline
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
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers=pdf_headers(filename, "inline"),
    )


# force download
@app.get("/research/download/{session_id}")
def download_result(
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
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers=pdf_headers(filename),
    )


# debug
@app.get("/debug/env")
def debug_env():
    return {
        "TAVILY_API_KEY":   "set" if os.getenv("TAVILY_API_KEY") else "MISSING",
        "GROQ_API_KEY":     "set" if os.getenv("GROQ_API_KEY") else "MISSING",
        "TOGETHER_API_KEY": "set" if os.getenv("TOGETHER_API_KEY") else "MISSING",
        "DATABASE_URL":     "set" if os.getenv("DATABASE_URL") else "MISSING",
    }


# serve UI
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

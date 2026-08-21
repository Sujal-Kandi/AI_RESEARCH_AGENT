import os
import uuid
import threading
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn

from fastapi import APIRouter, Depends ,  status
from fastapi.security import OAuth2PasswordRequestForm
from auth import FAKE_USERS_DB, verify_password, create_access_token, get_current_user
from schemas import Token, UserData


app = FastAPI(title="AI Research Agent")


router = APIRouter(prefix="/api")

@router.post("/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = FAKE_USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password , user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    token = create_access_token(data={"sub" : user["username"] , "tenant_id": user["tenant_id"]})
    {"access_token" : token , "token_type": "bearer"}

@router.post("/research")
async def execute_research(
    query: str,
    current_user: UserData = Depends(get_current_user)
):


    return {
        "status": "success",
        "user": current_user.username,
        "query": query,
        "result": f"Executed research query for tenant: {current_user.tenant_id}"
    }




# app = FastAPI(title="AI Research Agent")



@app.on_event("startup")
async def startup_check():
    import os
    keys = {
        "TAVILY_API_KEY": bool(os.getenv("TAVILY_API_KEY")),
        "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY")),
        "GROQ_API_KEY_2": bool(os.getenv("GROQ_API_KEY_2")),
        "TOGETHER_API_KEY": bool(os.getenv("TOGETHER_API_KEY")),
    }
    print(f"[STARTUP] Environment keys loaded: {keys}")

# ── SESSION STORE ─────────────────────────────────────────────────────────────
sessions = {}
# session = {
#   "status": "planning" | "awaiting_approval" | "running" | "done" | "error",
#   "topic": str,
#   "queries": [...],
#   "reasoning": str,
#   "current_node": str,
#   "progress": int (0-100),
#   "pdf_bytes": bytes | None,
#   "pdf_filename": str | None,
#   "error": str | None,
# }

# ── REQUEST MODELS ─────────────────────────────────────────────────────────────
class PlanRequest(BaseModel):
    topic: str

class StartRequest(BaseModel):
    session_id: str
    queries: list = []  # optional override — user can remove queries before starting

# ── STEP 1: PLAN 
@app.post("/research/plan")
def plan_research(req: PlanRequest):
    """Run strategist node, return queries for user approval."""
    from agent import strategist_node, AgentState

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
            # carry forward state fields needed for run
            "_state": {
                "topic": req.topic,
                "plan": plan,
                "iteration": state["iteration"],
                "research_rounds": state["research_rounds"],
                "source_index": state["source_index"],
                "source_titles": state["source_titles"],
                "memory_context": state["memory_context"],
            }
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

# ── STEP 2: START 
@app.post("/research/start")
def start_research(req: StartRequest):
    """User approved the plan — kick off the full pipeline in background."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail=f"Session is in state: {session['status']}")

    session["status"] = "running"
    # Override queries if user removed some
    if req.queries:
        session["_state"]["plan"].queries = req.queries
    thread = threading.Thread(target=_run_pipeline, args=(req.session_id,), daemon=True)
    thread.start()

    return {"session_id": req.session_id, "status": "running"}

def _run_pipeline(session_id: str):
    """Background thread: runs crawler → architect → challenger → rewriter → factcheck → critic → export."""
    session = sessions[session_id]
    state = session["_state"].copy()

    def update(node: str, progress: int):
        session["current_node"] = node
        session["progress"] = progress

    try:
        from agent import (
            crawler_node, architect_node, section_challenger_node,
            targeted_rewrite_node, factcheck_node, critic_node,
            refine_node, should_refine, export_to_pdf
        )

        update("crawler", 15)
        state.update(crawler_node(state))

        update("architect", 35)
        state.update(architect_node(state))

        update("section_challenger", 60)
        state["challenge_notes"] = ""
        result = section_challenger_node(state)
        state.update(result)

        update("targeted_rewrite", 70)
        result = targeted_rewrite_node(state)
        if result:
            state.update(result)

        update("factcheck", 80)
        factcheck_node(state)

        update("critic", 88)
        result = critic_node(state)
        state.update(result)

        # One refine loop if needed
        if should_refine(state) == "refine":
            update("refine", 90)
            refine_result = refine_node(state)
            state.update(refine_result)
            update("crawler_2", 92)
            state.update(crawler_node(state))
            update("architect_2", 94)
            state.update(architect_node(state))
            update("factcheck_2", 96)
            factcheck_node(state)
            update("critic_2", 97)
            result = critic_node(state)
            state.update(result)

        update("exporting", 98)
        pdf_path = export_to_pdf(
            state["raw_report"],
            state["source_index"],
            state["source_titles"],
            state["topic"]
        )

        # Read PDF into memory then delete local file
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

# ── STATUS ────────────────────────────────────────────────────────────────────
@app.get("/research/status/{session_id}")
def get_status(session_id: str):
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

# ── RESULT: PDF ───────────────────────────────────────────────────────────────
@app.get("/research/result/{session_id}")
def get_result(session_id: str):
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
        filename = filename + ".pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename=\"{filename}\""}
    )

@app.get("/research/download/{session_id}")
def download_result(session_id: str):
    """Force download the PDF."""
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
        filename = filename + ".pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
    )

@app.get("/debug/env")
def debug_env():
    """Debug endpoint to check env vars are loaded."""
    import os
    return {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")[:8] + "..." if os.getenv("TAVILY_API_KEY") else "MISSING",
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "")[:8] + "..." if os.getenv("GROQ_API_KEY") else "MISSING",
        "GROQ_API_KEY_2": os.getenv("GROQ_API_KEY_2", "")[:8] + "..." if os.getenv("GROQ_API_KEY_2") else "MISSING",
        "TOGETHER_API_KEY": os.getenv("TOGETHER_API_KEY", "")[:8] + "..." if os.getenv("TOGETHER_API_KEY") else "MISSING",
        "secret_file_exists": os.path.exists("/etc/secrets/.env"),
    }

# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    with open("ui.html", "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)



from fastapi import FastAPI
from pydantic import BaseModel
from app.agent import AgentReply, run_agent, search_jobs_tool
from app.db import init_db

app = FastAPI(title="CareerScout API")

class ChatRequest(BaseModel):
    session_id: str = "elip-prep"
    message: str

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/jobs/search")
def jobs_search(
    q: str = "AI agent backend intern",
    work_mode: str | None = None,
    min_monthly_inr: int | None = None,
    limit: int = 4,
):
    return {
        "results": search_jobs_tool(
            query=q,
            work_mode=work_mode,
            min_monthly_inr=min_monthly_inr,
            limit=limit,
        )
    }

@app.post("/chat", response_model=AgentReply)
def chat(req: ChatRequest):
    return run_agent(session_id=req.session_id, user_message=req.message)

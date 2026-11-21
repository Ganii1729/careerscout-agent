import hashlib
import json
import math
import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from app.db import get_conn

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

MODEL_NAME = "openai/gpt-oss-20b"


class JobRecommendation(BaseModel):
    title: str
    company: str
    location: str
    work_mode: str
    compensation: str
    fit_reason: str


class AgentReply(BaseModel):
    answer: str
    recommendations: list[JobRecommendation] = Field(default_factory=list)


def get_embedding(text: str, dim: int = 1536) -> list[float]:
    vec = [0.0] * dim
    tokens = [t.strip(".,()/").lower() for t in text.split() if t.strip(".,()/")]
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if (digest[4] % 2 == 0) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def search_jobs_tool(
    query: str,
    work_mode: Optional[str] = None,
    min_monthly_inr: Optional[int] = None,
    limit: int = 4,
) -> list[dict]:
    query_vec = get_embedding(query)
    sql = """
        SELECT
            id, title, company, location, work_mode, job_type,
            compensation, min_monthly_inr, description,
            1 - (embedding <=> %s::vector) AS similarity
        FROM jobs
        WHERE (%s::text IS NULL OR LOWER(work_mode) = LOWER(%s))
          AND (%s::int IS NULL OR COALESCE(min_monthly_inr, 0) >= %s)
        ORDER BY embedding <=> %s::vector ASC
        LIMIT %s
    """
    params = (
        query_vec,
        work_mode,
        work_mode,
        min_monthly_inr,
        min_monthly_inr,
        query_vec,
        limit,
    )
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    results = []
    for r in rows:
        results.append(
            {
                "id": r[0],
                "title": r[1],
                "company": r[2],
                "location": r[3],
                "work_mode": r[4],
                "job_type": r[5],
                "compensation": r[6],
                "min_monthly_inr": r[7],
                "description": r[8],
                "similarity": round(float(r[9] or 0.0), 4),
            }
        )
    return results


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search jobs from PostgreSQL pgvector by semantic query, optional work_mode, and optional min_monthly_inr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Role, skills, or company search query"},
                    "work_mode": {
                        "type": "string",
                        "enum": ["Remote", "Hybrid", "On-site"],
                        "description": "Optional work mode filter",
                    },
                    "min_monthly_inr": {
                        "type": "integer",
                        "description": "Optional minimum monthly pay in INR",
                    },
                    "limit": {"type": "integer", "description": "Max jobs to return (default 4)"},
                },
                "required": ["query"],
            },
        },
    }
]


def _load_history(session_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT role, content, tool_calls_json, tool_call_id
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages: list[dict] = []
    for role, content, tool_calls_json, tool_call_id in rows:
        msg: dict = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls_json:
            msg["tool_calls"] = json.loads(tool_calls_json)
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        messages.append(msg)
    return messages


def _save_message(
    session_id: str,
    role: str,
    content: Optional[str],
    tool_calls_json: Optional[str] = None,
    tool_call_id: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, tool_calls_json, tool_call_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (session_id, role, content, tool_calls_json, tool_call_id),
        )


def run_agent(session_id: str, user_message: str) -> AgentReply:
    system_prompt = (
        "You are CareerScout, a concise engineering career agent helping candidates target "
        "backend and AI/LLM roles like SDE Intern at Elip (YC W25). Always call search_jobs "
        "when looking up roles, then return valid JSON matching: "
        '{"answer": "...", "recommendations": [{"title": "...", "company": "...", '
        '"location": "...", "work_mode": "...", "compensation": "...", "fit_reason": "..."}]}'
    )

    _save_message(session_id, "user", user_message)
    messages = [{"role": "system", "content": system_prompt}] + _load_history(session_id)

    for _ in range(4):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []

        if tool_calls:
            serialized_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            _save_message(
                session_id,
                "assistant",
                msg.content,
                tool_calls_json=json.dumps(serialized_calls),
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": serialized_calls,
                }
            )

            for tc in tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if tc.function.name == "search_jobs":
                    data = search_jobs_tool(
                        query=args.get("query", user_message),
                        work_mode=args.get("work_mode"),
                        min_monthly_inr=args.get("min_monthly_inr"),
                        limit=int(args.get("limit", 4)),
                    )
                else:
                    data = {"error": f"Unknown tool: {tc.function.name}"}

                payload = json.dumps(data)
                _save_message(session_id, "tool", payload, tool_call_id=tc.id)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": payload,
                    }
                )
            continue

        raw_text = msg.content or "{}"
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        json_slice = cleaned[start : end + 1] if start != -1 and end != -1 else cleaned

        try:
            data_obj = json.loads(json_slice, strict=False)
            parsed = AgentReply.model_validate(data_obj)
        except Exception:
            parsed = AgentReply(answer=raw_text, recommendations=[])

        _save_message(session_id, "assistant", parsed.model_dump_json())
        return parsed
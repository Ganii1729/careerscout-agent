import hashlib
import math
from app.db import get_conn, init_db

SAMPLE_JOBS = [
    {
        "title": "SDE Intern - AI Career Agent",
        "company": "Elip (YC W25)",
        "location": "Gurugram / Bengaluru",
        "work_mode": "Remote",
        "job_type": "Internship",
        "compensation": "₹75,000/month",
        "min_monthly_inr": 75000,
        "description": "Build LLM-powered agents with tool calling, semantic search, embeddings, and Python backend APIs.",
    },
    {
        "title": "Backend Engineer Intern (Python / FastAPI)",
        "company": "FinPulse",
        "location": "Bengaluru",
        "work_mode": "Hybrid",
        "job_type": "Internship",
        "compensation": "₹55,000/month",
        "min_monthly_inr": 55000,
        "description": "Design REST APIs, PostgreSQL schemas, and asynchronous task workers for high-throughput payments.",
    },
    {
        "title": "Applied AI Intern - RAG & Search",
        "company": "DocuScale",
        "location": "Hyderabad",
        "work_mode": "Remote",
        "job_type": "Internship",
        "compensation": "₹60,000/month",
        "min_monthly_inr": 60000,
        "description": "Develop vector search, retrieval pipelines, reranking, and automated LLM evaluation suites.",
    },
    {
        "title": "Full Stack Software Engineer",
        "company": "ClearHire",
        "location": "Mumbai",
        "work_mode": "On-site",
        "job_type": "Full-time",
        "compensation": "₹18 LPA",
        "min_monthly_inr": 120000,
        "description": "Build React frontends and Node/Python microservices for candidate matching and workflow automation.",
    },
    {
        "title": "AI / ML Backend Intern",
        "company": "Vectorly",
        "location": "Bengaluru",
        "work_mode": "Remote",
        "job_type": "Internship",
        "compensation": "₹50,000/month",
        "min_monthly_inr": 50000,
        "description": "Work on embeddings, hybrid SQL and vector retrieval, prompt evaluation, and FastAPI services.",
    },
    {
        "title": "Software Engineer Intern - Search & Ranking",
        "company": "HireFlow",
        "location": "Gurugram",
        "work_mode": "On-site",
        "job_type": "Internship",
        "compensation": "₹65,000/month",
        "min_monthly_inr": 65000,
        "description": "Improve relevance ranking, query filtering, and low-latency APIs over structured job catalogs.",
    },
]


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


def seed():
    init_db()
    with get_conn() as conn:
        existing = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if existing > 0:
            print(f"Database already has {existing} jobs. Skipping duplicate seed.")
            return

        for job in SAMPLE_JOBS:
            searchable_text = (
                f"{job['title']} at {job['company']} ({job['work_mode']}, {job['location']}): "
                f"{job['description']}"
            )
            emb = get_embedding(searchable_text)
            conn.execute(
                """
                INSERT INTO jobs (
                    title, company, location, work_mode, job_type,
                    compensation, min_monthly_inr, description, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    job["title"],
                    job["company"],
                    job["location"],
                    job["work_mode"],
                    job["job_type"],
                    job["compensation"],
                    job["min_monthly_inr"],
                    job["description"],
                    emb,
                ),
            )
    print(f"Seeded {len(SAMPLE_JOBS)} jobs successfully.")


if __name__ == "__main__":
    seed()
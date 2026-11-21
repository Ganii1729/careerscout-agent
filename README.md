# CareerScout

FastAPI + PostgreSQL (`pgvector` on Neon) + Groq (`openai/gpt-oss-20b`) agentic job search and career prep API.

## Setup
1. Install dependencies: `pip install -r requirements.txt` (or `fastapi uvicorn openai psycopg[binary] pgvector pydantic python-dotenv`)
2. Configure `.env` with `DATABASE_URL` and `GROQ_API_KEY`.
3. Seed jobs: `python -m app.seed_jobs`
4. Run API: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

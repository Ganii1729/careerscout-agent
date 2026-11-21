# CareerScout

FastAPI + PostgreSQL (`pgvector` on Neon) + Groq (`openai/gpt-oss-20b`) agentic job search and career prep API.

## Live Demo
- **Swagger / OpenAPI Docs:** https://careerscout-agent.onrender.com/docs
- **Health Check:** https://careerscout-agent.onrender.com/health

*Note: Hosted on Render's free tier-the first request after inactivity may take up to ~60 seconds to wake up.*

## Local Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Configure `.env` with `DATABASE_URL` and `GROQ_API_KEY` (see `.env.example`).
3. Seed jobs: `python -m app.seed_jobs`
4. Run API: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

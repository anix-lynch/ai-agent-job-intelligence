# 🎯 AI Job Intelligence Platform - Project Specification

## Vision
Combine Resume MCP with Silicon Beach job data to create an AI-powered job matching platform featuring:
- **Vector-based semantic matching** using embeddings
- **AI agent orchestration** for autonomous job hunting
- **ML-powered predictions** for ATS success rates
- **Commute-aware filtering** for LA tech jobs
- **Auto resume tailoring** per job description

---

## Project Structure

```
ai-agent-job-intelligence/
├── 📁 data/
│   ├── resume.json                    # Tech resume (Resume MCP)
│   ├── b_past_life_resume.json        # Finance resume (Resume MCP)
│   ├── silicon_beach.duckdb           # LA job database (Silicon Beach)
│   ├── la_vcs_20251111.csv           # VC firms data
│   ├── builtinla_mcp_20251111.csv    # Job postings
│   ├── serpAPI_jobs.csv              # Scraped jobs (serpAPI repo)
│   ├── foorila_jobs.csv              # Foorila AI jobs
│   └── himalayas_jobs.csv            # Scraped jobs (himalayasAPI repo)
│
├── 📁 mcp/
│   ├── server_http.py                # Resume MCP server
│   ├── match_rank.py                 # Job matching logic
│   ├── rulebook.yaml                 # Filtering rules
│   └── openapi_chatgpt.yaml          # OpenAI integration spec
│
├── 📁 ml/
│   ├── vector_store.py               # ChromaDB/Qdrant embeddings
│   ├── semantic_matcher.py           # Cosine similarity matching
│   ├── classifier.py                 # ML job predictor
│   └── skill_gap_analyzer.py         # Missing skill detection
│
├── 📁 agents/
│   ├── langchain_agent.py            # LangChain orchestration
│   ├── auto_tailor.py                # Resume customization
│   ├── ats_predictor.py              # Success rate prediction
│   └── network_optimizer.py          # Referral path finder
│
├── 📁 ui/
│   ├── app.py                        # Streamlit dashboard
│   ├── components/
│   │   ├── semantic_matcher.py       # Vector matching UI
│   │   ├── auto_tailor.py            # Resume editor
│   │   ├── ats_predictor.py          # Score display
│   │   ├── skill_gap.py              # Learning recommendations
│   │   ├── commute_filter.py         # Map + transit
│   │   └── network_viz.py            # Connection graph
│   └── styles.css                    # UI styling
│
├── 📁 api/
│   ├── endpoints.py                  # FastAPI routes
│   ├── auth_middleware.py            # API keys from Resume MCP
│   └── vercel.json                   # Deployment config
│
├── 📁 config/
│   ├── requirements.txt              # Python dependencies
│   └── .env.example                  # Template
│
└── 📄 README.md                      # Project overview
```

---

## Data Sources

### 1. Resume MCP (GitHub: anix-lynch/resume-mcp)
- **Location**: `https://github.com/anix-lynch/resume-mcp`
- **Files**:
  - `resume.json` - Tech resume with skills, projects, experience
  - `b_past_life_mcp/resume.json` - Finance/VC resume
  - `northstar_mcp/projects.json` - Portfolio projects
  - `match_rank.py` - Existing matching logic
  - `rulebook.yaml` - Job filtering rules
- **Already deployed**: Vercel with MCP protocol

### 2. Silicon Beach Jobs (GitHub: anix-lynch/silicon-beach-jobs-clean)
- **Location**: `https://github.com/anix-lynch/silicon-beach-jobs-clean`
- **Files**:
  - `data/silicon_beach.duckdb` - DuckDB with LA tech jobs
  - `data/la_vcs_20251111_083756_enriched.csv` - VC firms
  - `data/builtinla_mcp_20251111_085045.csv` - Job postings
  - `app.py` - Streamlit map with commute analysis
- **Features**: Commute scoring, referral tracking, network paths

### 3. Job Scraping Repos (Local)
- **serpAPI**: `/Users/anixlynch/dev/serpAPI` - Google Jobs scraper
- **himalayasAPI**: `/Users/anixlynch/dev/himalayasAPI` - Remote jobs
- **Foorila_AIjob**: `/Users/anixlynch/dev/Foorila_AIjob` - AI/ML jobs
- **Format**: CSV files with job descriptions, keywords, salaries

### 4. Credentials (Local)
- **Location**: `~/.config/secrets/global.env`
- **Available APIs**:
  - OpenAI, Anthropic, Gemini (LLM)
  - Qdrant, Pinecone, Chroma (Vector DBs)
  - SerpAPI, Firecrawl, Browserbase (Scraping)
  - Vertex AI ($300 credit), Azure ($200 credit), AWS (free tier)
  - LangChain, LangSmith (Agent tools)

---

## UI Mockup

```
╔════════════════════════════════════════════════════════════════════════════════╗
║  🎯 AI JOB INTELLIGENCE PLATFORM - Resume MCP × Silicon Beach Integration      ║
╚════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│  👤 YOUR PROFILE                    📊 LIVE STATS                               │
│  ┌──────────────────┐               ┌────────────────────┐                     │
│  │ Tech Resume  ✅  │               │ Jobs Matched: 47   │                     │
│  │ Finance Resume ✅│               │ Applications: 12   │                     │
│  │ Vector Store  ✅ │               │ Success Rate: 94%  │                     │
│  └──────────────────┘               └────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ 🤖 AI AGENT FEATURES ──────────────────────────────────────────────────────────┐
│                                                                                  │
│  ┌─🎯 SEMANTIC MATCHER─┐  ┌─🔄 AUTO TAILOR─┐  ┌─📊 ATS PREDICTOR─┐            │
│  │  Vector Embeddings  │  │  Extract Keywords│  │  Pass Rate: 96%  │            │
│  │  Cosine Similarity  │  │  Rewrite Resume  │  │  Keyword Score: 8│            │
│  │  Qdrant/ChromaDB    │  │  LLM Optimize    │  │  Skill Match: 9  │            │
│  │  [⚡ Match Jobs]    │  │  [✨ Customize]  │  │  [📈 Analyze]    │            │
│  └────────────────────┘  └─────────────────┘  └──────────────────┘            │
│                                                                                  │
│  ┌─🧠 SKILL GAP ANALYZER─┐  ┌─📍 COMMUTE INTEL─┐  ┌─🔗 NETWORK OPT─┐         │
│  │  Missing: RAG, Agents │  │  🚇 Expo: 25min  │  │  Warm Intros: 3 │         │
│  │  Learn: ReAct, MCP    │  │  🚗 Drive: 35min │  │  LinkedIn: 12   │         │
│  │  Salary Impact: +$20K │  │  🟢 Excellent    │  │  Booth: 5       │         │
│  │  [📚 Recommend]       │  │  [🗺️ Filter]     │  │  [🤝 Connect]   │         │
│  └──────────────────────┘  └──────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ 💼 TOP MATCHES (LA Silicon Beach) ─────────────────────────────────────────────┐
│                                                                                  │
│  🔥 1. Snap Inc. - Santa Monica                              ⭐ Match: 98%      │
│  ├─ 💰 $180K-220K  │  📍 25min commute  │  🎯 ML Engineer                      │
│  ├─ ✅ RAG, LLMs, Vector DB, Python, Transformers                               │
│  ├─ 🔗 Connection: Elise Sha (Booth) → David Shi (Hiring Mgr)                  │
│  └─ [🚀 Auto Apply] [📝 Tailor Resume] [💬 Ask Referral]                       │
│                                                                                  │
│  🔥 2. Hulu - Santa Monica                                   ⭐ Match: 96%      │
│  ├─ 💰 $165K-200K  │  📍 22min commute  │  🎯 AI/ML Engineer                   │
│  ├─ ✅ ML Pipelines, Scikit-learn, Deep Learning, AWS                           │
│  ├─ 🔗 Connection: Via LinkedIn 2nd degree                                      │
│  └─ [🚀 Auto Apply] [📝 Tailor Resume] [🔍 Find Connector]                     │
│                                                                                  │
│  🔥 3. SpaceX - Hawthorne                                    ⭐ Match: 94%      │
│  ├─ 💰 $175K-210K  │  📍 32min commute  │  🎯 Data Engineer                    │
│  ├─ ✅ Python, ML, Data Pipelines, Distributed Systems                          │
│  ├─ ⚠️  Missing: Spark, Kafka (Learn: 2-3 weeks)                               │
│  └─ [📚 Skill Up] [📝 Tailor Resume] [⏰ Set Reminder]                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ 🎛️ ADVANCED FEATURES ──────────────────────────────────────────────────────────┐
│                                                                                  │
│  [🧪 Career Trajectory]  [💱 Compare Personas]  [🔄 Real-time Scrape]          │
│  Predict next role       Tech vs Finance         SerpAPI + Firecrawl            │
│                                                                                  │
│  [🤖 MCP Agent Access]   [📊 ML Training]        [🎯 Keyword Optimizer]         │
│  ChatGPT integration     Train on your data      ATS keyword scoring            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ ⚙️ TECH STACK ─────────────────────────────────────────────────────────────────┐
│  Vector: ChromaDB/Qdrant  │  ML: scikit-learn  │  LLM: OpenAI/Claude           │
│  Agent: LangChain         │  Deploy: Vercel    │  Data: DuckDB                 │
│  Cost: $0/month (free tier + credits)                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## ATS Keywords to Showcase

### 🤖 LLM-Related (Highest Pay)
1. Retrieval Augmented Generation (RAG) - $180K+
2. Prompt Engineering - $160K+
3. Transformer Models - $150K+
4. Large Language Models (LLMs) - $145K+
5. Vector Embeddings - $140K+

### 🤵 Agent-Related (Highest Pay)
1. AI Agent Orchestration - $200K+
2. Autonomous Reasoning - $185K+
3. Multi-Agent Systems - $170K+
4. Tool-Using Agents - $165K+
5. ReAct Frameworks - $160K+

### 🧠 ML-Related (Highest Pay)
1. Machine Learning Classification - $175K+
2. Deep Learning Architecture - $165K+
3. Neural Networks - $155K+
4. Vector Databases - $150K+
5. Predictive Analytics - $145K+

---

## Tech Stack

### Frontend
- **Streamlit** (existing in Silicon Beach repo)
- **Folium** for maps
- **Plotly** for charts

### Backend
- **FastAPI** (from Resume MCP `server_http.py`)
- **DuckDB** (from Silicon Beach for job data)
- **Pandas** for data processing

### ML/AI
- **sentence-transformers** (SBERT) for embeddings
- **ChromaDB** or **FAISS** for vector storage (open source)
- **scikit-learn** for ML classification
- **LangChain** for agent orchestration
- **OpenAI SDK** or **Anthropic** for LLM

### Deployment
- **Vercel** (free tier, already configured)
- **Streamlit Cloud** (free tier for dashboard)

---

## Cost Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| Vercel | $0 | Free tier |
| ChromaDB/FAISS | $0 | Open source, self-hosted |
| OpenAI API | $0-10 | Using existing credits |
| Google Vertex AI | $0 | $300 credit available |
| Azure | $0 | $200 credit available |
| AWS | $0 | Free tier |
| Streamlit Cloud | $0 | Free tier |
| **Total** | **$0-10/month** | |

---

## Repository Links

- **Resume MCP**: `https://github.com/anix-lynch/resume-mcp`
- **Silicon Beach Jobs**: `https://github.com/anix-lynch/silicon-beach-jobs-clean`
- **This Repo**: `https://github.com/anix-lynch/ai-agent-job-intelligence`

---

**Author**: Anix Lynch  
**Contact**: alynch@gozeroshot.dev  
**Last Updated**: 2025-11-18
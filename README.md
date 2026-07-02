# 🌍 JAGHUT v0.1L – ESG & Sustainability Intelligence Platform

An AI-powered ESG, climate, and sustainability intelligence platform designed for corporations, financial institutions, governments, NGOs, researchers, and the public in Indonesia & Southeast Asia.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?logo=ollama)](https://ollama.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)](https://python.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB%20Server-purple)](https://www.trychroma.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Insights-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
![License](https://img.shields.io/badge/License-MIT-green)

## 🎯 Core Capabilities

- **Forest IQ Integration** — Native calculation of Deforestation & Conversion Exposure (Metric 1), Financial Materiality (Metric 2), and Performance Reporting (Metric 3) with verified DCF volumes. Company-level ForestIQ scoring via `test.companies` / `test.assessments` with automated LLM-generated insights & recommendations.
- **ESG Screening & Evaluation** — Company-level ESG scoring, cross-sector benchmarking, and sustainability performance analysis.
- **Greenwashing Detection** — NLP-driven analysis of decoupling between corporate ESG claims and ground reality using OSINT and satellite data.
- **Supply Chain Traceability** — End-to-end mapping of forest-risk commodities (Palm Oil, Pulp & Paper, Beef, Cocoa, Timber).
- **Climate Risk Analysis** — Physical and transition climate risk assessment, carbon accounting, and decarbonization pathway modeling.
- **Regulatory Compliance** — EUDR, ISPO, RSPO, NDPE policy interpretation and compliance checking.
- **OSINT Environmental Intelligence** — Satellite imagery analysis, land conflict monitoring, and public data aggregation.

## 🔥 Technical Features

- **Hybrid RAG** → BM25 + Dense Vector (mxbai-embed-large) + Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
- **Multi-Platform** → CLI interface & **Telegram Bot Integration**
- **Dynamic Retriever** → Automatic weighting based on query type (entity/climate/history)
- **3-Layer Query Rewriting** → rule-based (0ms) → Qwen2.5 fallback (~800ms) → original query
- **Entity Detection** → Company names, regulations, frameworks, and commodity detection with ESG entity extraction
- **Real-time Data Integration**
  - Climate & weather data from Open-Meteo (temperature, rainfall, soil moisture)
  - Biodiversity & plant data from Perenual API
  - Forest IQ datasets via RAG (company metrics, financial data)
- **Long-term Memory** → Session summaries stored in ChromaDB for multi-turn context
- **Daily Insight Engine** → Sends ESG sector insights, climate analysis, and company intelligence to MongoDB every 12 hours
- **Company Insight Engine** → Generates per-company ForestIQ insights & recommendations via `qwen2.5:1.5b`, stored in separate `insight` & `Recomendation` columns (400-500 char each, sentence-boundary safe). Polls for changes every 300s, incremental by default
- **INI-based Configuration** → All ESG keywords & entity maps in `config/settings/`, no code changes needed
- **Scope Guard** → Only answers ESG, sustainability, and environmental topics

## Tech Stack

| Component | Detail |
|---|---|
| **Primary LLM** | Llama 3.2 personal-tuned → `jaghut-v0.1L` (via Ollama) |
| **Utility Model** | `qwen2.5:1.5b` — query rewriting, entity extraction, eval loop, insights |
| **Embedding** | `mxbai-embed-large` |
| **Vector Store** | ChromaDB Server — 6 collections: `main_dataset`, `weather_data`, `plant_data`, `entity_data`, `conversation_memory`, `company_insights` |
| **Retriever** | Ensemble (BM25 + Vector) + Cross-Encoder reranker (top_n=5) |
| **Insight DB** | MongoDB — 6 collections in `jaghut_insights` database: `esg_insights`, `climate_insights`, `sector_insights`, `policy_insights`, `session_summaries`, `company_insight` |
| **External APIs** | Open-Meteo (Climate), Perenual (Biodiversity) |
| **Framework** | LangChain, LangChain-Classic |

## Installation

### 1. Prerequisites

- Python 3.10+
- Ollama installed with the following models:
  ```bash
  ollama pull llama3.2
  ollama pull qwen2.5:1.5b
  ollama create jaghut-v0.1L -f config/Modelfile
  ollama pull mxbai-embed-large
  ```

### 2. Setup

```bash
git clone <your-repo-url>
cd JagaHutan

python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Configuration (.env)

Copy `config/.env.example` to `config/.env` and fill in:
- `TELEGRAM_BOT_TOKEN` — Your bot token from @BotFather
- `MONGO_URI` — MongoDB Atlas connection string (for insight storage)
- `PERENUAL_API_KEY` — Perenual API key (optional, for biodiversity data)

### 4. Database Server

```bash
# Terminal 1: Start ChromaDB
chroma run --path data/db --port 8000
```

### 5. Run

```bash
# Terminal 2: Start all services + Telegram bot
python start_all.py

# Or run CLI directly:
python interfaces/cli/main.py

# Manual company insight generation (incremental, one-shot):
python services/company_insight_engine.py --now

# Reset & regenerate all company insights:
python services/company_insight_engine.py --once
```

Inside CLI or Telegram bot, type `!companies` to trigger company insight generation on-demand.

### 6. Startup Robustness

- MongoDB connection retries 3x with exponential backoff (3s → 6s) on transient Atlas failures
- All company insight generation is wrapped in try-except — one company failure won't crash the pipeline
- `python services/company_insight_engine.py --now` — incremental one-shot (no reset)
- `python services/company_insight_engine.py --once` — reset & regenerate all

## Project Architecture

```
JagaHutan/
├── config/                        # Configuration
│   ├── .env                       # Environment variables
│   ├── .env.example               # Template
│   ├── Modelfile                  # Ollama model definition (Jaghut)
│   └── settings/                  # INI-based keyword configs
│       ├── scope_config.ini       # ESG domain scope guard
│       ├── rewriter_config.ini    # Query rewriting rules
│       └── esg_entities.ini       # Entity maps & detection keywords
├── core/                          # Engine Logic
│   ├── jaghut_core.py             # Main RAG pipeline & query processing
│   ├── esg_api.py                 # ESG entity API + Perenual (legacy)
│   ├── eval_loop.py               # Faithfulness & relevance scoring
│   ├── query_logger.py            # Query tracing & debug logs
│   └── user_store.py              # User profile management
├── services/                      # Background Services
│   ├── vectorCSV.py               # CSV/XLSX watcher & indexer
│   ├── vectorpdf.py               # PDF watcher & indexer
│   ├── vectorWeather.py           # Open-Meteo climate data crawler
│   ├── daily_insight.py           # MongoDB ESG insight generator (12h cron)
│   └── company_insight_engine.py  # Per-company ForestIQ insight generator
├── interfaces/                    # Entry Points
│   ├── cli/
│   │   └── main.py                # Terminal chat client
│   └── telegram/
│       ├── telegram_bot.py        # Telegram bot entry point
│       ├── requirements_telegram.txt
│       └── DEPLOYMENT_GUIDE.md
├── data/                          # Runtime Data (git-ignored)
│   ├── db/                        # ChromaDB persistent storage
│   ├── logs/                      # Query logs
│   ├── raw_dataset/               # ESG CSV/XLSX source files (Forest IQ, etc.)
│   ├── raw_pdfs/                  # ESG PDF documents
│   └── users.json                 # User profiles
├── tests/                         # Testing utilities
├── start_all.py                   # Master deployment script
├── README.md
└── requirements.txt
```

## License
MIT License.

Last updated: July 2026 · v0.1L · Jaghut ESG Platform

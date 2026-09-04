# RazorGuard AI: Explainable AI Risk Manager for Payment Fraud

> Track 02 — AI Risk Manager (Hackathon Project)

RazorGuard AI is a production-grade, defense-only explainable risk management platform for payment fraud detection. It combines machine learning fraud probabilities, deterministic compliance rule overrides, SHAP local feature attributions, and an isolated LLM natural language explanation engine.

--- 

## 🏗️ Monorepo Architecture

```
razorguard-ai/
├── frontend/             # React + Vite + Tailwind CSS + Recharts Dashboard
├── backend/              # FastAPI Python backend (Modular APIs, Database, Services)
│   ├── app/
│   │   ├── api/          # REST Endpoint Routers (v1)
│   │   ├── core/         # Settings, Environment & Security Config
│   │   ├── database/     # SQLAlchemy Database Sessions & Engine
│   │   ├── models/       # Database ORM Schemas
│   │   ├── risk_engine/  # Risk Engine Scoring & Policy Overrides
│   │   ├── schemas/      # Pydantic Data Validation Schemas
│   │   └── services/     # Business & Explainability Logic
│   └── tests/            # Backend API Pytest Suite
├── ml/                   # Machine Learning Pipeline Package
│   ├── preprocessing/    # Data Cleaners & Scalers
│   ├── features/         # Feature Engineering Pipelines
│   ├── training/         # LightGBM / XGBoost Model Trainers
│   ├── evaluation/       # Held-out Test Metrics (ROC-AUC, Precision, Recall, F1)
│   ├── inference/        # High-performance Predictor Wrappers
│   └── utils/            # Synthetic Payment Generator & Utilities
├── data/                 # Raw & Processed Datasets
├── models/               # Serialized ML Model Artifacts & Metrics JSON
├── notebooks/            # Exploratory Data Analysis & Prototyping
├── tests/                # System Integration Tests
├── docs/                 # Architecture & API Documentation
└── scripts/              # Dev & Deployment Helper Scripts
```

---

## 🚀 Quick Start Instructions

### Prerequisites
- **Python 3.10+**
- **Node.js v18+** & **npm**

### 1. Environment Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 2. Backend Setup & Local Server

Install Python dependencies:
```bash
pip install -r requirements.txt
```

Run FastAPI server with Uvicorn:
```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Verify Backend Health:
```bash
curl http://localhost:8000/api/v1/health
```

### 3. Frontend Setup & Dev Server

Navigate to `frontend/`:
```bash
cd frontend
npm install
npm run dev
```

Open the dashboard in your browser:
`http://localhost:5173`

---

## 🧪 Testing

Run backend & ML tests:
```bash
pytest
```

---

## 🛡️ Key Safety Principles

1. **Defense-Only Scope**: Designed exclusively to protect merchants from payment loss.
2. **Zero LLM Execution Authority**: ML & Rules deterministically evaluate `APPROVE`, `REVIEW`, or `BLOCK`. The LLM is strictly restricted to text summaries and analyst support.

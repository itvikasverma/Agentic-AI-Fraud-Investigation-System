# Agentic AI Fraud Investigation System 🏦🤖

An end-to-end, enterprise-grade Fraud Detection and Investigation system that combines **Machine Learning (XGBoost)** with an **Agentic AI Workflow (LangGraph & Groq)** to detect, investigate, and explain suspicious banking transactions in real-time.

## 🌟 Key Features

1. **Ultra-Fast ML Detection:** Uses XGBoost to evaluate transactions in milliseconds, generating a Fraud Probability and Anomaly Score.
2. **SHAP Explainability:** Cracks open the ML "black box" to explain exactly which features (e.g., Velocity, Location Change) drove the fraud score.
3. **Smart Agentic Routing (LangGraph):** 
   - **Low Risk (0%):** Automatically approved. Bypasses expensive LLM and Database calls to save time and API costs.
   - **High Risk (>70%):** Triggers a deep-dive investigation.
4. **Context-Aware SQL Agent:** Automatically writes and executes SQL queries against the customer database to fetch historical spending habits, average transactions, and past fraud records.
5. **RAG Compliance Agent:** Uses Qdrant Vector DB to semantically search the bank's internal rulebook and identify specific policy violations (e.g., "New Device Policy").
6. **LLM Investigator (Groq/Qwen):** Synthesizes the ML score, SQL data, and RAG policies into a professional, human-readable English report.
7. **Human-in-the-Loop (HITL):** Presents critical risk transactions to a human manager on a Streamlit dashboard for final `BLOCK` or `APPROVE` action. Includes an Investigation History audit trail.

## 📊 System Performance Metrics

To meet enterprise banking standards, this system is optimized for both speed and reliability:
- **ML Inference Latency:** ~2-5 ms (XGBoost provides near-instant scoring for the initial gateway block/pass).
- **Agentic Workflow Latency:** ~3-5 seconds (Time taken by LangGraph & Groq API to query SQL, RAG, and generate a full English report for high-risk transactions).
- **Model Accuracy:** > 98% (Tested on 50,000 synthetic banking records).
- **Precision:** High Precision Optimization (Strict ML rules to minimize false positives and ensure genuine customers are not mistakenly blocked).

## 🏗️ Architecture

- **Frontend:** Streamlit (Real-time Dashboard & HITL Interface)
- **Backend:** FastAPI (REST API & Agent Orchestration)
- **Machine Learning:** XGBoost & Isolation Forest (Trained on 50k synthetic records)
- **Agent Orchestrator:** LangGraph
- **LLM Provider:** Groq (Qwen 3.8-27b)
- **Vector Database:** Qdrant (Semantic Search for Policies)
- **Relational Database:** SQLite (Customer & Transaction History)
- **Caching:** Redis (Optional, for API response caching)

## 📸 System Workflow

1. **Transaction Request:** Streamlit sends a JSON payload to FastAPI.
2. **ML Evaluation:** XGBoost predicts fraud probability.
3. **Supervisor Decision:** LangGraph routes to approval or deep investigation.
4. **Parallel Evidence Gathering:** SQL Agent fetches history; RAG Agent fetches policies.
5. **Report Generation:** LLM generates a comprehensive summary in Indian Rupees (₹).
6. **Decision Recording:** Human clicks "BLOCK", and the decision is audited in SQLite.

## 🚀 Setup & Installation

### 1. Clone the Repository
``bash
git clone https://github.com/itvikasverma/Agentic-AI-Fraud-Investigation-System.git
cd Agentic-AI-Fraud-Investigation-System
``

### 2. Set Up Virtual Environment
``bash
python -m venv .venv
# On Windows
.\.venv\Scripts\Activate.ps1
# On Mac/Linux
source .venv/bin/activate
``

### 3. Install Dependencies
``bash
pip install -r requirements.txt
``

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
``env
# Database Config
DATABASE_URL=sqlite:///./fraud_db.sqlite

# Qdrant Config
QDRANT_HOST=localhost
QDRANT_PORT=6333

# API Config
API_PORT=8000
STREAMLIT_PORT=8501

# Groq LLM API Key
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
``

### 5. Initialize the Databases
Seed the SQLite database with synthetic customer history and train the ML models:
``bash
python -m database.seed
python -m ml.train
``

Ingest the bank policies into the Qdrant Vector Database:
``bash
python -m rag.ingest
``

### 6. Run the Application

You need two separate terminal windows.

**Terminal 1 (Start the FastAPI Backend):**
``bash
python -m backend.main
``

**Terminal 2 (Start the Streamlit Frontend):**
``bash
streamlit run frontend/streamlit_app.py
``

Open your browser to `http://localhost:8501` to use the Dashboard!

## 🧪 How to Test

1. **Test a Low-Risk Transaction:**
   - Amount: `500`
   - Velocity: `1.0`
   - Failed Attempts: `0`
   - *Result: 0% Fraud, Auto-Approved, No LLM Cost.*

2. **Test a High-Risk Transaction:**
   - Customer ID: `60daafa2-b71b-4d58-b436-cabcc0169112`
   - Amount: `950000`
   - Velocity: `15.0`
   - *Result: 100% Fraud. Agents will fetch customer history from SQL, cite RAG policies, and write a full report demanding Human Review!*

## 📜 License
MIT License
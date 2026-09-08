# 🎯 FlyRank Diagnosis-First Triage API & MCP Agent Node

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MCP Ready](https://img.shields.io/badge/MCP-Agent%20Ready-6B46C1.svg?style=flat)](https://modelcontextprotocol.io)

A live scoring endpoint for FlyRank's diagnosis-first content triage model — ranks pages by likelihood of improvement and diagnoses why they're underperforming.

🌐 **Live Production Endpoint:** [https://flyrank-triage-api.onrender.com/docs](https://flyrank-triage-api.onrender.com/docs)  
📄 **Research Paper & Methodology:** [https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/](https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/)

---

## ⚡ What it Does

POST to `/score` with a page's features (`impressions`, `clicks`, `avg_position`, `in_striking_distance`, `has_real_volume`) and get back:
- **`model_score`**: Probability the page is worth reviewing.
- **`diagnosis`**: Performance classification (`genuine_decline` / `likely_serp_answered` / `ctr_fixable` / `stable_or_improving`).
- **`action`**: Recommended editor action (`refresh_or_rewrite` / `flag_for_human_review_only` / `review_title_and_meta` / `no_action`).

---

## ⚖️ Honest Limitations, Named Directly

- **Deployed container uses lightweight Logistic Regression model (`Flyrank_lr.pkl`).**  
  The live API serves the calibrated Logistic Regression model binary (`1.2 KB`) bundled directly with the repository, ensuring zero OOM errors and instant startup on Render Free Tier (<20MB RAM).
- **No-go checks are enforced, not just documented.**  
  `avg_position=0` and `impressions=0` are refused outright (`422` validation error) rather than scored, per the data-dictionary's warning that `avg_position=0` means "no data," not rank zero.
- **This snapshot has no prior-period value at inference time**, so the diagnosis logic runs on the current values only — a real month-over-month diagnosis (as used in the paper) needs a prior window this simple endpoint doesn't yet accept as input.
- **Free-tier hosting spins down after 15 minutes of inactivity** — the first request after idle time may take 30-60 seconds to respond.

---

## 🚀 Quick Start (Local Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Local API Server
```bash
uvicorn main:app --reload --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser for interactive OpenAPI / Swagger UI.

### 3. Test API Endpoint via cURL
```bash
curl -X POST "https://flyrank-triage-api.onrender.com/score" \
     -H "Content-Type: application/json" \
     -d '{
           "impressions": 450.0,
           "clicks": 12.0,
           "avg_position": 14.2,
           "in_striking_distance": 1,
           "has_real_volume": 1
         }'
```

**Sample JSON Response**:
```json
{
  "model_score": 0.742,
  "diagnosis": "ctr_fixable",
  "action": "review_title_and_meta",
  "details": {
    "ctr": 0.0267,
    "impressions": 450.0,
    "clicks": 12.0,
    "avg_position": 14.2,
    "in_striking_distance": true,
    "has_real_volume": true,
    "prior_period_provided": false
  }
}
```

---

## 🤖 MCP Agent Integration

To run the Model Context Protocol server for AI Agents (Claude Desktop, Gemini, LangChain):
```bash
python mcp_server.py
```
This exposes the tool `flyrank_triage_page` for agent invocation. See [RECRUITER_GUIDE.md](RECRUITER_GUIDE.md) for full architecture details.

---

## 📖 Research Paper & References

Read the full research paper covering methodology, validation split strategy, and Random Forest baseline results:  
👉 [FlyRank ML Research Paper](https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/)


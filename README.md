# 🎯 FlyRank Diagnosis-First Triage API & MCP Agent Node

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MCP Ready](https://img.shields.io/badge/MCP-Agent%20Ready-6B46C1.svg?style=flat)](https://modelcontextprotocol.io)
[![API Security](https://img.shields.io/badge/Security-X--API--Key%20Protected-green.svg?style=flat)](https://fastapi.tiangolo.com)

A live scoring endpoint for FlyRank's diagnosis-first content triage model — ranks pages by likelihood of improvement and diagnoses why they're underperforming.

🌐 **Live Render Endpoint:** [https://flyrank-triage-api.onrender.com/docs](https://flyrank-triage-api.onrender.com/docs)  
📄 **Research Paper & Methodology:** [https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/](https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/)

---

## ⚡ What it Does

POST to `/score` with a page's features (`impressions`, `clicks`, `avg_position`, `in_striking_distance`, `has_real_volume`) and get back:
- **`model_score`**: Probability the page is worth reviewing.
- **`diagnosis`**: Performance classification (`genuine_decline` / `likely_serp_answered` / `ctr_fixable` / `stable_or_improving`).
- **`action`**: Recommended editor action (`refresh_or_rewrite` / `flag_for_human_review_only` / `review_title_and_meta` / `no_action`).

---

## 🔐 API Security & Authentication

This API implements enterprise **`X-API-Key`** header authentication via FastAPI security dependencies.

* **Live Demo Key:** **`flyrank-demo-key`** (Pre-configured for instant zero-friction evaluation).
* **OpenAPI Docs Integration:** Visit `/docs` and click the **Authorize 🔓** button in the upper right corner to enter `flyrank-demo-key`.
* **Environment Configuration:**
  - `REQUIRE_API_KEY`: Set to `true` (default) or `false` (disables auth).
  - `VALID_API_KEYS`: Comma-separated list of accepted API keys.

---

## ⚖️ Honest Limitations, Named Directly

- **Deployed container uses lightweight Logistic Regression model (`model.pkl` & `scaler.pkl`).**  
  The live API serves the calibrated Logistic Regression model and `StandardScaler` binary (`1.6 KB` total) bundled directly with the repository, ensuring zero OOM errors and instant startup on Render Free Tier (<20MB RAM).
- **Research Benchmark vs. Production Serving**:  
  While the offline research paper validates a 200-tree Random Forest classifier (`n_estimators=200`, `Precision@50 = 0.892 ± 0.065`), deploying a 159 MB (compressed) / ~996 MB (uncompressed) Random Forest model on Render free-tier exceeds the 512 MB RAM limit. The calibrated Logistic Regression model is deployed for production container safety, while Random Forest remains the validated research result in the paper.
- **No-go checks are enforced, not just documented.**  
  `avg_position=0` and `impressions=0` are refused outright (`422` validation error) rather than scored, per the data-dictionary's warning that `avg_position=0` means "no data," not rank zero.
- **Prior-period comparison is supported**: Pass optional `impressions_prior` and `clicks_prior` fields for month-over-month diagnosis (`genuine_decline` vs. `likely_serp_answered`).
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
     -H "X-API-Key: flyrank-demo-key" \
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
  "model_score": 0.53,
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

## 🤖 MCP & AI Agent Integration

Exposes the FlyRank ML triage API as an MCP tool, allowing compatible AI applications and agents to access ML predictions without manually entering metrics through the API website.

**How it works**
1. **Run the model:** Submit page metrics (`impressions`, `clicks`, `avg_position`, etc.) to receive a `model_score`, `diagnosis`, and recommended `action`.
2. **Connect through MCP:** The `flyrank_triage_page` tool lets compatible AI applications call the prediction service using a standardized interface.
3. **Enable agent workflows:** When connected to analytics and other tools, an AI agent can retrieve page metrics, run ML triage, interpret results, draft improved titles and meta descriptions, and notify a team or prepare a pull request for approval.

**Run & test**
```bash
python mcp_server.py
python mcp_server.py --cli --sample
```

**Example inference:** `5,000` impressions, `40` clicks, average position `15` → `model_score: 0.532`, `diagnosis: ctr_fixable`, `action: review_title_and_meta`.

**Refusal enforcement:** `impressions=0` triggers a 422 refusal due to insufficient search signal.

MCP makes the model accessible to AI tools; the agent coordinates the workflow. Analytics access, integrations, and permissions must be configured separately.

**Architecture details:** [RECRUITER_GUIDE.md](https://github.com/simon-okosodo-ds/flyrank-triage-api/blob/main/RECRUITER_GUIDE.md)

---

## 📖 Research Paper & References

Read the full research paper covering methodology, validation split strategy, and Random Forest baseline results:  
👉 [FlyRank ML Research Paper](https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/)


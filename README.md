# 🎯 FlyRank Random Forest SEO Triage API & MCP Agent

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MCP Ready](https://img.shields.io/badge/MCP-Agent%20Ready-6B46C1.svg?style=flat)](https://modelcontextprotocol.io)

Live microservice wrapping FlyRank's trained Random Forest classifier (`Flyrank.model.pkl`) to serve real-time SEO triage scores, diagnostic classifications, and prescriptive content actions.

Includes an **MCP (Model Context Protocol)** server slot for autonomous AI Agent integration and multi-cloud free tier deployment templates for Hugging Face Spaces and Render.

---

## ⚡ Quick Start (Local Setup)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run API Server
```bash
uvicorn main:app --reload --port 8000
```
Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to view interactive OpenAPI / Swagger UI documentation.

### 3. Test API Endpoint via cURL
```bash
curl -X POST "http://127.0.0.1:8000/score" \
     -H "Content-Type: application/json" \
     -d '{
           "impressions": 450.0,
           "clicks": 12.0,
           "avg_position": 14.2,
           "in_striking_distance": 1,
           "has_real_volume": 1
         }'
```

**Expected Response**:
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

To run the Model Context Protocol server for AI Agents:
```bash
python mcp_server.py
```
This registers the tool `flyrank_triage_page` for agent execution. See [RECRUITER_GUIDE.md](RECRUITER_GUIDE.md) for detailed MCP agent workflows.

---

## ☁️ Free Cloud Deployment

- **Hugging Face Spaces**: Push repository with included `Dockerfile` to a Docker Space.
- **Render**: Connect repository and deploy using included `render.yaml`.

For recruiter guide and complete architectural breakdown, refer to [RECRUITER_GUIDE.md](RECRUITER_GUIDE.md).

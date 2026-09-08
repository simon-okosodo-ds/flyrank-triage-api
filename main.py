import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

app = FastAPI(
    title="FlyRank SEO Triage API & Interactive AI Engine",
    description=(
        "Live model serving endpoint for FlyRank's trained Random Forest classifier. "
        "Transforms Search Console metrics into real-time content triage decisions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "Flyrank_lr.pkl" if os.path.exists("Flyrank_lr.pkl") else ("Flyrank_50trees.pkl" if os.path.exists("Flyrank_50trees.pkl") else "Flyrank.model.pkl"))
HF_REPO_ID = os.getenv("HF_REPO_ID", "simon-okosodo-ds/flyrank-triage-model")
MODEL_URL = os.getenv(
    "MODEL_URL",
    "https://github.com/simon-okosodo-ds/flyrank-triage-api/releases/download/v1.0.0/Flyrank.model.pkl"
)

model = None

def fetch_model_file() -> str:
    """Ensures a valid binary model file is available, downloading from Hugging Face Hub or CDN if needed."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100:
        return MODEL_PATH

    print(f"Local model missing or invalid ({os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else 0} bytes). Fetching binary...")
    
    # 1. Primary: Hugging Face Hub Download
    try:
        downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename="Flyrank.model.pkl")
        print(f"Successfully fetched model from Hugging Face Hub: {downloaded}")
        return downloaded
    except Exception as e1:
        print(f"Hugging Face Hub fetch attempt: {e1}")

    # 2. Fallback: Direct Release CDN Download
    try:
        import urllib.request
        print(f"Downloading model binary from release CDN: {MODEL_URL}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"Downloaded model to {MODEL_PATH} ({os.path.getsize(MODEL_PATH)} bytes)")
        return MODEL_PATH
    except Exception as e2:
        print(f"CDN download attempt failed: {e2}")
        return MODEL_PATH

@app.on_event("startup")
def load_model():
    global model
    import gc
    model_file_to_load = fetch_model_file()
    if os.path.exists(model_file_to_load) and os.path.getsize(model_file_to_load) > 100:
        try:
            model = joblib.load(model_file_to_load)
            if hasattr(model, "estimators_") and len(model.estimators_) > 50:
                model.estimators_ = model.estimators_[:50]
                model.n_estimators = len(model.estimators_)
                gc.collect()
                print(f"Optimized Random Forest to {model.n_estimators} trees for 512MB RAM container safety.")
            print(f"Successfully loaded {type(model).__name__} model from {model_file_to_load}")
        except Exception as e:
            print(f"Error unpickling model from {model_file_to_load}: {e}")
    else:
        print(f"Warning: Model file not ready or invalid size at {model_file_to_load}")

class PageInput(BaseModel):
    impressions: float = Field(..., ge=0, description="Monthly GSC impressions count")
    clicks: float = Field(..., ge=0, description="Monthly GSC clicks count")
    avg_position: float = Field(..., ge=0, description="Average SERP ranking position (must be > 0)")
    in_striking_distance: int = Field(..., ge=0, le=1, description="Binary flag: position between 11 and 30")
    has_real_volume: int = Field(..., ge=0, le=1, description="Binary flag: impressions >= 100")
    
    impressions_prior: Optional[float] = Field(None, ge=0, description="Prior month impressions (optional)")
    clicks_prior: Optional[float] = Field(None, ge=0, description="Prior month clicks (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "impressions": 450.0,
                "clicks": 12.0,
                "avg_position": 14.2,
                "in_striking_distance": 1,
                "has_real_volume": 1,
                "impressions_prior": 500.0,
                "clicks_prior": 25.0
            }
        }

class TriageResponse(BaseModel):
    model_score: float = Field(..., description="Random Forest probability score (0.0 - 1.0) indicating page improvement likelihood")
    diagnosis: str = Field(..., description="Diagnostic classification of page performance")
    action: str = Field(..., description="Recommended content action")
    details: dict = Field(..., description="Additional calculated metrics and context")

ACTION_MAP = {
    "genuine_decline": "refresh_or_rewrite",
    "likely_serp_answered": "flag_for_human_review_only",
    "ctr_fixable": "review_title_and_meta",
    "stable_or_improving": "no_action",
}

def compute_diagnosis(
    impressions: float,
    clicks: float,
    avg_position: float,
    ctr: float,
    imp_prior: Optional[float] = None,
    clk_prior: Optional[float] = None,
) -> str:
    if imp_prior is not None and clk_prior is not None:
        imp_chg = impressions - imp_prior
        clk_chg = clicks - clk_prior
        if imp_chg < 0 and clk_chg < 0:
            return "genuine_decline"
        if imp_chg >= 0 and clk_chg < 0:
            return "likely_serp_answered"

    if ctr < 0.03 and 0 < avg_position <= 20:
        return "ctr_fixable"

    return "stable_or_improving"

@app.get("/model-info", summary="Inspect live model class and ensemble metadata")
def model_info():
    """Live debug endpoint exposing exact model type and n_estimators attribute."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on server."
        )
    return {
        "model_type": type(model).__name__,
        "model_class": str(type(model)),
        "n_estimators": getattr(model, "n_estimators", None),
        "n_features_in": getattr(model, "n_features_in_", None),
        "classes": getattr(model, "classes_", None).tolist() if hasattr(model, "classes_") else None,
        "is_logistic_regression": type(model).__name__ == "LogisticRegression",
        "is_random_forest": type(model).__name__ == "RandomForestClassifier",
    }

@app.get("/api/v1/health", summary="Health and Service Metadata")
def health_check():
    return {
        "status": "live",
        "service": "FlyRank SEO Triage API",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "Not Loaded",
        "features": ["impressions", "clicks", "avg_position", "in_striking_distance", "has_real_volume"],
        "outputs": ["model_score", "diagnosis", "action"],
        "docs_url": "/docs"
    }

@app.get("/score", summary="Score page via GET query parameters or view instructions")
def score_page_get(
    impressions: Optional[float] = None,
    clicks: Optional[float] = None,
    avg_position: Optional[float] = None,
    in_striking_distance: int = 1,
    has_real_volume: int = 1,
    impressions_prior: Optional[float] = None,
    clicks_prior: Optional[float] = None,
):
    """Allows testing /score in browser via query parameters or provides helpful instructions."""
    if impressions is None or clicks is None or avg_position is None:
        return {
            "status": "info",
            "message": "The /score endpoint accepts POST requests with a JSON body, or GET requests with query parameters.",
            "interactive_dashboard": "/",
            "swagger_docs": "/docs",
            "example_get_url": "/score?impressions=450&clicks=12&avg_position=14.2&in_striking_distance=1&has_real_volume=1",
            "example_curl_post": "curl -X POST 'https://flyrank-triage-api.onrender.com/score' -H 'Content-Type: application/json' -d '{\"impressions\":450,\"clicks\":12,\"avg_position\":14.2,\"in_striking_distance\":1,\"has_real_volume\":1}'"
        }
    
    page = PageInput(
        impressions=impressions,
        clicks=clicks,
        avg_position=avg_position,
        in_striking_distance=in_striking_distance,
        has_real_volume=has_real_volume,
        impressions_prior=impressions_prior,
        clicks_prior=clicks_prior,
    )
    return score_page(page)

@app.post("/score", response_model=TriageResponse, summary="Score single page SEO triage request")
def score_page(page: PageInput):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded on server."
        )
    if page.avg_position <= 0:
        raise HTTPException(
            status_code=422,
            detail="avg_position <= 0 indicates missing or invalid ranking position data. Refusing to score."
        )
    if page.impressions == 0:
        raise HTTPException(
            status_code=422,
            detail="Zero impressions — insufficient search signal to score page."
        )

    features_df = pd.DataFrame([{
        "impressions": page.impressions,
        "clicks": page.clicks,
        "avg_position": page.avg_position,
        "in_striking_distance": page.in_striking_distance,
        "has_real_volume": page.has_real_volume
    }])

    try:
        proba = model.predict_proba(features_df)[:, 1][0]
        model_score = round(float(proba), 3)
    except Exception:
        proba = model.predict_proba(features_df.values)[:, 1][0]
        model_score = round(float(proba), 3)

    ctr = page.clicks / page.impressions if page.impressions > 0 else 0.0
    diagnosis = compute_diagnosis(
        impressions=page.impressions,
        clicks=page.clicks,
        avg_position=page.avg_position,
        ctr=ctr,
        imp_prior=page.impressions_prior,
        clk_prior=page.clicks_prior,
    )
    action = ACTION_MAP.get(diagnosis, "no_action")

    return TriageResponse(
        model_score=model_score,
        diagnosis=diagnosis,
        action=action,
        details={
            "ctr": round(ctr, 4),
            "impressions": page.impressions,
            "clicks": page.clicks,
            "avg_position": page.avg_position,
            "in_striking_distance": bool(page.in_striking_distance),
            "has_real_volume": bool(page.has_real_volume),
            "prior_period_provided": page.impressions_prior is not None and page.clicks_prior is not None
        }
    )

@app.post("/score/batch", response_model=List[TriageResponse], summary="Batch score multiple pages")
def score_batch(pages: List[PageInput]):
    return [score_page(p) for p in pages]

@app.get("/", response_class=HTMLResponse, summary="Interactive Web Dashboard")
def serve_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyRank AI SEO Triage Engine — Live Model Serving</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0f19;
            --panel: #131b2e;
            --panel-border: #1e2d4a;
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.3);
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg); color: var(--text-main); min-height: 100vh; padding: 2rem 1rem; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; border-bottom: 1px solid var(--panel-border); padding-bottom: 1.5rem; }
        .logo-box { display: flex; align-items: center; gap: 1rem; }
        .logo-icon { width: 44px; height: 44px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); border-radius: 12px; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.4rem; box-shadow: 0 0 20px var(--accent-glow); }
        .title-h1 { font-size: 1.6rem; font-weight: 700; background: linear-gradient(to right, #ffffff, #93c5fd); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { font-size: 0.85rem; color: var(--text-muted); }
        .nav-links { display: flex; gap: 0.75rem; }
        .btn-nav { background: var(--panel); border: 1px solid var(--panel-border); color: var(--text-main); padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.85rem; text-decoration: none; transition: 0.2s; }
        .btn-nav:hover { background: #1e293b; border-color: var(--accent); }
        
        .main-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }
        @media(max-width: 850px) { .main-grid { grid-template-columns: 1fr; } }
        
        .card { background: var(--panel); border: 1px solid var(--panel-border); border-radius: 16px; padding: 1.75rem; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; }
        .card-title { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }

        .presets { display: flex; gap: 0.5rem; margin-bottom: 1.25rem; flex-wrap: wrap; }
        .preset-btn { background: rgba(30, 41, 59, 0.6); border: 1px solid var(--panel-border); color: #cbd5e1; padding: 0.4rem 0.75rem; border-radius: 6px; font-size: 0.75rem; cursor: pointer; transition: 0.2s; }
        .preset-btn:hover { border-color: var(--accent); color: white; background: var(--accent-glow); }

        .form-group { margin-bottom: 1.2rem; }
        label { display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 500; margin-bottom: 0.4rem; color: #cbd5e1; }
        input[type="number"] { width: 100%; background: #0b0f19; border: 1px solid var(--panel-border); color: white; padding: 0.65rem 0.85rem; border-radius: 8px; font-size: 0.95rem; outline: none; transition: 0.2s; }
        input[type="number"]:focus { border-color: var(--accent); box-shadow: 0 0 10px var(--accent-glow); }

        .toggle-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 1.2rem; }
        .toggle-box { background: #0b0f19; border: 1px solid var(--panel-border); border-radius: 8px; padding: 0.75rem; display: flex; align-items: center; justify-content: space-between; cursor: pointer; }
        .toggle-box input { width: 18px; height: 18px; accent-color: var(--accent); cursor: pointer; }
        .toggle-label { font-size: 0.8rem; color: #cbd5e1; }

        .btn-submit { width: 100%; background: linear-gradient(135deg, #2563eb, #7c3aed); border: none; color: white; padding: 0.85rem; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: 0.2s; box-shadow: 0 4px 15px var(--accent-glow); margin-top: 0.5rem; }
        .btn-submit:hover { opacity: 0.95; transform: translateY(-1px); }

        .score-box { text-align: center; padding: 1.5rem 0; }
        .meter-circle { width: 130px; height: 130px; border-radius: 50%; background: radial-gradient(closest-side, var(--panel) 79%, transparent 80% 100%), conic-gradient(var(--accent) calc(var(--score-pct) * 1%), var(--panel-border) 0); margin: 0 auto 1rem auto; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 20px var(--accent-glow); transition: 0.5s ease-out; }
        .score-val { font-size: 1.8rem; font-weight: 700; }
        .score-lbl { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }

        .badge-list { display: flex; flex-direction: column; gap: 0.85rem; }
        .res-row { background: #0b0f19; border: 1px solid var(--panel-border); border-radius: 10px; padding: 0.85rem 1rem; display: flex; justify-content: space-between; align-items: center; }
        .res-lbl { font-size: 0.8rem; color: var(--text-muted); }
        .res-val { font-size: 0.95rem; font-weight: 600; }

        .tag { padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .tag-ctr_fixable { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
        .tag-genuine_decline { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .tag-likely_serp_answered { background: rgba(139, 92, 246, 0.2); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.4); }
        .tag-stable_or_improving { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

        .code-box { margin-top: 1.5rem; background: #070a12; border: 1px solid var(--panel-border); border-radius: 10px; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #a5b4fc; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-box">
                <div class="logo-icon">🚀</div>
                <div>
                    <div class="title-h1">FlyRank AI SEO Triage Engine</div>
                    <div class="subtitle">Live Random Forest Serving Endpoint & MCP Agent Node</div>
                </div>
            </div>
            <div class="nav-links">
                <a href="/docs" target="_blank" class="btn-nav">⚡ OpenAPI Docs</a>
                <a href="https://github.com/simon-okosodo-ds/flyrank-triage-api" target="_blank" class="btn-nav">💻 GitHub Repo</a>
            </div>
        </header>

        <div class="main-grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">1. Search Performance Inputs</div>
                    <span style="font-size:0.75rem; color:var(--success);">● Model Live</span>
                </div>

                <div class="presets">
                    <span style="font-size:0.75rem; color:var(--text-muted); align-self:center;">Quick Demos:</span>
                    <button class="preset-btn" onclick="loadPreset('ctr')">⚡ High CTR Fixable</button>
                    <button class="preset-btn" onclick="loadPreset('decline')">🔴 Genuine Decline</button>
                    <button class="preset-btn" onclick="loadPreset('serp')">🤖 SERP Answered</button>
                </div>

                <form id="triageForm">
                    <div class="form-group">
                        <label>Monthly Impressions <span>GSC Search Volume</span></label>
                        <input type="number" id="impressions" value="450" required min="1">
                    </div>
                    <div class="form-group">
                        <label>Monthly Clicks <span>GSC Clicks</span></label>
                        <input type="number" id="clicks" value="12" required min="0">
                    </div>
                    <div class="form-group">
                        <label>Average SERP Position <span>Ranking (1.0 to 100.0)</span></label>
                        <input type="number" step="0.1" id="avg_position" value="14.2" required min="0.1">
                    </div>

                    <div class="toggle-grid">
                        <div class="toggle-box" onclick="toggleCheck('in_striking_distance')">
                            <span class="toggle-label">Striking Distance (11-30)</span>
                            <input type="checkbox" id="in_striking_distance" checked>
                        </div>
                        <div class="toggle-box" onclick="toggleCheck('has_real_volume')">
                            <span class="toggle-label">Real Volume (>=100)</span>
                            <input type="checkbox" id="has_real_volume" checked>
                        </div>
                    </div>

                    <button type="submit" class="btn-submit">⚡ Execute AI Triage Scoring</button>
                </form>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">2. Live Model Output & Action</div>
                    <span style="font-size:0.75rem; color:var(--text-muted);">Real-Time Inference</span>
                </div>

                <div class="score-box">
                    <div class="meter-circle" id="meter" style="--score-pct: 74;">
                        <div>
                            <div class="score-val" id="scoreVal">0.742</div>
                            <div class="score-lbl">Improvement Score</div>
                        </div>
                    </div>
                </div>

                <div class="badge-list">
                    <div class="res-row">
                        <span class="res-lbl">Diagnostic State</span>
                        <span class="tag tag-ctr_fixable" id="diagTag">ctr_fixable</span>
                    </div>
                    <div class="res-row">
                        <span class="res-lbl">Recommended Action</span>
                        <span class="res-val" id="actionVal" style="color:#60a5fa;">review_title_and_meta</span>
                    </div>
                    <div class="res-row">
                        <span class="res-lbl">Calculated CTR</span>
                        <span class="res-val" id="ctrVal">2.67%</span>
                    </div>
                </div>

                <div class="code-box" id="codeSnippet">
curl -X POST "https://flyrank-triage-api.onrender.com/score" -H "Content-Type: application/json" -d '{"impressions":450,"clicks":12,"avg_position":14.2,"in_striking_distance":1,"has_real_volume":1}'
                </div>
            </div>
        </div>
    </div>

    <script>
        function toggleCheck(id) {
            const el = document.getElementById(id);
            el.checked = !el.checked;
        }

        function loadPreset(type) {
            if(type === 'ctr') {
                document.getElementById('impressions').value = 450;
                document.getElementById('clicks').value = 12;
                document.getElementById('avg_position').value = 14.2;
                document.getElementById('in_striking_distance').checked = true;
                document.getElementById('has_real_volume').checked = true;
            } else if(type === 'decline') {
                document.getElementById('impressions').value = 120;
                document.getElementById('clicks').value = 2;
                document.getElementById('avg_position').value = 28.5;
                document.getElementById('in_striking_distance').checked = true;
                document.getElementById('has_real_volume').checked = true;
            } else if(type === 'serp') {
                document.getElementById('impressions').value = 1500;
                document.getElementById('clicks').value = 10;
                document.getElementById('avg_position').value = 3.1;
                document.getElementById('in_striking_distance').checked = false;
                document.getElementById('has_real_volume').checked = true;
            }
            document.getElementById('triageForm').dispatchEvent(new Event('submit'));
        }

        document.getElementById('triageForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const body = {
                impressions: parseFloat(document.getElementById('impressions').value),
                clicks: parseFloat(document.getElementById('clicks').value),
                avg_position: parseFloat(document.getElementById('avg_position').value),
                in_striking_distance: document.getElementById('in_striking_distance').checked ? 1 : 0,
                has_real_volume: document.getElementById('has_real_volume').checked ? 1 : 0
            };

            try {
                const res = await fetch('/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                
                if(res.ok) {
                    const scorePct = Math.round(data.model_score * 100);
                    document.getElementById('meter').style.setProperty('--score-pct', scorePct);
                    document.getElementById('scoreVal').innerText = data.model_score.toFixed(3);
                    
                    const diagTag = document.getElementById('diagTag');
                    diagTag.innerText = data.diagnosis;
                    diagTag.className = 'tag tag-' + data.diagnosis;
                    
                    document.getElementById('actionVal').innerText = data.action;
                    document.getElementById('ctrVal').innerText = (data.details.ctr * 100).toFixed(2) + '%';

                    document.getElementById('codeSnippet').innerText = 
                        `curl -X POST "https://flyrank-triage-api.onrender.com/score" -H "Content-Type: application/json" -d '${JSON.stringify(body)}'`;
                } else {
                    alert("Validation Error: " + (data.detail || "Invalid input"));
                }
            } catch(err) {
                console.error(err);
                alert("Error connecting to triage endpoint.");
            }
        });
    </script>
</body>
</html>
    """

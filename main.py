"""
FlyRank SEO Triage API  --  v2.0.0

Serves the Random Forest from the capstone paper ("Diagnosis-First Content Triage").
What changed from v1.1 (which served a Logistic Regression + scaler):
  * Loads model_artifact.joblib (Random Forest, 5 features). No scaler: a forest is trained on raw values.
  * FAILS LOUDLY if the model file is missing, has the wrong features, or is the wrong class.
    v1.1 silently fell back to downloading an older model; v2 never does.
  * in_striking_distance / has_real_volume are DERIVED on the server from avg_position / impressions,
    exactly as in training. Client-supplied flags are optional and are checked, not trusted.
  * /version reports the model fingerprint (sha256) so anyone can confirm which file is serving.
  * Batch endpoint is capped and vectorised; GET /score no longer 500s on bad input; error messages no
    longer echo the submitted key; CORS no longer combines a wildcard origin with credentials.
"""
import hashlib
import hmac
import os
from typing import List, Optional

import joblib
import pandas as pd
import sklearn
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------------- config
API_KEY_NAME = "X-API-Key"
DEFAULT_DEMO_KEY = "flyrank-demo-key"          # public demo key, intentionally documented
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

MODEL_PATH = os.getenv("MODEL_PATH", "model_artifact.joblib")
MODEL_URL = os.getenv("MODEL_URL", "")          # optional; opt-in download if the file is absent
EXPECTED_MODEL_SHA = os.getenv("EXPECTED_MODEL_SHA", "24fa9123e005")   # first 12 hex chars of sha256
# Serve ONLY the file from the paper. Set STRICT_FINGERPRINT=false (and update EXPECTED_MODEL_SHA) to serve a retrained model.
STRICT_FINGERPRINT = os.getenv("STRICT_FINGERPRINT", "true").lower() in ("true", "1", "yes")
FEATURES = ["impressions", "clicks", "avg_position", "in_striking_distance", "has_real_volume"]
MAX_BATCH = int(os.getenv("MAX_BATCH", "500"))
# CTR (as a FRACTION, clicks/impressions) below which a well-ranked page is called "ctr_fixable".
CTR_FIXABLE_MAX = float(os.getenv("CTR_FIXABLE_MAX", "0.03"))

REPORTED_VALIDATION = {
    "precision_at_50_baseline": 0.400,
    "precision_at_50_nested_cv": "0.892 +/- 0.065",
    "bootstrap_95ci_gap_over_baseline": [0.240, 0.600],
    "source": "capstone paper (nested cross-validation). Reported estimate; NOT recomputed by this API.",
}

model = None
model_sha: Optional[str] = None
load_error: Optional[str] = None


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validates the X-API-Key header against the configured keys."""
    if os.getenv("REQUIRE_API_KEY", "true").lower() not in ("true", "1", "yes"):
        return api_key or DEFAULT_DEMO_KEY
    valid_keys = [k.strip() for k in os.getenv("VALID_API_KEYS", DEFAULT_DEMO_KEY).split(",") if k.strip()]
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing API key. Send the header '{API_KEY_NAME}: {DEFAULT_DEMO_KEY}' (public demo key).",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    if not any(hmac.compare_digest(api_key.encode(), k.encode()) for k in valid_keys):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
            headers={"WWW-Authenticate": API_KEY_NAME},
        )
    return api_key


app = FastAPI(
    title="FlyRank SEO Triage API & Interactive AI Engine",
    description=(
        "Live serving endpoint for the Random Forest ranking model from the capstone paper "
        "'Diagnosis-First Content Triage'. Turns Search Console metrics into a model score, a diagnosis "
        "and a recommended editor action.\n\n"
        "**Model check:** open `/version` -- it returns the fingerprint of the file being served.\n\n"
        "**Authentication:** endpoints require an `X-API-Key` header. Public demo key: "
        "**`flyrank-demo-key`** (or use the **Authorize** button above)."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,          # no cookies are used; wildcard origin + credentials is not valid
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------- model loading
def _download_if_requested() -> None:
    """Opt-in only: download MODEL_URL if the file is absent. Never falls back to any other model."""
    if os.path.exists(MODEL_PATH) or not MODEL_URL:
        return
    import urllib.request
    print(f"Model file absent; downloading from MODEL_URL -> {MODEL_PATH}")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


@app.on_event("startup")
def load_model() -> None:
    global model, model_sha, load_error
    model, model_sha, load_error = None, None, None
    try:
        _download_if_requested()
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"model file not found at '{MODEL_PATH}'")
        with open(MODEL_PATH, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()[:12]
        if STRICT_FINGERPRINT and sha != EXPECTED_MODEL_SHA:
            raise ValueError(f"model fingerprint {sha} != expected {EXPECTED_MODEL_SHA}; refusing to serve an unlisted model "
                             f"(STRICT_FINGERPRINT=false to override)")
        m = joblib.load(MODEL_PATH)
        if not hasattr(m, "predict_proba"):
            raise TypeError(f"{type(m).__name__} has no predict_proba")
        if getattr(m, "n_features_in_", None) != len(FEATURES):
            raise ValueError(f"model expects {getattr(m, 'n_features_in_', None)} features, API provides {len(FEATURES)}")
        names = list(getattr(m, "feature_names_in_", []))
        if names and names != FEATURES:
            raise ValueError(f"model feature order {names} != API feature order {FEATURES}")
        model, model_sha = m, sha
        print(f"MODEL LOADED: {type(m).__name__} from {MODEL_PATH} sha={sha} "
              f"(matches expected: {sha == EXPECTED_MODEL_SHA}) sklearn={sklearn.__version__}")
    except Exception as exc:                      # fail loudly, never serve a different model
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"MODEL NOT LOADED -> {load_error}")


# ----------------------------------------------------------------------------- schemas
class PageInput(BaseModel):
    impressions: float = Field(..., ge=0, description="Monthly GSC impressions")
    clicks: float = Field(..., ge=0, description="Monthly GSC clicks")
    avg_position: float = Field(..., ge=0, description="Average SERP position (must be > 0)")
    in_striking_distance: Optional[int] = Field(
        None, ge=0, le=1, description="Optional. Derived on the server (position > 10 and <= 30); if sent it is checked, not used")
    has_real_volume: Optional[int] = Field(
        None, ge=0, le=1, description="Optional. Derived on the server (impressions >= 100); if sent it is checked, not used")
    impressions_prior: Optional[float] = Field(None, ge=0, description="Prior month impressions (optional)")
    clicks_prior: Optional[float] = Field(None, ge=0, description="Prior month clicks (optional)")

    class Config:
        json_schema_extra = {
            "example": {
                "impressions": 450.0, "clicks": 12.0, "avg_position": 14.2,
                "impressions_prior": 500.0, "clicks_prior": 25.0,
            }
        }


class TriageResponse(BaseModel):
    model_score: float = Field(..., description="Random Forest probability (0-1) that the page's position or CTR improves next month")
    diagnosis: str = Field(..., description="Diagnostic classification of page performance")
    action: str = Field(..., description="Recommended content action")
    details: dict = Field(..., description="Derived values, flags used, and model fingerprint")


ACTION_MAP = {
    "genuine_decline": "refresh_or_rewrite",
    "likely_serp_answered": "flag_for_human_review_only",
    "ctr_fixable": "review_title_and_meta",
    "stable_or_improving": "no_action",
}


def compute_diagnosis(impressions: float, clicks: float, avg_position: float, ctr: float,
                      imp_prior: Optional[float] = None, clk_prior: Optional[float] = None) -> str:
    if imp_prior is not None and clk_prior is not None:
        imp_chg, clk_chg = impressions - imp_prior, clicks - clk_prior
        if imp_chg < 0 and clk_chg < 0:
            return "genuine_decline"
        if imp_chg >= 0 and clk_chg < 0:
            return "likely_serp_answered"
    if ctr < CTR_FIXABLE_MAX and 0 < avg_position <= 20:
        return "ctr_fixable"
    return "stable_or_improving"


def _validate(page: PageInput) -> None:
    if page.avg_position <= 0:
        raise HTTPException(422, "avg_position <= 0 means missing ranking data. Refusing to score.")
    if page.impressions <= 0:
        raise HTTPException(422, "Zero impressions: insufficient search signal to score this page.")
    if page.clicks > page.impressions:
        raise HTTPException(422, "clicks cannot exceed impressions.")


def _derive_flags(page: PageInput):
    """Flags exactly as defined in training: striking distance = 10 < position <= 30; real volume = impressions >= 100."""
    striking = int(10 < page.avg_position <= 30)
    volume = int(page.impressions >= 100)
    mismatch = []
    if page.in_striking_distance is not None and page.in_striking_distance != striking:
        mismatch.append("in_striking_distance")
    if page.has_real_volume is not None and page.has_real_volume != volume:
        mismatch.append("has_real_volume")
    return striking, volume, mismatch


def _require_model() -> None:
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Model is not loaded on the server. {load_error or ''}".strip(),
        )


def _score_pages(pages: List[PageInput]) -> List[TriageResponse]:
    _require_model()
    rows, meta = [], []
    for page in pages:
        _validate(page)
        striking, volume, mismatch = _derive_flags(page)
        rows.append([page.impressions, page.clicks, page.avg_position, striking, volume])
        meta.append((page, striking, volume, mismatch))
    probs = model.predict_proba(pd.DataFrame(rows, columns=FEATURES))[:, 1]

    out = []
    for p, (page, striking, volume, mismatch) in zip(probs, meta):
        ctr = page.clicks / page.impressions
        diagnosis = compute_diagnosis(page.impressions, page.clicks, page.avg_position, ctr,
                                      page.impressions_prior, page.clicks_prior)
        details = {
            "ctr": round(ctr, 4),
            "impressions": page.impressions,
            "clicks": page.clicks,
            "avg_position": page.avg_position,
            "in_striking_distance": bool(striking),
            "has_real_volume": bool(volume),
            "flags_source": "derived_from_inputs",
            "flags_sent_but_overridden": mismatch,
            "prior_period_provided": page.impressions_prior is not None and page.clicks_prior is not None,
            "model_sha": model_sha,
        }
        out.append(TriageResponse(model_score=round(float(p), 3), diagnosis=diagnosis,
                                  action=ACTION_MAP[diagnosis], details=details))
    return out


# ----------------------------------------------------------------------------- endpoints
@app.get("/version", summary="Which model file is serving? (fingerprint)")
def version():
    info = {
        "api_version": app.version,
        "model_loaded": model is not None,
        "load_error": load_error,
        "model_sha256_12": model_sha,
        "expected_sha256_12": EXPECTED_MODEL_SHA,
        "fingerprint_matches_expected": model_sha is not None and model_sha == EXPECTED_MODEL_SHA,
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "reported_validation": REPORTED_VALIDATION,
    }
    if model is not None:
        info.update({
            "model_type": type(model).__name__,
            "n_estimators": getattr(model, "n_estimators", None),
            "max_depth": getattr(model, "max_depth", None),
            "min_samples_leaf": getattr(model, "min_samples_leaf", None),
            "n_features_in": getattr(model, "n_features_in_", None),
        })
    return info


@app.get("/model-info", summary="Inspect the loaded model")
def model_info():
    if model is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Model is not loaded on server. {load_error or ''}".strip())
    return {
        "model_type": type(model).__name__,
        "model_class": str(type(model)),
        "n_estimators": getattr(model, "n_estimators", None),
        "max_depth": getattr(model, "max_depth", None),
        "min_samples_leaf": getattr(model, "min_samples_leaf", None),
        "n_features_in": getattr(model, "n_features_in_", None),
        "feature_names": list(getattr(model, "feature_names_in_", [])),
        "classes": getattr(model, "classes_", None).tolist() if hasattr(model, "classes_") else None,
        "scaler_used": False,
        "is_logistic_regression": type(model).__name__ == "LogisticRegression",
        "is_random_forest": type(model).__name__ == "RandomForestClassifier",
        "model_sha256_12": model_sha,
    }


@app.get("/api/v1/health", summary="Health and service metadata")
def health_check():
    return {
        "status": "live" if model is not None else "degraded",
        "service": "FlyRank SEO Triage API",
        "api_version": app.version,
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model is not None else "Not Loaded",
        "model_sha256_12": model_sha,
        "fingerprint_matches_expected": model_sha is not None and model_sha == EXPECTED_MODEL_SHA,
        "scaler_used": False,
        "features": FEATURES,
        "outputs": ["model_score", "diagnosis", "action"],
        "docs_url": "/docs",
    }


@app.get("/score", summary="Score a page via GET query parameters (or view instructions)",
         dependencies=[Security(verify_api_key)])
def score_page_get(
    impressions: Optional[float] = None,
    clicks: Optional[float] = None,
    avg_position: Optional[float] = None,
    in_striking_distance: Optional[int] = None,
    has_real_volume: Optional[int] = None,
    impressions_prior: Optional[float] = None,
    clicks_prior: Optional[float] = None,
):
    if impressions is None or clicks is None or avg_position is None:
        return {
            "status": "info",
            "message": "POST a JSON body to /score, or call GET /score with query parameters.",
            "interactive_dashboard": "/",
            "swagger_docs": "/docs",
            "api_key_required": True,
            "demo_api_key": DEFAULT_DEMO_KEY,
            "example_get_url": "/score?impressions=450&clicks=12&avg_position=14.2",
            "example_curl_post": (
                "curl -X POST 'https://flyrank-triage-api.onrender.com/score' -H 'Content-Type: application/json' "
                f"-H 'X-API-Key: {DEFAULT_DEMO_KEY}' "
                "-d '{\"impressions\":450,\"clicks\":12,\"avg_position\":14.2}'"
            ),
        }
    try:
        page = PageInput(impressions=impressions, clicks=clicks, avg_position=avg_position,
                         in_striking_distance=in_striking_distance, has_real_volume=has_real_volume,
                         impressions_prior=impressions_prior, clicks_prior=clicks_prior)
    except Exception as exc:
        raise HTTPException(422, f"Invalid input: {exc}")
    return _score_pages([page])[0]


@app.post("/score", response_model=TriageResponse, summary="Score a single page",
          dependencies=[Security(verify_api_key)])
def score_page(page: PageInput):
    return _score_pages([page])[0]


@app.post("/score/batch", response_model=List[TriageResponse], summary="Batch score pages",
          dependencies=[Security(verify_api_key)])
def score_batch(pages: List[PageInput]):
    if not pages:
        raise HTTPException(422, "Send at least one page.")
    if len(pages) > MAX_BATCH:
        raise HTTPException(422, f"Batch too large: {len(pages)} pages (max {MAX_BATCH}).")
    return _score_pages(pages)


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyRank AI SEO Triage Engine — Live Model Serving</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root { --bg:#0b0f19; --panel:#131b2e; --panel-border:#1e2d4a; --accent:#3b82f6; --accent-glow:rgba(59,130,246,0.3);
                --success:#10b981; --text-main:#f3f4f6; --text-muted:#9ca3af; }
        * { box-sizing:border-box; margin:0; padding:0; font-family:'Outfit',sans-serif; }
        body { background:var(--bg); color:var(--text-main); min-height:100vh; padding:2rem 1rem; }
        .container { max-width:1100px; margin:0 auto; }
        header { display:flex; justify-content:space-between; align-items:center; margin-bottom:2rem; border-bottom:1px solid var(--panel-border); padding-bottom:1.5rem; flex-wrap:wrap; gap:1rem; }
        .logo-box { display:flex; align-items:center; gap:1rem; }
        .logo-icon { width:44px; height:44px; background:linear-gradient(135deg,#3b82f6,#8b5cf6); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.4rem; box-shadow:0 0 20px var(--accent-glow); }
        .title-h1 { font-size:1.6rem; font-weight:700; background:linear-gradient(to right,#ffffff,#93c5fd); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
        .subtitle { font-size:0.85rem; color:var(--text-muted); }
        .nav-links { display:flex; gap:0.75rem; flex-wrap:wrap; }
        .btn-nav { background:var(--panel); border:1px solid var(--panel-border); color:var(--text-main); padding:0.5rem 1rem; border-radius:8px; font-size:0.85rem; text-decoration:none; transition:0.2s; }
        .btn-nav:hover { background:#1e293b; border-color:var(--accent); }
        .main-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }
        @media(max-width:850px){ .main-grid { grid-template-columns:1fr; } }
        .card { background:var(--panel); border:1px solid var(--panel-border); border-radius:16px; padding:1.75rem; box-shadow:0 10px 30px rgba(0,0,0,0.4); }
        .card-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; }
        .card-title { font-size:1.1rem; font-weight:600; color:#e2e8f0; }
        .presets { display:flex; gap:0.5rem; margin-bottom:1.25rem; flex-wrap:wrap; }
        .preset-btn { background:rgba(30,41,59,0.6); border:1px solid var(--panel-border); color:#cbd5e1; padding:0.4rem 0.75rem; border-radius:6px; font-size:0.75rem; cursor:pointer; transition:0.2s; }
        .preset-btn:hover { border-color:var(--accent); color:white; background:var(--accent-glow); }
        .form-group { margin-bottom:1.1rem; }
        label { display:flex; justify-content:space-between; font-size:0.85rem; font-weight:500; margin-bottom:0.4rem; color:#cbd5e1; }
        input[type="number"], input[type="text"] { width:100%; background:#0b0f19; border:1px solid var(--panel-border); color:white; padding:0.65rem 0.85rem; border-radius:8px; font-size:0.95rem; outline:none; transition:0.2s; }
        input:focus { border-color:var(--accent); box-shadow:0 0 10px var(--accent-glow); }
        .two-col { display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; }
        .derived { background:#0b0f19; border:1px dashed var(--panel-border); border-radius:8px; padding:0.7rem 0.85rem; font-size:0.8rem; color:#cbd5e1; margin-bottom:1.1rem; }
        .btn-submit { width:100%; background:linear-gradient(135deg,#2563eb,#7c3aed); border:none; color:white; padding:0.85rem; border-radius:10px; font-size:1rem; font-weight:600; cursor:pointer; transition:0.2s; box-shadow:0 4px 15px var(--accent-glow); margin-top:0.5rem; }
        .btn-submit:hover { opacity:0.95; transform:translateY(-1px); }
        .score-box { text-align:center; padding:1.5rem 0; }
        .meter-circle { width:130px; height:130px; border-radius:50%; background:radial-gradient(closest-side,var(--panel) 79%,transparent 80% 100%),conic-gradient(var(--accent) calc(var(--score-pct) * 1%),var(--panel-border) 0); margin:0 auto 1rem auto; display:flex; align-items:center; justify-content:center; box-shadow:0 0 20px var(--accent-glow); transition:0.5s ease-out; }
        .score-val { font-size:1.8rem; font-weight:700; }
        .score-lbl { font-size:0.75rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px; }
        .badge-list { display:flex; flex-direction:column; gap:0.85rem; }
        .res-row { background:#0b0f19; border:1px solid var(--panel-border); border-radius:10px; padding:0.85rem 1rem; display:flex; justify-content:space-between; align-items:center; }
        .res-lbl { font-size:0.8rem; color:var(--text-muted); }
        .res-val { font-size:0.95rem; font-weight:600; }
        .tag { padding:0.35rem 0.75rem; border-radius:6px; font-size:0.8rem; font-weight:600; text-transform:uppercase; }
        .tag-ctr_fixable { background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid rgba(245,158,11,0.4); }
        .tag-genuine_decline { background:rgba(239,68,68,0.2); color:#f87171; border:1px solid rgba(239,68,68,0.4); }
        .tag-likely_serp_answered { background:rgba(139,92,246,0.2); color:#c084fc; border:1px solid rgba(139,92,246,0.4); }
        .tag-stable_or_improving { background:rgba(16,185,129,0.2); color:#34d399; border:1px solid rgba(16,185,129,0.4); }
        .code-box { margin-top:1.5rem; background:#070a12; border:1px solid var(--panel-border); border-radius:10px; padding:1rem; font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:#a5b4fc; overflow-x:auto; }
        .model-line { margin-top:1.25rem; font-size:0.78rem; color:var(--text-muted); font-family:'JetBrains Mono',monospace; }
        .note { font-size:0.75rem; color:var(--text-muted); margin-top:0.35rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo-box">
                <div class="logo-icon">🚀</div>
                <div>
                    <div class="title-h1">FlyRank AI SEO Triage Engine</div>
                    <div class="subtitle">Live Random Forest serving endpoint &amp; MCP agent node</div>
                </div>
            </div>
            <div class="nav-links">
                <a href="/docs" target="_blank" class="btn-nav">⚡ OpenAPI Docs</a>
                <a href="/version" target="_blank" class="btn-nav">🔎 Model fingerprint</a>
                <a href="https://simon-okosodo-ds.github.io/flyrank-ml-internship-starter/" target="_blank" class="btn-nav">📄 Research Paper</a>
                <a href="https://github.com/simon-okosodo-ds/flyrank-triage-api" target="_blank" class="btn-nav">💻 GitHub Repo</a>
            </div>
        </header>

        <div class="main-grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">1. Search Performance Inputs</div>
                    <span style="font-size:0.75rem; color:var(--success);" id="liveBadge">● Checking model…</span>
                </div>

                <div class="presets">
                    <span style="font-size:0.75rem; color:var(--text-muted); align-self:center;">Quick demos:</span>
                    <button type="button" class="preset-btn" onclick="loadPreset('ctr')">⚡ Low CTR, good rank</button>
                    <button type="button" class="preset-btn" onclick="loadPreset('decline')">🔴 Genuine decline</button>
                    <button type="button" class="preset-btn" onclick="loadPreset('serp')">🤖 SERP answered</button>
                </div>

                <form id="triageForm">
                    <div class="form-group">
                        <label>X-API-Key header <span>authentication</span></label>
                        <input type="text" id="api_key" value="flyrank-demo-key" required>
                        <div class="note">Public demo key, pre-filled for instant evaluation.</div>
                    </div>
                    <div class="form-group">
                        <label>Monthly impressions <span>GSC</span></label>
                        <input type="number" id="impressions" value="450" required min="1">
                    </div>
                    <div class="form-group">
                        <label>Monthly clicks <span>GSC</span></label>
                        <input type="number" id="clicks" value="12" required min="0">
                    </div>
                    <div class="form-group">
                        <label>Average SERP position <span>must be &gt; 0</span></label>
                        <input type="number" step="0.1" id="avg_position" value="14.2" required min="0.1">
                    </div>
                    <div class="derived" id="derivedBox">Derived by the server: striking distance (10 &lt; position ≤ 30) and real volume (impressions ≥ 100).</div>

                    <div class="two-col">
                        <div class="form-group">
                            <label>Prior-month impressions <span>optional</span></label>
                            <input type="number" id="impressions_prior" min="0" placeholder="e.g. 500">
                        </div>
                        <div class="form-group">
                            <label>Prior-month clicks <span>optional</span></label>
                            <input type="number" id="clicks_prior" min="0" placeholder="e.g. 25">
                        </div>
                    </div>
                    <div class="note">Prior-month values unlock the “genuine decline” and “likely SERP-answered” diagnoses.</div>

                    <button type="submit" class="btn-submit">⚡ Execute AI Triage Scoring</button>
                </form>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">2. Live Model Output &amp; Action</div>
                    <span style="font-size:0.75rem; color:var(--text-muted);">Real-time inference</span>
                </div>

                <div class="score-box">
                    <div class="meter-circle" id="meter" style="--score-pct: 0;">
                        <div>
                            <div class="score-val" id="scoreVal">—</div>
                            <div class="score-lbl">Improvement score</div>
                        </div>
                    </div>
                </div>

                <div class="badge-list">
                    <div class="res-row"><span class="res-lbl">Diagnostic state</span><span class="tag" id="diagTag">—</span></div>
                    <div class="res-row"><span class="res-lbl">Recommended action</span><span class="res-val" id="actionVal" style="color:#60a5fa;">—</span></div>
                    <div class="res-row"><span class="res-lbl">Calculated CTR</span><span class="res-val" id="ctrVal">—</span></div>
                </div>

                <div class="code-box" id="codeSnippet">Run a score to see the matching curl command.</div>
                <div class="model-line" id="modelLine">model: loading…</div>
            </div>
        </div>
    </div>

    <script>
        const PRESETS = {
            ctr:     { impressions: 450,  clicks: 12, avg_position: 14.2, impressions_prior: '',  clicks_prior: '' },
            decline: { impressions: 120,  clicks: 2,  avg_position: 28.5, impressions_prior: 300, clicks_prior: 10 },
            serp:    { impressions: 1500, clicks: 10, avg_position: 3.1,  impressions_prior: 1200, clicks_prior: 40 }
        };

        function updateDerived() {
            const pos = parseFloat(document.getElementById('avg_position').value);
            const imp = parseFloat(document.getElementById('impressions').value);
            const striking = pos > 10 && pos <= 30;
            const volume = imp >= 100;
            document.getElementById('derivedBox').innerText =
                'Derived by the server: striking distance = ' + (striking ? 'yes' : 'no') +
                ' (10 < position ≤ 30), real volume = ' + (volume ? 'yes' : 'no') + ' (impressions ≥ 100).';
        }
        ['impressions', 'avg_position'].forEach(id => document.getElementById(id).addEventListener('input', updateDerived));

        function loadPreset(name) {
            const p = PRESETS[name];
            Object.keys(p).forEach(k => { document.getElementById(k).value = p[k]; });
            updateDerived();
            document.getElementById('triageForm').dispatchEvent(new Event('submit'));
        }

        function buildBody() {
            const body = {
                impressions: parseFloat(document.getElementById('impressions').value),
                clicks: parseFloat(document.getElementById('clicks').value),
                avg_position: parseFloat(document.getElementById('avg_position').value)
            };
            const ip = document.getElementById('impressions_prior').value;
            const cp = document.getElementById('clicks_prior').value;
            if (ip !== '' && cp !== '') { body.impressions_prior = parseFloat(ip); body.clicks_prior = parseFloat(cp); }
            return body;
        }

        document.getElementById('triageForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const apiKey = document.getElementById('api_key').value.trim() || 'flyrank-demo-key';
            const body = buildBody();
            try {
                const res = await fetch('/score', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
                    body: JSON.stringify(body)
                });
                const data = await res.json();
                if (res.ok) {
                    document.getElementById('meter').style.setProperty('--score-pct', Math.round(data.model_score * 100));
                    document.getElementById('scoreVal').innerText = data.model_score.toFixed(3);
                    const diagTag = document.getElementById('diagTag');
                    diagTag.innerText = data.diagnosis;
                    diagTag.className = 'tag tag-' + data.diagnosis;
                    document.getElementById('actionVal').innerText = data.action;
                    document.getElementById('ctrVal').innerText = (data.details.ctr * 100).toFixed(2) + '%';
                    document.getElementById('codeSnippet').innerText =
                        'curl -X POST "' + location.origin + '/score" -H "Content-Type: application/json" -H "X-API-Key: ' +
                        apiKey + '" -d \\'' + JSON.stringify(body) + '\\'';
                } else {
                    alert('API error (' + res.status + '): ' + (data.detail || 'invalid request'));
                }
            } catch (err) {
                console.error(err);
                alert('Error connecting to the triage endpoint.');
            }
        });

        fetch('/version').then(r => r.json()).then(v => {
            const ok = v.model_loaded && v.fingerprint_matches_expected;
            document.getElementById('liveBadge').innerText = v.model_loaded ? '● Model live' : '● Model NOT loaded';
            document.getElementById('modelLine').innerText = v.model_loaded
                ? 'model: ' + v.model_type + ' · sha ' + v.model_sha256_12 + (ok ? ' · matches the published paper ✓' : ' · differs from the published paper')
                : 'model: not loaded (' + v.load_error + ')';
        }).catch(() => { document.getElementById('modelLine').innerText = 'model: status unavailable'; });
        updateDerived();
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse, summary="Interactive web dashboard")
def serve_dashboard():
    return DASHBOARD_HTML

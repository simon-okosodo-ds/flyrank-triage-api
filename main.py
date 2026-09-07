import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="FlyRank SEO Triage API",
    description=(
        "Live model serving endpoint for FlyRank's trained Random Forest classifier. "
        "Transforms raw Search Console & traffic metrics into actionable SEO content triage decisions."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web dashboards or agent frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "Flyrank.model.pkl")

# Global model variable loaded on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"Successfully loaded model from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")

class PageInput(BaseModel):
    impressions: float = Field(..., ge=0, description="Monthly GSC impressions count")
    clicks: float = Field(..., ge=0, description="Monthly GSC clicks count")
    avg_position: float = Field(..., ge=0, description="Average SERP ranking position (must be > 0)")
    in_striking_distance: int = Field(..., ge=0, le=1, description="Binary flag: position between 11 and 30")
    has_real_volume: int = Field(..., ge=0, le=1, description="Binary flag: impressions >= 100")
    
    # Optional prior period metrics for period-over-period delta diagnosis
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
    """Computes diagnostic classification based on metrics and optional prior period deltas."""
    if imp_prior is not None and clk_prior is not None:
        imp_chg = impressions - imp_prior
        clk_chg = clicks - clk_prior
    else:
        # Default snapshot heuristics when prior data is not provided
        imp_chg = 0.0
        clk_chg = 0.0

    if imp_prior is not None and clk_prior is not None:
        if imp_chg < 0 and clk_chg < 0:
            return "genuine_decline"
        if imp_chg >= 0 and clk_chg < 0:
            return "likely_serp_answered"

    if ctr < 0.3 and 0 < avg_position <= 20:
        return "ctr_fixable"

    return "stable_or_improving"

@app.get("/", summary="Health and Service Info")
def health_check():
    return {
        "status": "live",
        "service": "FlyRank Random Forest SEO Triage API",
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model else "Not Loaded",
        "features": [
            "impressions",
            "clicks",
            "avg_position",
            "in_striking_distance",
            "has_real_volume"
        ],
        "outputs": [
            "model_score",
            "diagnosis",
            "action"
        ],
        "docs_url": "/docs"
    }

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

    # Feature DataFrame strictly matching training features
    features_df = pd.DataFrame([{
        "impressions": page.impressions,
        "clicks": page.clicks,
        "avg_position": page.avg_position,
        "in_striking_distance": page.in_striking_distance,
        "has_real_volume": page.has_real_volume
    }])

    # Predict improvement probability score
    try:
        proba = model.predict_proba(features_df)[:, 1][0]
        model_score = round(float(proba), 3)
    except Exception as e:
        # Fallback for array inputs without feature names
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
    results = []
    for page in pages:
        results.append(score_page(page))
    return results

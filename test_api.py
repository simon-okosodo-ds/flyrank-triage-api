"""
Automated Test Suite for FlyRank Triage API & Model Serving
"""

import os
import joblib
import pandas as pd
from fastapi import HTTPException
import main
from main import app, load_model, score_page, health_check, score_batch, PageInput, MODEL_PATH

def test_model_file_exists():
    """Verify that model file is present."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at {MODEL_PATH}"
    print("[OK] Model file presence test passed.")

def test_model_loading_and_prediction():
    """Verify model unpickling and direct prediction."""
    model = joblib.load(MODEL_PATH)
    sample_df = pd.DataFrame([{
        "impressions": 500.0,
        "clicks": 15.0,
        "avg_position": 12.5,
        "in_striking_distance": 1,
        "has_real_volume": 1
    }])
    proba = model.predict_proba(sample_df)[:, 1][0]
    assert 0.0 <= proba <= 1.0, f"Probability out of range: {proba}"
    print(f"[OK] Direct model test passed. Sample prediction proba: {proba:.4f}")

def test_health_endpoint():
    """Verify GET / health check function."""
    res = health_check()
    assert res["status"] == "live"
    assert res["model_loaded"] is True
    print("[OK] Health check endpoint test passed.")

def test_score_endpoint_valid():
    """Verify scoring logic with valid input schema."""
    page = PageInput(
        impressions=450.0,
        clicks=12.0,
        avg_position=14.2,
        in_striking_distance=1,
        has_real_volume=1
    )
    res = score_page(page)
    assert res.model_score >= 0.0
    assert res.diagnosis == "ctr_fixable"
    assert res.action == "review_title_and_meta"
    print(f"[OK] Valid scoring endpoint test passed. Score: {res.model_score}, Diagnosis: {res.diagnosis}, Action: {res.action}")

def test_score_endpoint_prior_period_decline():
    """Verify diagnosis with prior period decline."""
    page = PageInput(
        impressions=300.0,
        clicks=10.0,
        avg_position=15.0,
        in_striking_distance=1,
        has_real_volume=1,
        impressions_prior=500.0,
        clicks_prior=25.0
    )
    res = score_page(page)
    assert res.diagnosis == "genuine_decline"
    assert res.action == "refresh_or_rewrite"
    print(f"[OK] Genuine decline test passed. Diagnosis: {res.diagnosis}, Action: {res.action}")

def test_score_endpoint_zero_position_rejection():
    """Verify HTTP 422 error on invalid position=0."""
    page = PageInput(
        impressions=100.0,
        clicks=5.0,
        avg_position=0.0,
        in_striking_distance=0,
        has_real_volume=1
    )
    try:
        score_page(page)
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 422
        print("[OK] Zero position rejection test passed.")

def test_score_endpoint_zero_impressions_rejection():
    """Verify HTTP 422 error on zero impressions."""
    page = PageInput(
        impressions=0.0,
        clicks=0.0,
        avg_position=10.0,
        in_striking_distance=0,
        has_real_volume=0
    )
    try:
        score_page(page)
        assert False, "Should have raised HTTPException"
    except HTTPException as e:
        assert e.status_code == 422
        print("[OK] Zero impressions rejection test passed.")

def test_batch_scoring():
    """Verify batch scoring endpoint."""
    pages = [
        PageInput(impressions=400.0, clicks=10.0, avg_position=12.0, in_striking_distance=1, has_real_volume=1),
        PageInput(impressions=800.0, clicks=50.0, avg_position=5.0, in_striking_distance=0, has_real_volume=1)
    ]
    res_list = score_batch(pages)
    assert len(res_list) == 2
    print(f"[OK] Batch scoring test passed. Evaluated {len(res_list)} pages.")

def test_mcp_standalone_cli():
    """Verify MCP standalone execution module."""
    from mcp_server import score_page as mcp_score, PageInput as MCPInput
    inp = MCPInput(
        impressions=600.0,
        clicks=30.0,
        avg_position=8.5,
        in_striking_distance=0,
        has_real_volume=1
    )
    res = mcp_score(inp)
    assert res.model_score >= 0.0
    print(f"[OK] MCP standalone agent test passed. Score: {res.model_score}, Diagnosis: {res.diagnosis}")

if __name__ == "__main__":
    print("--- Running FlyRank Triage API Automated Test Suite ---")
    load_model()
    test_model_file_exists()
    test_model_loading_and_prediction()
    test_health_endpoint()
    test_score_endpoint_valid()
    test_score_endpoint_prior_period_decline()
    test_score_endpoint_zero_position_rejection()
    test_score_endpoint_zero_impressions_rejection()
    test_batch_scoring()
    test_mcp_standalone_cli()
    print("\nALL AUTOMATED TESTS PASSED SUCCESSFULLY!")

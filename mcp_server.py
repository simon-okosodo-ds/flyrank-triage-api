"""
FlyRank Model Context Protocol (MCP) Server
Exposes the FlyRank Random Forest SEO Triage model as an MCP Tool for AI Agents.
"""

import sys
import json
import asyncio
from typing import Dict, Any

try:
    from mcp.server.fastmcp import FastMCP
    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False

# Import triage scoring logic directly from main FastAPI app
from main import load_model, score_page, PageInput

# Initialize model
load_model()

if HAS_FASTMCP:
    mcp = FastMCP("FlyRank SEO Triage Agent Tool")

    @mcp.tool(
        name="flyrank_triage_page",
        description=(
            "Evaluates SEO performance of a web page using FlyRank's Random Forest classifier. "
            "Inputs GSC impressions, clicks, avg_position, in_striking_distance (0/1), and has_real_volume (0/1). "
            "Returns model_score, performance diagnosis, and actionable content decision."
        )
    )
    def flyrank_triage_page(
        impressions: float,
        clicks: float,
        avg_position: float,
        in_striking_distance: int = 0,
        has_real_volume: int = 0,
        impressions_prior: float = None,
        clicks_prior: float = None
    ) -> Dict[str, Any]:
        input_data = PageInput(
            impressions=impressions,
            clicks=clicks,
            avg_position=avg_position,
            in_striking_distance=in_striking_distance,
            has_real_volume=has_real_volume,
            impressions_prior=impressions_prior,
            clicks_prior=clicks_prior
        )
        res = score_page(input_data)
        return res.dict()

def run_standalone_agent_cli():
    """Fallback CLI mode for testing MCP / agent execution directly via stdin/stdout."""
    print(json.dumps({
        "status": "ready",
        "tool_name": "flyrank_triage_page",
        "description": "FlyRank SEO Triage Agent Tool active."
    }))
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        sample_input = PageInput(
            impressions=320.0,
            clicks=8.0,
            avg_position=15.2,
            in_striking_distance=1,
            has_real_volume=1
        )
        res = score_page(sample_input)
        print("Sample Execution Result:")
        print(json.dumps(res.dict(), indent=2))

if __name__ == "__main__":
    if HAS_FASTMCP and "--cli" not in sys.argv:
        mcp.run()
    else:
        run_standalone_agent_cli()

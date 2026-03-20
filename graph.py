"""
LangGraph workflow — wires the agent team into a state machine.

Pipeline flow:
  START → intake → direct → art_direct → cinematograph → generate →
  supervise → [conditional: retry or finalize] → END

Retry loop: on quality failure, returns to cinematographer to adjust
prompts based on the Supervisor's guidance, then re-generates.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from agents.art_director import art_director_node
from agents.cinematographer import cinematographer_node
from agents.director import director_node
from agents.editor import editor_node
from agents.supervisor import supervisor_node
from config import settings
from states import AdBrief, AdForgeState

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility Nodes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def intake_node(state: AdForgeState) -> dict:
    """
    Validate the brief and set up the output directory.

    This is the entry point — it ensures the brief is well-formed
    and creates the workspace for all downstream artifacts.
    """
    brief = AdBrief(**state["brief"])  # Validates via Pydantic

    # Create a unique output directory for this run
    run_id = uuid4().hex[:8]
    output_dir = (
        Path(settings.pipeline.output_base_dir)
        / f"{brief.product_name.lower().replace(' ', '_')}_{run_id}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("📋 Intake validated brief for '%s'", brief.product_name)
    logger.info("   Output directory: %s", output_dir)

    return {
        "output_dir": str(output_dir),
        "retry_count": 0,
        "max_retries": settings.pipeline.max_retries,
        "errors": [],
        "generated_clips": [],
        "reference_images": [],
        "quality_passed": False,
        "status": "intake_complete",
    }


def finalize_node(state: AdForgeState) -> dict:
    """
    Finalize the ad — copy the cumulative video to the final output path.

    The latest_video_path from the Editor (after all extends) IS the
    complete ad. We just copy it to a clean output name.
    """
    output_dir = Path(state["output_dir"])
    latest = state.get("latest_video_path", "")

    if latest and Path(latest).exists():
        final_path = output_dir / "final_ad.mp4"
        shutil.copy2(latest, final_path)
        logger.info("🎉 Final ad exported → %s", final_path)
        return {
            "final_video_path": str(final_path),
            "status": "complete",
        }

    logger.error("❌ No video available for final export")
    return {
        "final_video_path": "",
        "status": "failed",
        "errors": state.get("errors", []) + ["No video available for export"],
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Retry Routing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def increment_retry(state: AdForgeState) -> dict:
    """Bump the retry counter before re-entering the generation loop."""
    new_count = state.get("retry_count", 0) + 1
    logger.info("🔄 Retry %d/%d", new_count, state.get("max_retries", 2))
    return {"retry_count": new_count}


def should_retry(state: AdForgeState) -> str:
    """
    Conditional edge after Supervisor:
      - "retry"    → quality failed AND retries remaining
      - "finalize" → quality passed OR retries exhausted
    """
    passed = state.get("quality_passed", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.pipeline.max_retries)

    if passed:
        logger.info("✅ Quality gate passed — proceeding to finalize")
        return "finalize"

    if retry_count < max_retries:
        logger.info(
            "⚠️  Quality gate failed — retrying (%d/%d)",
            retry_count + 1, max_retries,
        )
        return "retry"

    logger.warning(
        "⚠️  Quality gate failed but retries exhausted (%d/%d) — "
        "accepting current output",
        retry_count, max_retries,
    )
    return "finalize"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Graph Assembly
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_graph() -> StateGraph:
    """
    Construct and return the compiled AdForge LangGraph.

    Graph topology:

        START → intake → direct → art_direct → cinematograph → generate
                                                    ↑                ↓
                                              increment_retry   supervise
                                                    ↑                ↓
                                                    └── retry ◄──condition
                                                                    ↓
                                                                finalize → END
    """
    graph = StateGraph(AdForgeState)

    # ── Add nodes ───────────────────────────────────────────────────────
    graph.add_node("intake", intake_node)
    graph.add_node("direct", director_node)
    graph.add_node("art_direct", art_director_node)
    graph.add_node("cinematograph", cinematographer_node)
    graph.add_node("generate", editor_node)
    graph.add_node("supervise", supervisor_node)
    graph.add_node("increment_retry", increment_retry)
    graph.add_node("finalize", finalize_node)

    # ── Add edges (happy path) ──────────────────────────────────────────
    graph.add_edge(START, "intake")
    graph.add_edge("intake", "direct")
    graph.add_edge("direct", "art_direct")
    graph.add_edge("art_direct", "cinematograph")
    graph.add_edge("cinematograph", "generate")
    graph.add_edge("generate", "supervise")

    # ── Conditional: retry or finalize ──────────────────────────────────
    graph.add_conditional_edges(
        "supervise",
        should_retry,
        {
            "retry": "increment_retry",
            "finalize": "finalize",
        },
    )

    # Retry loop: increment counter → re-prompt → re-generate → re-review
    graph.add_edge("increment_retry", "cinematograph")

    # Terminal
    graph.add_edge("finalize", END)

    return graph.compile()

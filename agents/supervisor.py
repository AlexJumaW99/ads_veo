"""
Supervisor Agent — quality assurance via Gemini vision analysis.

Extracts keyframes from the generated video, sends them to Gemini
alongside the Scene Bible, and produces a structured QualityReport.

Uses `with_structured_output(QualityReport)` for deterministic scoring.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from prompts import SUPERVISOR_HUMAN, SUPERVISOR_SYSTEM
from services.gemini_client import extract_keyframes
from states import AdForgeState, QualityReport, SceneBible, VeoPrompt

logger = logging.getLogger(__name__)


def _encode_image_b64(path: Path) -> str:
    """Read an image file and return base64 string."""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def supervisor_node(state: AdForgeState) -> dict:
    """
    LangGraph node: review the generated video for quality.

    Workflow:
      1. Extract keyframes from the latest cumulative video
      2. Send frames + Scene Bible to Gemini 2.5 Flash for analysis
      3. Return QualityReport with pass/fail and retry guidance

    Reads:  state["latest_video_path"], state["scene_bible"], state["veo_prompts"]
    Writes: state["quality_report"], state["quality_passed"]
    """
    video_path = state.get("latest_video_path", "")
    if not video_path or not Path(video_path).exists():
        logger.error("❌ No video file to review")
        fail_report = QualityReport(
            overall_pass=False,
            overall_score=0,
            criteria=[],
            issues=["No video file found for review"],
            retry_guidance="Regenerate all shots — no video was produced.",
        )
        return {
            "quality_report": fail_report.model_dump(),
            "quality_passed": False,
            "status": "review_failed",
        }

    scene_bible = SceneBible(**state["scene_bible"])
    veo_prompts = [VeoPrompt(**p) for p in state.get("veo_prompts", [])]
    output_dir = Path(state["output_dir"]) / "qa_frames"

    # ── Step 1: Extract keyframes ───────────────────────────────────────
    logger.info("🔍 Supervisor extracting keyframes from %s", Path(video_path).name)

    num_frames = min(8, max(4, len(veo_prompts) * 2))
    frames = extract_keyframes(video_path, output_dir, num_frames=num_frames)

    if not frames:
        logger.error("❌ Could not extract frames from video")
        fail_report = QualityReport(
            overall_pass=False,
            overall_score=0,
            criteria=[],
            issues=["Frame extraction failed — video may be corrupted"],
            retry_guidance="Regenerate all shots — video file appears corrupt.",
        )
        return {
            "quality_report": fail_report.model_dump(),
            "quality_passed": False,
            "status": "review_failed",
        }

    # ── Step 2: Build multimodal message with frames ────────────────────
    logger.info("🔍 Supervisor reviewing %d keyframes", len(frames))

    llm = ChatGoogleGenerativeAI(
        model=settings.models.reviewer_llm,
        temperature=settings.pipeline.reviewer_temperature,
        google_api_key=settings.api_key,
    )
    structured_llm = llm.with_structured_output(QualityReport)

    # Build content parts: text + images
    system_msg = SUPERVISOR_SYSTEM.format(
        quality_threshold=settings.pipeline.quality_threshold,
    )

    human_text = SUPERVISOR_HUMAN.format(
        scene_bible_json=json.dumps(scene_bible.model_dump(), indent=2),
        veo_prompts_json=json.dumps(
            [p.model_dump() for p in veo_prompts], indent=2,
        ),
    )

    # Construct multimodal content: text + inline images
    content_parts: list[dict] = [
        {"type": "text", "text": human_text},
    ]
    for frame_path in frames:
        b64 = _encode_image_b64(frame_path)
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })

    result: QualityReport = structured_llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=content_parts),
    ])

    # ── Step 3: Log and return ──────────────────────────────────────────
    passed = result.overall_pass and result.overall_score >= settings.pipeline.quality_threshold

    if passed:
        logger.info(
            "✅ Supervisor PASSED — score %d/10", result.overall_score,
        )
    else:
        logger.warning(
            "⚠️  Supervisor FAILED — score %d/10 | Issues: %s",
            result.overall_score,
            "; ".join(result.issues) if result.issues else "none specified",
        )
        if result.retry_guidance:
            logger.info("   Retry guidance: %s", result.retry_guidance)

    return {
        "quality_report": result.model_dump(),
        "quality_passed": passed,
        "status": "reviewed",
    }

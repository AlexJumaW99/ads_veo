"""
Cinematographer Agent — transforms the Scene Bible + Shot Plan into
production-ready Veo 3.1 prompts.

Each prompt follows Google's five-part formula:
  [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]

Uses `with_structured_output(CinematographerOutput)` for schema enforcement.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from prompts import CINEMATOGRAPHER_HUMAN, CINEMATOGRAPHER_SYSTEM
from states import (
    AdBrief,
    AdForgeState,
    CinematographerOutput,
    QualityReport,
    SceneBible,
    ShotPlan,
)

logger = logging.getLogger(__name__)


def cinematographer_node(state: AdForgeState) -> dict:
    """
    LangGraph node: generate Veo 3.1 prompts for every shot.

    On retry (retry_count > 0), incorporates the Supervisor's
    retry_guidance into the prompt for targeted improvements.

    Reads:  state["brief"], state["scene_bible"], state["shot_plan"],
            state["quality_report"] (on retry)
    Writes: state["veo_prompts"]
    """
    brief = AdBrief(**state["brief"])
    scene_bible = SceneBible(**state["scene_bible"])
    shots = [ShotPlan(**s) for s in state["shot_plan"]]

    retry_count = state.get("retry_count", 0)

    logger.info(
        "🎥 Cinematographer building %d Veo prompts%s",
        len(shots),
        f" (retry #{retry_count})" if retry_count > 0 else "",
    )

    # Build LLM with structured output
    llm = ChatGoogleGenerativeAI(
        model=settings.models.planner_llm,
        temperature=settings.pipeline.planner_temperature,
        google_api_key=settings.api_key,
    )
    structured_llm = llm.with_structured_output(CinematographerOutput)

    # Incorporate retry guidance if available
    retry_section = ""
    if retry_count > 0 and state.get("quality_report"):
        qr = QualityReport(**state["quality_report"])
        if qr.retry_guidance:
            retry_section = (
                f"\n## RETRY GUIDANCE (from QA review — apply these fixes)\n"
                f"{qr.retry_guidance}\n\n"
                f"Issues found: {'; '.join(qr.issues)}\n"
            )

    human = CINEMATOGRAPHER_HUMAN.format(
        scene_bible_json=json.dumps(scene_bible.model_dump(), indent=2),
        shot_plan_json=json.dumps([s.model_dump() for s in shots], indent=2),
        product_name=brief.product_name,
        product_description=brief.product_description,
        retry_guidance_section=retry_section,
    )

    result: CinematographerOutput = structured_llm.invoke([
        SystemMessage(content=CINEMATOGRAPHER_SYSTEM),
        HumanMessage(content=human),
    ])

    # Validate prompt count
    if len(result.prompts) != len(shots):
        logger.warning(
            "Cinematographer returned %d prompts for %d shots — adjusting",
            len(result.prompts), len(shots),
        )
        result.prompts = result.prompts[:len(shots)]

    for p in result.prompts:
        logger.info(
            "  Shot %d: %s", p.shot_number, p.full_prompt[:100] + "…",
        )

    logger.info("✅ Cinematographer complete — %d prompts ready", len(result.prompts))

    return {
        "veo_prompts": [p.model_dump() for p in result.prompts],
        "status": "prompted",
    }

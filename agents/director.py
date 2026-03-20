"""
Director Agent — the creative brain of the pipeline.

Takes an AdBrief and produces:
  - A SceneBible (visual consistency contract)
  - A ShotPlan (ordered sequence of shots)

Uses `with_structured_output(DirectorOutput)` to guarantee valid schema.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from prompts import DIRECTOR_HUMAN, DIRECTOR_SYSTEM
from states import AdBrief, AdForgeState, DirectorOutput

logger = logging.getLogger(__name__)


def director_node(state: AdForgeState) -> dict:
    """
    LangGraph node: plan the ad's creative structure.

    Reads:  state["brief"]
    Writes: state["scene_bible"], state["shot_plan"], state["creative_rationale"]
    """
    brief = AdBrief(**state["brief"])

    logger.info("🎬 Director planning %d-shot ad for '%s'", brief.num_clips, brief.product_name)

    # Build the LLM with structured output
    llm = ChatGoogleGenerativeAI(
        model=settings.models.planner_llm,
        temperature=settings.pipeline.planner_temperature,
        google_api_key=settings.api_key,
    )
    structured_llm = llm.with_structured_output(DirectorOutput)

    # Format user-provided shot list (if any)
    user_shots_section = ""
    if brief.user_shot_list:
        formatted = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(brief.user_shot_list))
        user_shots_section = (
            f"\n**User-provided shot list** (use as the basis for your plan, "
            f"refine but do not ignore):\n{formatted}\n"
        )

    # Assemble messages
    system = DIRECTOR_SYSTEM.format(num_clips=brief.num_clips)
    human = DIRECTOR_HUMAN.format(
        product_name=brief.product_name,
        product_description=brief.product_description,
        target_audience=brief.target_audience,
        brand_tone=brief.brand.tone,
        brand_colors=", ".join(brief.brand.colors),
        key_message=brief.key_message,
        cta_text=brief.cta_text,
        num_clips=brief.num_clips,
        aspect_ratio=brief.aspect_ratio.value,
        user_shot_list_section=user_shots_section,
    )

    result: DirectorOutput = structured_llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=human),
    ])

    # Validate shot count matches requested clips
    if len(result.shots) != brief.num_clips:
        logger.warning(
            "Director produced %d shots but %d requested — truncating/padding",
            len(result.shots), brief.num_clips,
        )
        result.shots = result.shots[:brief.num_clips]

    logger.info("✅ Director complete — %d shots planned", len(result.shots))
    logger.info("   Rationale: %s", result.creative_rationale[:120])

    return {
        "scene_bible": result.scene_bible.model_dump(),
        "shot_plan": [s.model_dump() for s in result.shots],
        "creative_rationale": result.creative_rationale,
        "status": "directed",
    }

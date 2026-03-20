"""
Art Director Agent — generates reference images for Veo 3.1 identity anchoring.

If the user provides reference images, those are used directly.
Otherwise, this agent generates product reference images using
Gemini 2.5 Flash Image to serve as Veo's visual anchors.

Veo 3.1 accepts up to 3 reference images per generation call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from prompts import ART_DIRECTOR_IMAGE_PROMPT
from services.gemini_client import generate_reference_image
from states import AdBrief, AdForgeState, ReferenceImage, SceneBible

logger = logging.getLogger(__name__)

# Angles for generated reference images (if user doesn't provide any)
_ANGLES = ["front-facing", "three-quarter angle", "side profile"]


def art_director_node(state: AdForgeState) -> dict:
    """
    LangGraph node: prepare reference images for Veo 3.1.

    Strategy:
      - If user provided images → validate and use them (up to 3).
      - If not enough user images → generate additional ones with Gemini.

    Reads:  state["brief"], state["scene_bible"]
    Writes: state["reference_images"]
    """
    brief = AdBrief(**state["brief"])
    scene_bible = SceneBible(**state["scene_bible"])
    output_dir = Path(state["output_dir"]) / "reference_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_refs = settings.pipeline.max_reference_images
    references: list[dict] = []

    # ── Step 1: Use user-provided images ────────────────────────────────
    for i, img_path in enumerate(brief.reference_image_paths[:max_refs]):
        p = Path(img_path)
        if p.exists():
            references.append(
                ReferenceImage(
                    path=str(p.resolve()),
                    source="user_provided",
                    description=f"User reference image {i+1}",
                ).model_dump()
            )
            logger.info("📷 Using user reference: %s", p.name)
        else:
            logger.warning("⚠️  Reference image not found: %s", img_path)

    # ── Step 2: Generate missing references ─────────────────────────────
    num_to_generate = max_refs - len(references)

    if num_to_generate > 0:
        logger.info(
            "🎨 Art Director generating %d reference image(s)", num_to_generate,
        )
        for i in range(num_to_generate):
            angle = _ANGLES[i % len(_ANGLES)]
            prompt = ART_DIRECTOR_IMAGE_PROMPT.format(
                product_name=brief.product_name,
                product_description=brief.product_description,
                angle=angle,
                color_palette=scene_bible.color_palette,
                lighting_setup=scene_bible.lighting_setup,
            )
            out_path = output_dir / f"ref_{angle.replace(' ', '_')}_{i:02d}.png"

            try:
                saved = generate_reference_image(
                    prompt=prompt,
                    output_path=out_path,
                    aspect_ratio=brief.aspect_ratio.value,
                )
                references.append(
                    ReferenceImage(
                        path=str(saved),
                        source="generated",
                        description=f"AI-generated {angle} reference",
                    ).model_dump()
                )
            except Exception as e:
                logger.error("Failed to generate reference image: %s", e)

    logger.info(
        "✅ Art Director complete — %d reference images ready", len(references),
    )

    return {
        "reference_images": references,
        "status": "art_directed",
    }

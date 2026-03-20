"""
Editor Agent — generates the actual video clips using Veo 3.1.

Strategy (Veo Extend chain):
  1. Shot 1: `generate_with_references()` — anchors identity with ref images
  2. Shots 2+: `extend_video()` — extends from previous clip's final second,
     guided by the new shot's prompt

Each extend call produces a cumulative video (original + extension).
The final video after all extends IS the complete ad.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import settings
from services.veo_client import (
    extend_video,
    generate_plain,
    generate_with_references,
)
from states import (
    AdBrief,
    AdForgeState,
    GeneratedClip,
    ReferenceImage,
    VeoPrompt,
)

logger = logging.getLogger(__name__)


def editor_node(state: AdForgeState) -> dict:
    """
    LangGraph node: generate all video clips sequentially.

    The Veo Extend chain works as follows:
      - Shot 1: generated with reference images → 8s clip
      - Shot 2: extend shot 1 output → cumulative ~15-16s video
      - Shot 3: extend shot 2 output → cumulative ~22-24s video
      - ...

    The final cumulative video is the complete ad.

    Reads:  state["brief"], state["veo_prompts"], state["reference_images"]
    Writes: state["generated_clips"], state["latest_video_path"]
    """
    brief = AdBrief(**state["brief"])
    prompts = [VeoPrompt(**p) for p in state["veo_prompts"]]
    ref_images = [ReferenceImage(**r) for r in state.get("reference_images", [])]
    output_dir = Path(state["output_dir"]) / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)

    ref_paths = [r.path for r in ref_images]
    clips: list[dict] = []
    video_obj = None  # Tracks the latest Veo video object for extend chain

    logger.info(
        "🎬 Editor starting %d-shot generation (Veo Extend chain)", len(prompts)
    )

    for i, prompt in enumerate(prompts):
        shot_num = prompt.shot_number
        clip_path = output_dir / f"shot_{shot_num:02d}.mp4"

        try:
            if i == 0:
                # ── First shot: use reference images ────────────────────
                if ref_paths:
                    logger.info("  Shot %d → generating with %d references", shot_num, len(ref_paths))
                    saved_path, video_obj = generate_with_references(
                        prompt=prompt.full_prompt,
                        reference_image_paths=ref_paths,
                        output_path=clip_path,
                        aspect_ratio=brief.aspect_ratio.value,
                        negative_prompt=prompt.negative_prompt,
                    )
                else:
                    logger.info("  Shot %d → generating (no references)", shot_num)
                    saved_path, video_obj = generate_plain(
                        prompt=prompt.full_prompt,
                        output_path=clip_path,
                        aspect_ratio=brief.aspect_ratio.value,
                        negative_prompt=prompt.negative_prompt,
                    )

                clips.append(
                    GeneratedClip(
                        shot_number=shot_num,
                        video_path=str(saved_path),
                        duration_seconds=settings.pipeline.clip_duration,
                        generation_mode="reference_images" if ref_paths else "text_only",
                        prompt_used=prompt.full_prompt,
                        is_cumulative=False,
                    ).model_dump()
                )

            else:
                # ── Subsequent shots: Veo Extend ────────────────────────
                if video_obj is not None:
                    logger.info("  Shot %d → extending from previous clip", shot_num)
                    saved_path, video_obj = extend_video(
                        prompt=prompt.full_prompt,
                        video_to_extend=video_obj,
                        output_path=clip_path,
                    )
                    clips.append(
                        GeneratedClip(
                            shot_number=shot_num,
                            video_path=str(saved_path),
                            duration_seconds=settings.pipeline.extend_duration,
                            generation_mode="extend",
                            prompt_used=prompt.full_prompt,
                            is_cumulative=True,
                        ).model_dump()
                    )
                else:
                    # Fallback: extend chain broke, generate independently
                    logger.warning(
                        "  Shot %d → extend chain broken, generating independently",
                        shot_num,
                    )
                    saved_path, video_obj = generate_plain(
                        prompt=prompt.full_prompt,
                        output_path=clip_path,
                        aspect_ratio=brief.aspect_ratio.value,
                        negative_prompt=prompt.negative_prompt,
                    )
                    clips.append(
                        GeneratedClip(
                            shot_number=shot_num,
                            video_path=str(saved_path),
                            duration_seconds=settings.pipeline.clip_duration,
                            generation_mode="text_only_fallback",
                            prompt_used=prompt.full_prompt,
                            is_cumulative=False,
                        ).model_dump()
                    )

        except Exception as e:
            logger.error("  ❌ Shot %d failed: %s", shot_num, e)
            # Record the failure but continue with remaining shots
            clips.append(
                GeneratedClip(
                    shot_number=shot_num,
                    video_path="",
                    duration_seconds=0,
                    generation_mode="failed",
                    prompt_used=prompt.full_prompt,
                    is_cumulative=False,
                ).model_dump()
            )
            # Break the extend chain — subsequent shots fall back to plain
            video_obj = None

    # The latest successfully generated video is the cumulative output
    latest_path = ""
    for c in reversed(clips):
        if c["video_path"]:
            latest_path = c["video_path"]
            break

    logger.info(
        "✅ Editor complete — %d/%d clips generated",
        sum(1 for c in clips if c["video_path"]),
        len(prompts),
    )

    return {
        "generated_clips": clips,
        "latest_video_path": latest_path,
        "status": "generated",
    }

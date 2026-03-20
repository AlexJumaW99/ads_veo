"""
Wrapper around google-genai SDK for Veo 3.1 video generation.

Handles three generation modes:
  1. Text-to-video with reference images  (first clip)
  2. Veo Extend                            (subsequent clips)
  3. Text-to-video plain                   (fallback)

All operations are long-running — this module polls until completion
or timeout, then downloads and saves the result.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.api_key)


def _poll_operation(
    client: genai.Client,
    operation: Any,
    label: str = "video",
) -> Any:
    """
    Poll a long-running Veo operation until done or timeout.

    Returns the completed operation.
    Raises TimeoutError if poll_timeout_seconds is exceeded.
    """
    cfg = settings.pipeline
    elapsed = 0

    while not operation.done:
        if elapsed >= cfg.poll_timeout_seconds:
            raise TimeoutError(
                f"Veo generation timed out after {cfg.poll_timeout_seconds}s "
                f"for {label}"
            )
        logger.info(
            "  ⏳ Waiting for %s… (%ds elapsed)", label, elapsed
        )
        time.sleep(cfg.poll_interval_seconds)
        elapsed += cfg.poll_interval_seconds
        operation = client.operations.get(operation)

    logger.info("  ✅ %s generation complete (%ds)", label, elapsed)
    return operation


def _save_video(
    client: genai.Client,
    operation: Any,
    output_path: Path,
) -> tuple[Path, Any]:
    """
    Download the first generated video from a completed operation.

    Returns (saved_path, video_object) where video_object can be
    passed to Veo Extend for the next clip.
    """
    generated_video = operation.response.generated_videos[0]

    # Download the video bytes into the video object
    client.files.download(file=generated_video.video)

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated_video.video.save(str(output_path))
    logger.info("  💾 Video saved → %s", output_path)

    return output_path, generated_video.video


def _load_reference_images(
    image_paths: list[str | Path],
    max_images: int = 3,
) -> list[Image.Image]:
    """Load reference images as PIL Images (Veo accepts up to 3)."""
    loaded = []
    for p in image_paths[:max_images]:
        img = Image.open(str(p))
        loaded.append(img)
        logger.info("  📷 Loaded reference image: %s", Path(p).name)
    return loaded


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_with_references(
    prompt: str,
    reference_image_paths: list[str | Path],
    output_path: str | Path,
    aspect_ratio: str = "16:9",
    negative_prompt: str = "",
) -> tuple[Path, Any]:
    """
    Generate the FIRST clip using reference images for identity anchoring.

    This is used for shot 1 only. Subsequent shots use `extend_video`.
    Veo 3.1 defaults to 8-second clips.

    Args:
        prompt: Full Veo 3.1 prompt.
        reference_image_paths: 1-3 product/brand reference images.
        output_path: Where to save the MP4.
        aspect_ratio: "16:9" or "9:16".
        negative_prompt: Content to avoid.

    Returns:
        (saved_path, video_object) for chaining into extend calls.
    """
    client = _get_client()
    output_path = Path(output_path)
    ref_images = _load_reference_images(reference_image_paths)

    logger.info("🎬 Generating shot 1 with %d reference images", len(ref_images))

    config_kwargs: dict = {
        "reference_images": ref_images,
        "aspect_ratio": aspect_ratio,
        "person_generation": "allow_adult",
    }
    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt

    operation = client.models.generate_videos(
        model=settings.models.video_gen,
        prompt=prompt,
        config=types.GenerateVideosConfig(**config_kwargs),
    )

    operation = _poll_operation(client, operation, label="shot-1")
    return _save_video(client, operation, output_path)


def extend_video(
    prompt: str,
    video_to_extend: Any,
    output_path: str | Path,
) -> tuple[Path, Any]:
    """
    Extend an existing Veo-generated video with a new scene.

    Veo Extend appends ~7-8s of new footage that continues visually
    from the final second of the input video.

    Args:
        prompt: Prompt describing what happens NEXT.
        video_to_extend: Video object from a previous generate/extend call.
        output_path: Where to save the extended MP4.

    Returns:
        (saved_path, video_object) for further chaining.
    """
    client = _get_client()
    output_path = Path(output_path)

    logger.info("🎬 Extending video with new scene → %s", output_path.name)

    operation = client.models.generate_videos(
        model=settings.models.video_gen,
        prompt=prompt,
        video=video_to_extend,
    )

    operation = _poll_operation(client, operation, label=output_path.stem)
    return _save_video(client, operation, output_path)


def generate_plain(
    prompt: str,
    output_path: str | Path,
    aspect_ratio: str = "16:9",
    negative_prompt: str = "",
) -> tuple[Path, Any]:
    """
    Fallback: generate a clip from text only (no references, no extend).
    Veo 3.1 defaults to 8-second clips.

    Used when reference images are unavailable or as a retry strategy.
    """
    client = _get_client()
    output_path = Path(output_path)

    logger.info("🎬 Generating plain text-to-video → %s", output_path.name)

    config_kwargs: dict = {
        "aspect_ratio": aspect_ratio,
        "person_generation": "allow_adult",
    }
    if negative_prompt:
        config_kwargs["negative_prompt"] = negative_prompt

    operation = client.models.generate_videos(
        model=settings.models.video_gen,
        prompt=prompt,
        config=types.GenerateVideosConfig(**config_kwargs),
    )

    operation = _poll_operation(client, operation, label=output_path.stem)
    return _save_video(client, operation, output_path)

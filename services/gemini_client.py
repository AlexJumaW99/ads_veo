"""
Wrapper around google-genai SDK for image generation and vision analysis.

Uses:
  - gemini-2.5-flash-image  → reference image generation (Art Director)
  - gemini-2.5-flash         → frame-level video QA (Supervisor)
"""

from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from config import settings

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    """Create a google-genai client (reads GOOGLE_API_KEY from env)."""
    return genai.Client(api_key=settings.api_key)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Image Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_reference_image(
    prompt: str,
    output_path: str | Path,
    aspect_ratio: str = "16:9",
) -> Path:
    """
    Generate a single reference image using Gemini 2.5 Flash Image.

    Args:
        prompt: Descriptive prompt for the image.
        output_path: Where to save the PNG.
        aspect_ratio: "16:9" or "9:16".

    Returns:
        Path to the saved image file.
    """
    client = _get_client()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating reference image → %s", output_path.name)

    response = client.models.generate_content(
        model=settings.models.image_gen,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    # Extract and save the image
    for part in response.parts:
        if part.inline_data:
            image = part.as_image()
            image.save(str(output_path))
            logger.info("Reference image saved → %s", output_path)
            return output_path

    raise RuntimeError(f"No image returned for prompt: {prompt[:80]}…")


def generate_reference_image_with_context(
    prompt: str,
    output_path: str | Path,
    aspect_ratio: str = "16:9",
    context_image_paths: list[str | Path] | None = None,
) -> Path:
    """
    Generate a reference image with optional visual context from prior images.

    When context images are provided, they are sent as inline images alongside
    an explicit consistency instruction, forcing Gemini to match the product's
    appearance (shape, proportions, colors, materials) across angles.

    When no context images exist, behaves identically to generate_reference_image().

    Args:
        prompt: Descriptive prompt for the new image.
        output_path: Where to save the PNG.
        aspect_ratio: "16:9" or "9:16".
        context_image_paths: Paths to previously generated reference images.

    Returns:
        Path to the saved image file.
    """
    # No context — fall back to the standard path
    if not context_image_paths:
        return generate_reference_image(prompt, output_path, aspect_ratio)

    client = _get_client()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating reference image with %d context image(s) → %s",
        len(context_image_paths),
        output_path.name,
    )

    # Build multimodal contents: prior images + consistency instruction + prompt
    contents: list = []

    consistency_text = (
        "Here are previously generated reference images of this exact product. "
        "Your new image MUST depict the identical product — same shape, "
        "proportions, colors, materials, and finish. Only the camera angle changes."
    )
    contents.append(consistency_text)

    for img_path in context_image_paths:
        img = Image.open(str(img_path))
        contents.append(img)

    contents.append(prompt)

    response = client.models.generate_content(
        model=settings.models.image_gen,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
            ),
        ),
    )

    # Extract and save the image
    for part in response.parts:
        if part.inline_data:
            image = part.as_image()
            image.save(str(output_path))
            logger.info("Reference image saved → %s", output_path)
            return output_path

    raise RuntimeError(f"No image returned for prompt: {prompt[:80]}…")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Frame Extraction (for Supervisor quality checks)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path,
    num_frames: int = 4,
) -> list[Path]:
    """
    Extract evenly-spaced keyframes from a video file using OpenCV.

    Returns list of paths to extracted JPEG frames.
    """
    import cv2

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        num_frames = max(1, total_frames)

    # Evenly space frame indices
    indices = [
        int(i * (total_frames - 1) / (num_frames - 1)) if num_frames > 1 else 0
        for i in range(num_frames)
    ]

    saved: list[Path] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            out = output_dir / f"frame_{idx:05d}.jpg"
            cv2.imwrite(str(out), frame)
            saved.append(out)

    cap.release()
    logger.info("Extracted %d keyframes from %s", len(saved), video_path.name)
    return saved

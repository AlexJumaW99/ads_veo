"""
Data models and LangGraph state definitions for AdForge.

Pydantic models serve dual purpose:
  1. Structured-output schemas for LLM calls via `with_structured_output()`
  2. Type-safe data containers passed through the pipeline

LangGraph state uses TypedDict with dict-serialised Pydantic models so that
every field is JSON-serialisable and checkpoint-friendly.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Enums
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AspectRatio(str, Enum):
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"


class ShotType(str, Enum):
    """Narrative role of a shot within the ad sequence."""
    HOOK = "hook"
    SHOWCASE = "showcase"
    LIFESTYLE = "lifestyle"
    DETAIL = "detail"
    CTA = "cta"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Input Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BrandGuidelines(BaseModel):
    """Brand visual and tonal identity."""
    tone: str = Field(
        description="Brand voice (e.g. 'Bold, energetic, aspirational')"
    )
    colors: list[str] = Field(
        description="Hex colours, primary first (e.g. ['#000000', '#FF0000'])"
    )
    visual_style: str = Field(
        default="",
        description="Free-form visual style notes"
    )


class AdBrief(BaseModel):
    """
    Complete input brief that the user provides.
    Everything the pipeline needs to generate an ad.
    """
    product_name: str = Field(description="Product being advertised")
    product_description: str = Field(description="What the product is / does")
    target_audience: str = Field(description="Demographic & psychographic profile")
    brand: BrandGuidelines
    key_message: str = Field(description="Core tagline or message")
    cta_text: str = Field(description="Call-to-action text shown/spoken at end")
    reference_image_paths: list[str] = Field(
        default_factory=list,
        description="Paths to user-supplied product / brand images"
    )
    num_clips: int = Field(
        default=4, ge=1, le=8,
        description="Number of ~8s clips to generate"
    )
    aspect_ratio: AspectRatio = Field(default=AspectRatio.LANDSCAPE)
    resolution: str = Field(default="1080p")
    user_shot_list: Optional[list[str]] = Field(
        default=None,
        description="Optional user-provided shot descriptions (overrides auto-plan)"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LLM Structured-Output Schemas
#  (used with `llm.with_structured_output(Model)`)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Director output ─────────────────────────────────────────────────────────

class SceneBible(BaseModel):
    """
    Visual consistency contract.
    Every token here is reused verbatim in every Veo prompt to guarantee
    cross-shot coherence.
    """
    identity_anchors: list[str] = Field(
        description=(
            "Visual identity elements maintained across ALL shots. "
            "E.g. 'product centred in frame', 'red accent rim lighting', "
            "'logo always visible on upper-left of product'."
        )
    )
    lens_vocabulary: str = Field(
        description=(
            "Camera language reused across all shots. "
            "E.g. '35mm equivalent, shallow depth of field, smooth stabilised movements'."
        )
    )
    color_palette: str = Field(
        description=(
            "Colour-grading description applied to every shot. "
            "E.g. 'Warm golden-hour tones, teal shadows, lifted blacks, high contrast'."
        )
    )
    lighting_setup: str = Field(
        description=(
            "Lighting approach consistent across shots. "
            "E.g. 'Soft diffused key light from camera-left, hard rim light on product edges'."
        )
    )
    audio_direction: str = Field(
        description=(
            "Sound design direction. "
            "E.g. 'Upbeat lo-fi beat, whoosh transitions between shots, subtle bass on CTA'."
        )
    )
    negative_constraints: list[str] = Field(
        description=(
            "Explicit prohibitions applied to ALL shots. "
            "E.g. 'No on-screen text', 'No weather changes', 'No people'."
        )
    )


class ShotPlan(BaseModel):
    """Single shot in the ad sequence."""
    shot_number: int = Field(description="1-indexed position in the sequence")
    shot_type: ShotType = Field(description="Narrative role of this shot")
    description: str = Field(
        description="2-3 sentence description of what happens in this shot"
    )
    camera_movement: str = Field(
        description="Specific camera motion (e.g. 'Slow dolly push-in, medium to close-up')"
    )
    subject_action: str = Field(
        description="What the product / subject does during this shot"
    )
    audio_notes: str = Field(
        description="Shot-specific audio cues (e.g. 'Bass drop on reveal, crowd ambient')"
    )


class DirectorOutput(BaseModel):
    """
    Full Director agent output.
    Used as: `planner_llm.with_structured_output(DirectorOutput)`
    """
    scene_bible: SceneBible
    shots: list[ShotPlan] = Field(
        description="Ordered shot list comprising the full ad"
    )
    creative_rationale: str = Field(
        description="Brief (2-3 sentence) explanation of the creative strategy"
    )


# ── Cinematographer output ──────────────────────────────────────────────────

class VeoPrompt(BaseModel):
    """
    Structured Veo 3.1 prompt following Google's five-part formula:
        [Cinematography] + [Subject] + [Action] + [Context] + [Style & Ambiance]
    """
    shot_number: int = Field(description="Matches ShotPlan.shot_number")
    cinematography: str = Field(
        description="Camera work and composition (e.g. 'Slow arc shot, low angle, 35mm')"
    )
    subject: str = Field(
        description="Main focal element (e.g. 'A pair of white Nike Air Max 90 sneakers')"
    )
    action: str = Field(
        description="What the subject is doing (e.g. 'rotating on a glossy pedestal')"
    )
    context: str = Field(
        description="Environment and background (e.g. 'in a neon-lit studio with smoke wisps')"
    )
    style_ambiance: str = Field(
        description="Aesthetic and mood (e.g. 'Cinematic, moody, teal-and-orange colour grade')"
    )
    audio_prompt: str = Field(
        description="Audio / dialogue direction for Veo's native audio generation"
    )
    negative_prompt: str = Field(
        description="What to exclude (e.g. 'No text overlays, no people, no lens flare')"
    )
    full_prompt: str = Field(
        description=(
            "The complete, assembled prompt string ready for the Veo 3.1 API. "
            "Must incorporate ALL fields above into a single coherent paragraph."
        )
    )


class CinematographerOutput(BaseModel):
    """
    Full Cinematographer agent output.
    Used as: `planner_llm.with_structured_output(CinematographerOutput)`
    """
    prompts: list[VeoPrompt] = Field(
        description="One VeoPrompt per shot, in sequence order"
    )


# ── Supervisor output ───────────────────────────────────────────────────────

class QualityCriterion(BaseModel):
    """Single axis of quality evaluation."""
    criterion: str = Field(
        description="What was checked (e.g. 'identity_consistency', 'prompt_adherence')"
    )
    score: int = Field(
        ge=1, le=10,
        description="Score from 1 (worst) to 10 (perfect)"
    )
    notes: str = Field(description="Brief observation for this criterion")


class QualityReport(BaseModel):
    """
    Quality assessment for generated video.
    Used as: `reviewer_llm.with_structured_output(QualityReport)`
    """
    overall_pass: bool = Field(description="True if the video meets the quality bar")
    overall_score: int = Field(
        ge=0, le=10,
        description="Aggregate quality score (0 = no video to review)"
    )
    criteria: list[QualityCriterion] = Field(
        description="Per-axis assessments"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="Specific problems found (empty if clean)"
    )
    retry_guidance: Optional[str] = Field(
        default=None,
        description=(
            "If failed: concrete prompt-adjustment instructions for the "
            "Cinematographer on the next attempt"
        )
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Internal Tracking Models (not LLM outputs)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ReferenceImage(BaseModel):
    """Metadata for a reference image (user-supplied or AI-generated)."""
    path: str
    source: str                 # "user_provided" | "generated"
    description: str


class GeneratedClip(BaseModel):
    """Metadata for a video clip produced by Veo 3.1."""
    shot_number: int
    video_path: str
    duration_seconds: float
    generation_mode: str        # "reference_images" | "extend"
    prompt_used: str
    is_cumulative: bool = Field(
        default=False,
        description="True when this file includes all prior shots (Veo Extend output)"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LangGraph State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdForgeState(TypedDict, total=False):
    """
    Central state flowing through the LangGraph pipeline.

    Convention:
      - Complex objects are stored as `dict` (via Pydantic `.model_dump()`).
      - Each node returns a partial dict updating only the keys it owns.
      - `total=False` allows progressive population across nodes.
    """

    # ── Inputs (set by intake node) ─────────────────────────────────────
    brief: dict                                     # AdBrief
    output_dir: str                                 # Resolved output path

    # ── Planning (set by Director) ──────────────────────────────────────
    scene_bible: dict                               # SceneBible
    shot_plan: list[dict]                           # list[ShotPlan]
    creative_rationale: str

    # ── Reference images (set by Art Director) ──────────────────────────
    reference_images: list[dict]                    # list[ReferenceImage]

    # ── Prompts (set by Cinematographer) ────────────────────────────────
    veo_prompts: list[dict]                         # list[VeoPrompt]

    # ── Generated video (set by Editor) ─────────────────────────────────
    generated_clips: list[dict]                     # list[GeneratedClip]
    latest_video_path: str                          # most recent cumulative video

    # ── Quality (set by Supervisor) ─────────────────────────────────────
    quality_report: dict                            # QualityReport
    quality_passed: bool

    # ── Control flow ────────────────────────────────────────────────────
    final_video_path: str
    retry_count: int
    max_retries: int
    errors: list[str]
    status: str                                     # human-readable stage name

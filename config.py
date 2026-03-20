"""
Central configuration for AdForge pipeline.

All model IDs, defaults, and tunable parameters live here.
Set GOOGLE_API_KEY (or GEMINI_API_KEY) in your environment before running.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Google model identifiers – update here when new versions ship."""

    # LLM for planning / reasoning (Director, Cinematographer)
    planner_llm: str = "gemini-2.5-pro"

    # LLM for quality assessment (Supervisor) – fast + vision
    reviewer_llm: str = "gemini-2.5-flash"

    # Image generation (reference frames, storyboard images)
    image_gen: str = "gemini-2.5-flash-image"

    # Video generation
    video_gen: str = "veo-3.1-generate-preview"
    video_gen_fast: str = "veo-3.1-fast-generate-preview"


@dataclass(frozen=True)
class PipelineConfig:
    """Tunable pipeline behaviour."""

    # -- Video generation ------------------------------------------------
    clip_duration: int = 8                 # seconds per Veo generation
    extend_duration: int = 8               # seconds added per extend call
    max_reference_images: int = 3          # Veo 3.1 limit
    default_aspect_ratio: str = "16:9"
    default_resolution: str = "1080p"

    # -- Quality gate ----------------------------------------------------
    quality_threshold: int = 7             # min overall score to pass (1-10)
    max_retries: int = 2                   # retry cycles before accepting

    # -- Polling ---------------------------------------------------------
    poll_interval_seconds: int = 10        # sleep between operation polls
    poll_timeout_seconds: int = 300        # max wait per generation

    # -- LLM tuning ------------------------------------------------------
    planner_temperature: float = 0.8       # creative planning
    reviewer_temperature: float = 0.2      # deterministic QA

    # -- Paths -----------------------------------------------------------
    output_base_dir: str = "output"


@dataclass
class AdForgeSettings:
    """Resolved runtime settings (combines config + env)."""

    models: ModelConfig = field(default_factory=ModelConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def api_key(self) -> str:
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EnvironmentError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment."
            )
        return key


# Singleton – import this across the project
settings = AdForgeSettings()

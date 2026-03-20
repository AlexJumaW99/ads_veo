# AdForge — Multi-Agent AI Video Ad Generation

A production-grade pipeline that orchestrates a team of specialised AI agents to generate consistent, high-quality product video advertisements using **Google Veo 3.1** and **Gemini 2.5**.

## Architecture

```
                            ┌─────────────────────────────────────────────┐
                            │              AdForge Pipeline               │
                            └─────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────────┐
  │  Intake   │───▶│ Director │───▶│ Art Director │───▶│Cinematographer │
  │ (validate)│    │ (plan)   │    │ (ref images) │    │  (Veo prompts) │
  └──────────┘    └──────────┘    └──────────────┘    └───────┬────────┘
                                                              │
                                                              ▼
  ┌──────────┐    ┌────────────┐                      ┌──────────────┐
  │ Finalize │◀───│ Supervisor │◀─────────────────────│    Editor     │
  │ (export) │    │   (QA)     │                      │ (Veo gen +   │
  └──────────┘    └─────┬──────┘                      │  extend)     │
                        │                             └──────────────┘
                        │ ✗ quality failed
                        │ & retries remain
                        ▼
                  ┌─────────────┐
                  │   Retry     │──▶ back to Cinematographer
                  │ (increment) │
                  └─────────────┘
```

## Agent Team

| Agent | Role | Model | Key Technique |
|-------|------|-------|---------------|
| **Director** | Creative planning — scene bible + shot plan | Gemini 2.5 Pro | `with_structured_output(DirectorOutput)` |
| **Art Director** | Reference image generation for identity anchoring | Gemini 2.5 Flash Image | Up to 3 product reference images |
| **Cinematographer** | Constructs Veo 3.1 prompts from the scene bible | Gemini 2.5 Pro | Google's 5-part prompt formula |
| **Editor** | Sequential video generation via Veo Extend chain | Veo 3.1 | Shot 1: reference images → Shot 2+: extend |
| **Supervisor** | Vision-based quality assessment on extracted keyframes | Gemini 2.5 Flash | Multimodal QA with structured scoring |

## How Consistency Works

The system prevents the visual inconsistencies common in AI video ads through three reinforcing mechanisms:

1. **Scene Bible** — The Director generates a binding contract of identity anchors, lens vocabulary, colour palette, lighting, audio direction, and negative constraints. Every downstream agent copies these tokens verbatim into their outputs.

2. **Reference Images** — The Art Director provides 1-3 product reference images (user-supplied or AI-generated) that Veo 3.1 uses as visual anchors for the first clip, locking the product's appearance.

3. **Veo Extend Chain** — Instead of generating clips independently, shots 2+ are created by *extending* the previous clip. Veo generates new footage from the final second of the prior video, maintaining visual continuity across the entire ad.

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Gemini API key
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

### CLI

```bash
# Run with an example brief
python -m ad_forge --brief ad_forge/examples/nike_air_max.json

# With verbose logging
python -m ad_forge --brief ad_forge/examples/autonomous_desk.json -v
```

### Programmatic

```python
from ad_forge.main import run_ad_pipeline

result = run_ad_pipeline(
    product_name="Sony WH-1000XM5",
    product_description="Premium wireless noise-cancelling headphones",
    target_audience="Audio enthusiasts and frequent travellers, 25-45",
    brand_tone="Sleek, premium, tech-forward",
    brand_colors=["#000000", "#C0A36A", "#FFFFFF"],
    key_message="Silence is golden.",
    cta_text="Experience it at sony.com",
    num_clips=3,
    aspect_ratio="16:9",
    resolution="1080p",
)

print(result["final_video_path"])  # → output/sony_wh_1000xm5_a3f2b1c0/final_ad.mp4
print(result["quality_report"])    # → full QA breakdown
```

### With User-Provided References + Shot List

```python
result = run_ad_pipeline(
    product_name="Aeron Chair",
    product_description="Ergonomic mesh office chair by Herman Miller",
    target_audience="Design-conscious professionals",
    brand_tone="Refined, architectural, design-led",
    brand_colors=["#1A1A1A", "#6B7B8D", "#F5F5F5"],
    key_message="Designed for the way you work.",
    cta_text="Configure yours at hermanmiller.com",
    reference_image_paths=[
        "assets/aeron_front.jpg",
        "assets/aeron_side.jpg",
        "assets/aeron_detail.jpg",
    ],
    user_shot_list=[
        "Extreme close-up of the mesh material, light filtering through",
        "Camera pulls back to reveal the full chair, rotating slowly",
        "Person sits down, chair adjusts, wide shot of modern office",
        "Logo reveal with the chair silhouetted against clean backdrop",
    ],
    num_clips=4,
)
```

## Brief JSON Format

```json
{
  "product_name": "Product Name",
  "product_description": "What it is and does",
  "target_audience": "Who the ad is for",
  "brand": {
    "tone": "Brand voice description",
    "colors": ["#primary", "#secondary", "#accent"],
    "visual_style": "Optional style notes"
  },
  "key_message": "Core tagline",
  "cta_text": "Call to action",
  "reference_image_paths": ["path/to/image1.jpg"],
  "num_clips": 4,
  "aspect_ratio": "16:9",
  "resolution": "1080p",
  "user_shot_list": ["Shot 1 description", "Shot 2 description"]
}
```

## Output Structure

```
output/nike_air_max_90_a3f2b1c0/
├── reference_images/
│   ├── ref_front-facing_00.png
│   ├── ref_three-quarter_angle_01.png
│   └── ref_side_profile_02.png
├── clips/
│   ├── shot_01.mp4          # First clip (8s, reference-anchored)
│   ├── shot_02.mp4          # Cumulative: shot 1 + shot 2 (~16s)
│   ├── shot_03.mp4          # Cumulative: shots 1-3 (~24s)
│   └── shot_04.mp4          # Cumulative: shots 1-4 (~32s)
├── qa_frames/
│   ├── frame_00000.jpg
│   ├── frame_00030.jpg
│   └── ...
├── final_ad.mp4             # ← The complete ad
└── pipeline_state.json      # Full pipeline state for debugging
```

## Configuration

Edit `ad_forge/config.py` to tune behaviour:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `clip_duration` | 8 | Seconds per Veo generation |
| `quality_threshold` | 7 | Min score (1-10) to pass QA |
| `max_retries` | 2 | Retry loops before accepting |
| `planner_temperature` | 0.8 | LLM creativity for planning |
| `reviewer_temperature` | 0.2 | LLM determinism for QA |
| `poll_interval_seconds` | 10 | Veo polling frequency |
| `poll_timeout_seconds` | 300 | Max wait per generation |

## Design Decisions

**Why LangGraph over plain functions?** The retry loop (Supervisor → Cinematographer → Editor → Supervisor) is a conditional cycle that LangGraph handles natively with `add_conditional_edges`. Plain sequential code would require manual loop management and make the state flow harder to inspect/debug.

**Why Veo Extend instead of FFmpeg concat?** Independent clips generated from text alone drift visually — different lighting, colours, product proportions. Veo Extend generates from the final second of the previous clip, producing inherently continuous footage. The trade-off is speed (sequential generation), but quality wins.

**Why Gemini 2.5 Flash for QA instead of Pro?** The Supervisor needs to be fast (it runs on every retry) and deterministic (scoring should be consistent). Flash's vision capabilities are sufficient for frame-level quality checks, and at ~0.2 temperature it produces reliable scores.

**Why structured output everywhere?** Using `with_structured_output()` on every LLM call guarantees the pipeline never breaks on malformed JSON. The Pydantic models in `states.py` enforce field types, ranges (scores 1-10), and required fields at the schema level.

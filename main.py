"""
AdForge — CLI and programmatic entry point.

Usage:
  # CLI
  python -m ad_forge.main --brief brief.json

  # Programmatic
  from main import run_ad_pipeline
  result = run_ad_pipeline(
      product_name="Nike Air Max 90",
      product_description="Classic running shoe...",
      ...
  )
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from graph import build_graph
from states import AdBrief, AdForgeState, AspectRatio, BrandGuidelines

load_dotenv()
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Programmatic API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_ad_pipeline(
    product_name: str,
    product_description: str,
    target_audience: str,
    brand_tone: str,
    brand_colors: list[str],
    key_message: str,
    cta_text: str,
    reference_image_paths: list[str] | None = None,
    num_clips: int = 4,
    aspect_ratio: str = "16:9",
    resolution: str = "1080p",
    visual_style: str = "",
    user_shot_list: list[str] | None = None,
    verbose: bool = False,
) -> dict:
    """
    Run the full AdForge pipeline and return the final state.

    Args:
        product_name:          Name of the product.
        product_description:   What the product is / does.
        target_audience:       Who the ad targets.
        brand_tone:            Brand voice (e.g. "Bold, energetic").
        brand_colors:          Hex colours, primary first.
        key_message:           Core ad message / tagline.
        cta_text:              Call-to-action text.
        reference_image_paths: Paths to product images (up to 3).
        num_clips:             Number of ~8s clips (1-8).
        aspect_ratio:          "16:9" or "9:16".
        resolution:            "720p", "1080p", or "4k".
        visual_style:          Free-form style notes.
        user_shot_list:        Optional pre-defined shot descriptions.
        verbose:               Enable debug logging.

    Returns:
        Final AdForgeState as a dict, including `final_video_path`.
    """
    _setup_logging(verbose)

    # Build the brief
    brief = AdBrief(
        product_name=product_name,
        product_description=product_description,
        target_audience=target_audience,
        brand=BrandGuidelines(
            tone=brand_tone,
            colors=brand_colors,
            visual_style=visual_style,
        ),
        key_message=key_message,
        cta_text=cta_text,
        reference_image_paths=reference_image_paths or [],
        num_clips=num_clips,
        aspect_ratio=AspectRatio(aspect_ratio),
        resolution=resolution,
        user_shot_list=user_shot_list,
    )

    # Log the brief
    console.print(Panel(
        f"[bold]{brief.product_name}[/bold]\n"
        f"{brief.product_description[:100]}…\n\n"
        f"Audience: {brief.target_audience}\n"
        f"Clips: {brief.num_clips} × 8s | {brief.aspect_ratio.value} | {brief.resolution}\n"
        f"Message: {brief.key_message}",
        title="🎬 AdForge — Starting Pipeline",
        border_style="blue",
    ))

    # Build and run the graph
    graph = build_graph()
    initial_state: AdForgeState = {"brief": brief.model_dump()}

    final_state = graph.invoke(initial_state)

    # Report results
    status = final_state.get("status", "unknown")
    final_path = final_state.get("final_video_path", "")

    if status == "complete" and final_path:
        console.print(Panel(
            f"[bold green]✅ Ad generated successfully![/bold green]\n\n"
            f"📁 Output: [link file://{final_path}]{final_path}[/link]\n"
            f"🎬 Clips: {len(final_state.get('generated_clips', []))}\n"
            f"🔄 Retries used: {final_state.get('retry_count', 0)}\n"
            f"⭐ Quality score: {final_state.get('quality_report', {}).get('overall_score', 'N/A')}/10",
            title="🎉 AdForge — Complete",
            border_style="green",
        ))
    else:
        errors = final_state.get("errors", [])
        console.print(Panel(
            f"[bold red]❌ Pipeline finished with status: {status}[/bold red]\n"
            f"Errors: {'; '.join(errors) if errors else 'None recorded'}",
            title="⚠️  AdForge — Issues",
            border_style="red",
        ))

    return final_state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="adforge",
        description="Generate AI video ads using Veo 3.1 with multi-agent orchestration.",
    )
    parser.add_argument(
        "--brief", type=str, required=True,
        help="Path to JSON brief file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def cli_main() -> None:
    """CLI entry point."""
    args = _parse_args()

    brief_path = Path(args.brief)
    if not brief_path.exists():
        console.print(f"[red]Brief file not found: {brief_path}[/red]")
        sys.exit(1)

    with open(brief_path) as f:
        brief_data = json.load(f)

    # Map JSON keys to function args
    result = run_ad_pipeline(
        product_name=brief_data["product_name"],
        product_description=brief_data["product_description"],
        target_audience=brief_data["target_audience"],
        brand_tone=brief_data["brand"]["tone"],
        brand_colors=brief_data["brand"]["colors"],
        key_message=brief_data["key_message"],
        cta_text=brief_data["cta_text"],
        reference_image_paths=brief_data.get("reference_image_paths", []),
        num_clips=brief_data.get("num_clips", 4),
        aspect_ratio=brief_data.get("aspect_ratio", "16:9"),
        resolution=brief_data.get("resolution", "1080p"),
        visual_style=brief_data.get("brand", {}).get("visual_style", ""),
        user_shot_list=brief_data.get("user_shot_list"),
        verbose=args.verbose,
    )

    # Write final state for debugging
    output_dir = result.get("output_dir", "")
    if output_dir:
        state_path = Path(output_dir) / "pipeline_state.json"
        # Remove non-serialisable values before dumping
        serialisable = {
            k: v for k, v in result.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        with open(state_path, "w") as f:
            json.dump(serialisable, f, indent=2, default=str)
        console.print(f"   State saved → {state_path}")


if __name__ == "__main__":
    cli_main()

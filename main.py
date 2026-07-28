"""Command-line entry point for extraction and benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drawing_localizer.evaluate import evaluate_result  # noqa: E402
from drawing_localizer.pipeline import (  # noqa: E402
    GeminiDrawingExtractor,
    run_adaptive,
    run_baseline,
    run_tiled,
)
from drawing_localizer.render import render_localized_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Build a deliberately small CLI for the take-home task."""
    parser = argparse.ArgumentParser(
        description="High-recall Cyrillic text extraction from technical drawings."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Run Gemini extraction.")
    extract.add_argument("--image", type=Path, required=True)
    extract.add_argument(
        "--strategy", choices=["baseline", "adaptive", "tiled"], default="tiled"
    )
    extract.add_argument("--output-dir", type=Path, default=Path("outputs"))
    extract.add_argument("--model", default=None)

    batch = subparsers.add_parser("batch", help="Process every image in a folder.")
    batch.add_argument("--input-dir", type=Path, default=Path("data/input"))
    batch.add_argument(
        "--strategy", choices=["baseline", "adaptive", "tiled"], default="tiled"
    )
    batch.add_argument("--output-dir", type=Path, default=Path("outputs"))
    batch.add_argument("--model", default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one JSON result.")
    evaluate.add_argument("--result", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--threshold", type=int, default=86)

    render = subparsers.add_parser(
        "render", help="Erase Cyrillic text and draw the English replacement."
    )
    render.add_argument("--image", type=Path, required=True)
    render.add_argument("--result", type=Path, required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--font", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    """Dispatch CLI commands."""
    args = parse_args()

    if args.command == "evaluate":
        report = evaluate_result(args.result, args.gold, threshold=args.threshold)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "render":
        stats = render_localized_image(
            image_path=args.image,
            result_path=args.result,
            output_path=args.output,
            font_path=args.font,
        )
        print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
        return

    extractor = GeminiDrawingExtractor(model=args.model)
    image_paths = (
        [args.image]
        if args.command == "extract"
        else sorted(
            path
            for path in args.input_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    )

    for image_path in image_paths:
        if args.strategy == "baseline":
            result = run_baseline(image_path, args.output_dir, extractor)
        elif args.strategy == "adaptive":
            result = run_adaptive(image_path, args.output_dir, extractor)
        else:
            result = run_tiled(image_path, args.output_dir, extractor)
        print(
            f"{image_path.name}: {len(result.regions)} unique regions, "
            f"{result.api_calls} total API calls "
            f"({result.new_api_calls} new, {result.cache_hits} cache hits)"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
dropless.py — Water droplet removal from video footage.

Usage
-----
  python dropless.py INPUT.mp4 OUTPUT.mp4 [options]

Examples
--------
  # Classical method with defaults
  python dropless.py raw.mp4 clean.mp4

  # Classical method with debug overlay and median infill
  python dropless.py raw.mp4 clean.mp4 --method classical --inpaint-method median --debug

  # Wider temporal window, lower threshold, stationarity check enabled
  python dropless.py raw.mp4 clean.mp4 --window 51 --threshold 20 --use-frame-diff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from methods import REGISTRY, get_method, register_all_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dropless",
        description="Remove water droplets from video using various methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Positional arguments ---
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input MP4 file.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Path to write the cleaned output MP4.",
    )

    # --- Method selector ---
    parser.add_argument(
        "--method",
        choices=list(REGISTRY.keys()),
        default="classical",
        help="Droplet-removal method to use. Default: classical",
    )

    # --- Let each method register its own arguments ---
    register_all_args(parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path: Path = args.input
    output_path: Path = args.output

    # Basic validation
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")
    if not input_path.is_file():
        parser.error(f"Input path is not a file: {input_path}")
    if output_path.suffix.lower() != ".mp4":
        print(
            f"Warning: output path does not end in .mp4 ({output_path}). "
            "The file will still be written but may not play in all players.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate and run the chosen method
    MethodClass = get_method(args.method)
    method = MethodClass(args)
    method.process(input_path, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

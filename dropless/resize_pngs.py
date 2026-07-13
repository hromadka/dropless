#!/usr/bin/env python3
"""
resize_pngs.py — Batch-resize PNG images to 1280x704.

Reads every PNG in the source folder and writes a resized copy to
a subfolder named 'resized/' (created automatically).

Usage
-----
  python resize_pngs.py FOLDER [options]

Examples
--------
  python resize_pngs.py frames/
  python resize_pngs.py frames/ --out resized_704 --interp area
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

TARGET_W = 1280
TARGET_H = 704

INTERP_CHOICES = {
    "area":    cv2.INTER_AREA,      # best for downscaling
    "linear":  cv2.INTER_LINEAR,    # good general purpose
    "cubic":   cv2.INTER_CUBIC,     # sharper, slower
    "nearest": cv2.INTER_NEAREST,   # pixel-perfect, no blending
    "lanczos": cv2.INTER_LANCZOS4,  # highest quality, slowest
}


def resize_pngs(
    src_folder: Path,
    out_name: str,
    interp: int,
    verbose: bool,
) -> None:
    pngs = sorted(src_folder.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in: {src_folder}")
        return

    out_folder = src_folder / out_name
    out_folder.mkdir(exist_ok=True)

    print(f"Source : {src_folder}  ({len(pngs)} PNGs)")
    print(f"Output : {out_folder}")
    print(f"Target : {TARGET_W}x{TARGET_H}")
    print()

    ok = skip = err = 0

    for png in pngs:
        out_path = out_folder / png.name

        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [skip] Could not read: {png.name}")
            skip += 1
            continue

        h, w = img.shape[:2]
        if w == TARGET_W and h == TARGET_H:
            if verbose:
                print(f"  [skip] Already {TARGET_W}x{TARGET_H}: {png.name}")
            skip += 1
            # Still copy so the output folder is complete
            cv2.imwrite(str(out_path), img)
            continue

        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=interp)
        cv2.imwrite(str(out_path), resized)

        if verbose:
            print(f"  {png.name}  {w}x{h} -> {TARGET_W}x{TARGET_H}")
        ok += 1

    print(f"\nDone. resized={ok}  already-correct={skip}  errors={err}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="resize_pngs",
        description=f"Batch-resize PNG images to {TARGET_W}x{TARGET_H}.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing source PNG files.",
    )
    parser.add_argument(
        "--out",
        default="resized",
        metavar="NAME",
        help="Name of the output subfolder. Default: resized",
    )
    parser.add_argument(
        "--interp",
        choices=list(INTERP_CHOICES.keys()),
        default="area",
        help="Interpolation method. 'area' is best for downscaling. "
             "Default: area",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print a line for every file processed.",
    )

    args = parser.parse_args(argv)

    folder = args.folder
    if not folder.exists():
        parser.error(f"Folder not found: {folder}")
    if not folder.is_dir():
        parser.error(f"Not a directory: {folder}")

    resize_pngs(folder, args.out, INTERP_CHOICES[args.interp], args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())

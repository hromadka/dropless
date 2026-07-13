#!/usr/bin/env python3
"""
simple-avg-buffer.py -- Sliding-window aligned average over a PNG frame sequence.

A window of N frames slides through the folder one step at a time (or by
--stride steps). For each window position, every frame is feature-matched to
the window's reference frame and warped into its coordinate space via RANSAC
homography. The aligned stack is averaged and written as a single output image
named after the reference frame.

With stride=1 (default) this produces one averaged image per input frame
(excluding the first and last half-window at the edges), creating a smoothed,
noise-reduced sequence that can be re-encoded directly to video.

A rolling buffer keeps only the active window in memory at any time,
so memory use stays constant regardless of folder size.

Usage
-----
  python simple-avg-buffer.py FOLDER [options]

Examples
--------
  # Sliding window of 15 frames, stride 1, output to frames/averaged/
  python simple-avg-buffer.py frames/

  # Larger window, non-overlapping (stride = window)
  python simple-avg-buffer.py frames/ --window 30 --stride 30

  # Every 5th frame as output, custom output folder
  python simple-avg-buffer.py frames/ --stride 5 --out /data/smoothed

  # Weight by alignment confidence, SIFT detector
  python simple-avg-buffer.py frames/ --weight-by-inliers --detector sift

  # Process only frames 50-199
  python simple-avg-buffer.py frames/ --start 50 --end 199
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        desc = kwargs.get("desc", "")
        unit = kwargs.get("unit", "it")
        items = list(iterable)
        n = len(items)
        for i, item in enumerate(items):
            pct = int(100 * (i + 1) / n) if n else 0
            print(f"\r{desc}: {i+1}/{n} ({pct}%) {unit}", end="", flush=True)
            yield item
        print()


# ---------------------------------------------------------------------------
# Feature detector / matcher
# ---------------------------------------------------------------------------

def make_detector(name: str, max_features: int):
    """Return (detector, matcher). Falls back to ORB if SIFT unavailable."""
    if name == "sift":
        try:
            det = cv2.SIFT_create(nfeatures=max_features)
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            mat = cv2.FlannBasedMatcher(index_params, search_params)
            return det, mat
        except AttributeError:
            print("[warn] SIFT unavailable (needs opencv-contrib-python). "
                  "Falling back to ORB.")
    det = cv2.ORB_create(nfeatures=max_features)
    mat = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return det, mat


# ---------------------------------------------------------------------------
# Homography estimation
# ---------------------------------------------------------------------------

def compute_homography(
    src_gray: np.ndarray,
    dst_gray: np.ndarray,
    detector,
    matcher,
    ratio_thresh: float,
    ransac_thresh: float,
    min_inliers: int = 8,
) -> tuple[Optional[np.ndarray], int]:
    """
    Estimate H such that warpPerspective(src, H, size) aligns src to dst.
    Returns (H, n_inliers). H is None if alignment fails.
    """
    kp_s, des_s = detector.detectAndCompute(src_gray, None)
    kp_d, des_d = detector.detectAndCompute(dst_gray, None)

    if des_s is None or des_d is None or len(kp_s) < 4 or len(kp_d) < 4:
        return None, 0

    matches = matcher.knnMatch(des_s, des_d, k=2)
    good = [m for m, n in matches
            if len([m, n]) == 2 and m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        return None, 0

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good])
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in good])

    H, inlier_mask = cv2.findHomography(
        pts_s, pts_d,
        cv2.RANSAC, ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )

    if H is None or inlier_mask is None:
        return None, 0

    n_inliers = int(inlier_mask.sum())
    if n_inliers < min_inliers:
        return None, 0

    det = float(np.linalg.det(H[:2, :2]))
    if det < 0.1 or det > 10.0:
        return None, 0

    return H, n_inliers


# ---------------------------------------------------------------------------
# Rolling buffer
# ---------------------------------------------------------------------------

class RollingBuffer:
    """
    Lazy-loading sliding window over a list of image paths.

    Frames are loaded on demand and evicted as soon as the window
    advances past them, keeping memory use proportional to the window
    size rather than the total folder size.

    Each entry in the buffer is a dict:
        { "idx": int, "path": Path, "bgr": ndarray, "gray": ndarray }
    bgr / gray are None if the file could not be read.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._buf: deque[dict] = deque()
        self._next_idx = 0          # next path index to load into the buffer

    # ------------------------------------------------------------------

    def _load_up_to(self, end_idx: int) -> None:
        """Ensure all frames with index < end_idx have been loaded."""
        while self._next_idx < end_idx and self._next_idx < len(self._paths):
            path = self._paths[self._next_idx]
            bgr = cv2.imread(str(path))
            gray = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    if bgr is not None else None)
            self._buf.append({
                "idx":  self._next_idx,
                "path": path,
                "bgr":  bgr,
                "gray": gray,
            })
            self._next_idx += 1

    def _evict_before(self, start_idx: int) -> None:
        """Drop frames with index < start_idx."""
        while self._buf and self._buf[0]["idx"] < start_idx:
            self._buf.popleft()

    def get_window(self, start_idx: int, end_idx: int) -> list[dict]:
        """
        Return entries for frames [start_idx, end_idx).
        Loads any not-yet-loaded frames and evicts those before start_idx.
        """
        self._load_up_to(end_idx)
        self._evict_before(start_idx)
        return [e for e in self._buf if start_idx <= e["idx"] < end_idx]


# ---------------------------------------------------------------------------
# Per-window averaging
# ---------------------------------------------------------------------------

def average_window(
    entries: list[dict],
    ref_local: int,             # index within entries list
    detector,
    matcher,
    ratio_thresh: float,
    ransac_thresh: float,
    weight_by_inliers: bool,
    include_failed: bool,
    verbose: bool,
) -> tuple[Optional[np.ndarray], int, int]:
    """
    Align all entries to entries[ref_local] and return the weighted average.

    Returns (averaged_image, n_aligned, n_failed).
    averaged_image is None if no frames could be combined.
    """
    ref = entries[ref_local]
    if ref["bgr"] is None:
        return None, 0, len(entries)

    ref_bgr  = ref["bgr"]
    ref_gray = ref["gray"]
    H_img, W_img = ref_bgr.shape[:2]

    accum   = np.zeros((H_img, W_img, 3), dtype=np.float64)
    weights = np.zeros((H_img, W_img, 1), dtype=np.float64)
    n_aligned = n_failed = 0

    for entry in entries:
        if entry["bgr"] is None:
            n_failed += 1
            continue

        if entry["idx"] == ref["idx"]:
            w = 1.0
            warped = ref_bgr
            if verbose:
                print(f"    {entry['path'].name}  [reference]")
        else:
            H, n_inliers = compute_homography(
                entry["gray"], ref_gray,
                detector, matcher,
                ratio_thresh=ratio_thresh,
                ransac_thresh=ransac_thresh,
            )
            if H is None:
                n_failed += 1
                if verbose:
                    print(f"    {entry['path'].name}  [align failed]")
                if include_failed:
                    warped = entry["bgr"]
                    w = 0.1
                else:
                    continue
            else:
                warped = cv2.warpPerspective(
                    entry["bgr"], H, (W_img, H_img),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                w = float(n_inliers) if weight_by_inliers else 1.0
                n_aligned += 1
                if verbose:
                    print(f"    {entry['path'].name}  "
                          f"inliers={n_inliers}  w={w:.0f}")

        accum   += warped.astype(np.float64) * w
        weights += w

    if weights.max() == 0:
        return None, n_aligned, n_failed

    averaged = (accum / weights).clip(0, 255).astype(np.uint8)
    return averaged, n_aligned, n_failed


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    src = Path(args.folder)
    all_pngs = sorted(src.glob("*.png"))

    if not all_pngs:
        print(f"No PNG files found in: {src}")
        sys.exit(1)

    total = len(all_pngs)
    start = max(0, args.start)
    end   = min(total - 1, args.end if args.end is not None else total - 1)

    if start > end:
        print(f"--start {start} > --end {end}. Nothing to do.")
        sys.exit(1)

    pngs = all_pngs[start : end + 1]
    n    = len(pngs)
    win  = min(args.window, n)
    half = win // 2

    out_dir = Path(args.out) if args.out else src / "averaged"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Window positions: reference frame index (within pngs) for each output
    # The reference is always the middle of its window.
    # We slide by --stride each step.
    ref_indices = list(range(half, n - (win - half - 1), args.stride))

    print(f"Source    : {src}  ({total} total, {n} selected)")
    print(f"Window    : {win} frames  half={half}")
    print(f"Stride    : {args.stride}")
    print(f"Outputs   : {len(ref_indices)} images -> {out_dir}")
    print(f"Detector  : {args.detector}  max_features={args.max_features}")
    print(f"Weighting : {'inlier-count' if args.weight_by_inliers else 'uniform'}")
    print()

    detector, matcher = make_detector(args.detector, args.max_features)
    buf = RollingBuffer(pngs)

    total_aligned = total_failed = total_skipped = 0

    for ref_local in tqdm(ref_indices, desc="Windows", unit="win"):
        win_start = ref_local - half
        win_end   = win_start + win          # exclusive

        entries   = buf.get_window(win_start, win_end)
        ref_entry = next((e for e in entries if e["idx"] == ref_local), None)

        if ref_entry is None or ref_entry["bgr"] is None:
            total_skipped += 1
            continue

        ref_pos = next(i for i, e in enumerate(entries)
                       if e["idx"] == ref_local)

        averaged, n_aligned, n_failed = average_window(
            entries, ref_pos,
            detector, matcher,
            ratio_thresh=args.ratio_thresh,
            ransac_thresh=args.ransac_thresh,
            weight_by_inliers=args.weight_by_inliers,
            include_failed=args.include_failed,
            verbose=args.verbose,
        )

        total_aligned += n_aligned
        total_failed  += n_failed

        if averaged is None:
            total_skipped += 1
            continue

        out_path = out_dir / ref_entry["path"].name
        cv2.imwrite(str(out_path), averaged)

    print(f"\nDone.")
    print(f"  Outputs written : {len(ref_indices) - total_skipped}")
    print(f"  Align successes : {total_aligned}")
    print(f"  Align failures  : {total_failed}")
    print(f"  Skipped windows : {total_skipped}")
    print(f"  Output folder   : {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="simple-avg-buffer",
        description="Sliding-window aligned average over a PNG frame sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("folder", type=Path,
                   help="Folder containing sequentially-named PNG frames.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output folder path (absolute or relative). "
                        "Default: <folder>/averaged/")

    p.add_argument("--window", type=int, default=15, metavar="N",
                   help="Number of frames per averaging window. Default: 15")
    p.add_argument("--stride", type=int, default=1, metavar="K",
                   help="How many frames to advance the window each step. "
                        "stride=1 produces one output per input frame (maximum "
                        "overlap). stride=window produces non-overlapping windows. "
                        "Default: 1")

    p.add_argument("--start", type=int, default=0, metavar="N",
                   help="0-based index of the first frame to process. Default: 0")
    p.add_argument("--end", type=int, default=None, metavar="N",
                   help="0-based index of the last frame to process (inclusive). "
                        "Default: last frame in folder")

    g = p.add_argument_group("feature alignment")
    g.add_argument("--detector", choices=["orb", "sift"], default="orb",
                   help="Feature detector. sift requires opencv-contrib-python. "
                        "Default: orb")
    g.add_argument("--max-features", type=int, default=2000, metavar="N",
                   help="Max keypoints per frame. Default: 2000")
    g.add_argument("--ratio-thresh", type=float, default=0.75, metavar="R",
                   help="Lowe ratio-test threshold. Lower = stricter. Default: 0.75")
    g.add_argument("--ransac-thresh", type=float, default=4.0, metavar="T",
                   help="RANSAC reprojection threshold in pixels. Default: 4.0")

    g2 = p.add_argument_group("averaging")
    g2.add_argument("--weight-by-inliers", action="store_true",
                    help="Weight each frame's contribution by its RANSAC inlier "
                         "count. Frames with more confident alignment contribute "
                         "more. Default: uniform weighting")
    g2.add_argument("--include-failed", action="store_true",
                    help="Include frames that failed alignment at weight 0.1 "
                         "rather than skipping them entirely.")

    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print per-frame alignment details for every window.")

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.folder.exists() or not args.folder.is_dir():
        parser.error(f"Folder not found or not a directory: {args.folder}")
    if args.window < 2:
        parser.error("--window must be >= 2")
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

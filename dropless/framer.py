#!/usr/bin/env python3
"""
framer.py -- Feature-aligned multi-frame water droplet removal from PNG sequences.

Core insight
------------
Water droplets on the lens are fixed in IMAGE space. The scene behind them
shifts due to camera motion (pitch/roll/yaw on a boat). By aligning each
neighbouring frame to the reference frame using a feature-based homography,
we bring the scene into registration with the reference. After alignment:

  * Scene pixels are consistent across the aligned stack.
  * Droplet pixels in NEIGHBOUR frames are displaced (the warp moved them),
    so they no longer coincide with the reference droplet positions.

The per-pixel median of the aligned stack therefore converges to a clean
background estimate. Pixels where the reference deviates significantly from
this median are flagged as droplets and replaced.

Pipeline (per reference frame)
-------------------------------
1. Load a sliding window of N frames centred on the reference.
2. Detect and match features (ORB by default; SIFT if opencv-contrib present).
3. Compute RANSAC homography: warp each neighbour into reference image space.
4. Stack warped neighbours; compute per-pixel temporal median -> clean BG.
5. Threshold |reference - median| -> binary droplet mask.
6. Morphological cleanup (close gaps, remove noise, dilate edges).
7. Infill masked pixels from the median BG or cv2.inpaint.
8. Write cleaned frame; optionally write side-by-side debug image.

Usage
-----
  python framer.py FRAMES_FOLDER [options]

Examples
--------
  # Defaults: ORB, window=15, threshold=30, median infill
  python framer.py frames/

  # Wider window, debug panels, Telea inpainting
  python framer.py frames/ --window 25 --infill telea --debug

  # SIFT (higher quality on textureless water/sky, needs opencv-contrib)
  python framer.py frames/ --detector sift --max-features 4000

  # Aggressive settings for heavy spray
  python framer.py frames/ --threshold 20 --min-area 20 --morph-close 11
"""

from __future__ import annotations

import argparse
import sys
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
    """
    Return (detector, matcher) pair.

    ORB uses Hamming BFMatcher; SIFT uses FLANN with kd-tree.
    Falls back to ORB if SIFT is unavailable (needs opencv-contrib).
    """
    if name == "sift":
        try:
            det = cv2.SIFT_create(nfeatures=max_features)
            index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE=1
            search_params = dict(checks=50)
            mat = cv2.FlannBasedMatcher(index_params, search_params)
            return det, mat
        except AttributeError:
            print("[warn] SIFT unavailable (needs opencv-contrib-python). "
                  "Falling back to ORB.")
            name = "orb"

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
) -> Optional[np.ndarray]:
    """
    Estimate H such that warpPerspective(src, H, size) aligns src to dst.

    Returns None when there are insufficient matches or the homography is
    degenerate (e.g. near-identity, extreme scale, or too few RANSAC inliers).
    """
    kp_s, des_s = detector.detectAndCompute(src_gray, None)
    kp_d, des_d = detector.detectAndCompute(dst_gray, None)

    if des_s is None or des_d is None or len(kp_s) < 4 or len(kp_d) < 4:
        return None

    matches = matcher.knnMatch(des_s, des_d, k=2)
    good = [m for m, n in matches if len([m, n]) == 2 and
            m.distance < ratio_thresh * n.distance]

    if len(good) < min_inliers:
        return None

    pts_s = np.float32([kp_s[m.queryIdx].pt for m in good])
    pts_d = np.float32([kp_d[m.trainIdx].pt for m in good])

    H, inlier_mask = cv2.findHomography(
        pts_s, pts_d,
        cv2.RANSAC, ransac_thresh,
        maxIters=2000,
        confidence=0.995,
    )

    if H is None:
        return None
    if inlier_mask is None or int(inlier_mask.sum()) < min_inliers:
        return None

    # Reject wildly degenerate homographies by checking the determinant.
    # A valid planar homography should have det(H[0:2,0:2]) > 0 and
    # the scale factor should stay within a reasonable range.
    det = float(np.linalg.det(H[:2, :2]))
    if det < 0.1 or det > 10.0:
        return None

    return H


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_droplet_mask(
    ref: np.ndarray,
    median_bg: np.ndarray,
    threshold: float,
    min_area: int,
    morph_close: int,
    morph_dilate: int,
) -> np.ndarray:
    """
    Return uint8 binary mask (255 = droplet, 0 = clean) for ref.

    Steps:
      1. Per-pixel absolute deviation from the aligned temporal median.
      2. Threshold to obtain initial binary mask.
      3. Morphological closing to fill intra-droplet gaps.
      4. Connected-component filtering to discard tiny specks.
      5. Dilation to cover droplet halos before infill.
    """
    diff = cv2.absdiff(ref, median_bg)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 0:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        clean = np.zeros_like(mask)
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == lbl] = 255
        mask = clean

    if morph_dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (morph_dilate, morph_dilate))
        mask = cv2.dilate(mask, k)

    return mask


# ---------------------------------------------------------------------------
# Infill
# ---------------------------------------------------------------------------

def apply_infill(
    ref: np.ndarray,
    mask: np.ndarray,
    median_bg: np.ndarray,
    method: str,
    radius: int,
) -> np.ndarray:
    """
    Replace masked pixels in ref.

    median  -- direct substitution from the aligned background model.
               Fast and usually clean when the BG model is reliable.
    telea   -- Fast Marching Method (cv2.INPAINT_TELEA). Better for
               large or isolated droplets with no clean BG reference.
    ns      -- Navier-Stokes (cv2.INPAINT_NS). Smoother but slower.
    """
    if mask.max() == 0:
        return ref
    if method == "median":
        out = ref.copy()
        out[mask > 0] = median_bg[mask > 0]
        return out
    elif method == "telea":
        return cv2.inpaint(ref, mask, radius, cv2.INPAINT_TELEA)
    else:  # ns
        return cv2.inpaint(ref, mask, radius, cv2.INPAINT_NS)


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def make_debug_image(
    original: np.ndarray,
    mask: np.ndarray,
    result: np.ndarray,
    n_aligned: int,
    n_failed: int,
) -> np.ndarray:
    """Three-panel image: original | mask overlay | cleaned result."""
    h = original.shape[0]
    overlay = original.copy()
    overlay[mask > 0] = (0, 0, 220)
    panel_mask = cv2.addWeighted(original, 0.45, overlay, 0.55, 0)
    n_px = int(mask.sum()) // 255
    label_mask = f"detected: {n_px} px"
    label_res = f"aligned {n_aligned} | failed {n_failed}"
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    cv2.putText(panel_mask, label_mask, (8, 28), font, scale, (255,255,255), thick)
    cv2.putText(result,     label_res,  (8, 28), font, scale, (0, 220, 0),   thick)
    sep = np.zeros((h, 3, 3), dtype=np.uint8)
    return np.hstack([original, sep, panel_mask, sep, result])


# ---------------------------------------------------------------------------
# Frame cache (memory-bounded sliding buffer)
# ---------------------------------------------------------------------------

class FrameCache:
    """
    Lazy-loading frame cache. Frames are loaded on first access and evicted
    once they are no longer needed by the sliding window.
    """

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths
        self._cache: dict[int, Optional[np.ndarray]] = {}

    def get(self, idx: int) -> Optional[np.ndarray]:
        if idx not in self._cache:
            self._cache[idx] = cv2.imread(str(self._paths[idx]))
        return self._cache[idx]

    def evict_before(self, idx: int) -> None:
        """Remove all frames with index < idx from memory."""
        for k in [k for k in self._cache if k < idx]:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def process_folder(args: argparse.Namespace) -> None:
    src = Path(args.folder)
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in: {src}")
        return

    out_dir = src / args.out
    out_dir.mkdir(exist_ok=True)
    dbg_dir = (src / (args.out + "_debug")) if args.debug else None
    if dbg_dir:
        dbg_dir.mkdir(exist_ok=True)

    half = args.window // 2
    n = len(pngs)

    print(f"Source   : {src}  ({n} PNGs)")
    print(f"Output   : {out_dir}")
    print(f"Window   : {args.window} frames  (half={half})")
    print(f"Detector : {args.detector}  max_features={args.max_features}")
    print(f"Threshold: {args.threshold}  min_area={args.min_area}")
    print(f"Infill   : {args.infill}")
    if args.debug:
        print(f"Debug    : {dbg_dir}")
    print()

    detector, matcher = make_detector(args.detector, args.max_features)
    cache = FrameCache(pngs)

    stats_aligned = stats_failed = stats_skipped = 0

    for ref_idx in tqdm(range(n), desc="Frames", unit="frame"):
        ref_bgr = cache.get(ref_idx)
        if ref_bgr is None:
            print(f"\n[skip] unreadable: {pngs[ref_idx].name}")
            stats_skipped += 1
            continue

        H_ref, W_ref = ref_bgr.shape[:2]
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

        # Collect and align neighbours
        ctx_start = max(0, ref_idx - half)
        ctx_end   = min(n, ref_idx + half + 1)
        neighbor_idxs = [i for i in range(ctx_start, ctx_end) if i != ref_idx]

        aligned: list[np.ndarray] = []
        n_failed_this = 0

        for ni in neighbor_idxs:
            nbr = cache.get(ni)
            if nbr is None:
                n_failed_this += 1
                continue

            nbr_gray = cv2.cvtColor(nbr, cv2.COLOR_BGR2GRAY)
            H = compute_homography(
                nbr_gray, ref_gray,
                detector, matcher,
                ratio_thresh=args.ratio_thresh,
                ransac_thresh=args.ransac_thresh,
            )

            if H is not None:
                warped = cv2.warpPerspective(
                    nbr, H, (W_ref, H_ref),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                aligned.append(warped)
                stats_aligned += 1
            else:
                n_failed_this += 1
                stats_failed += 1
                # Optional: include unwarped frame as weak evidence
                if args.fallback_no_align:
                    aligned.append(nbr)

        if not aligned:
            # No usable neighbours -- copy frame unchanged
            cv2.imwrite(str(out_dir / pngs[ref_idx].name), ref_bgr)
            cache.evict_before(ref_idx - half)
            continue

        # Temporal median background from aligned stack
        stack = np.stack([ref_bgr] + aligned, axis=0).astype(np.float32)
        median_bg = np.median(stack, axis=0).astype(np.uint8)

        # Detect anomalies
        mask = detect_droplet_mask(
            ref_bgr, median_bg,
            threshold=args.threshold,
            min_area=args.min_area,
            morph_close=args.morph_close,
            morph_dilate=args.morph_dilate,
        )

        # Infill
        result = apply_infill(
            ref_bgr, mask, median_bg,
            method=args.infill,
            radius=args.inpaint_radius,
        )

        cv2.imwrite(str(out_dir / pngs[ref_idx].name), result)

        if dbg_dir is not None:
            dbg = make_debug_image(
                ref_bgr, mask, result.copy(),
                n_aligned=len(aligned),
                n_failed=n_failed_this,
            )
            cv2.imwrite(str(dbg_dir / pngs[ref_idx].name), dbg)

        # Evict frames the window has moved past
        cache.evict_before(ref_idx - half)

    total_neighbours = stats_aligned + stats_failed
    pct = 100 * stats_aligned / total_neighbours if total_neighbours else 0
    print(f"\nAlignment: {stats_aligned}/{total_neighbours} succeeded ({pct:.1f}%)")
    print(f"Skipped (unreadable): {stats_skipped}")
    print(f"Output: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="framer",
        description="Feature-aligned multi-frame water droplet removal from PNG sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("folder", type=Path,
                   help="Folder containing sequential PNG frames (sorted by filename).")
    p.add_argument("--out", default="cleaned", metavar="NAME",
                   help="Output subfolder name. Default: cleaned")

    p.add_argument("--window", type=int, default=15, metavar="N",
                   help="Sliding window size: how many frames to use per reference. "
                        "Larger windows give better detection but use more memory. "
                        "Default: 15")

    g = p.add_argument_group("feature alignment")
    g.add_argument("--detector", choices=["orb", "sift"], default="orb",
                   help="Feature detector. sift requires opencv-contrib-python. "
                        "Default: orb")
    g.add_argument("--max-features", type=int, default=2000, metavar="N",
                   help="Max keypoints per frame. Raise for scenes with lots of "
                        "texture (e.g. coastline). Default: 2000")
    g.add_argument("--ratio-thresh", type=float, default=0.75, metavar="R",
                   help="Lowe ratio-test threshold for match filtering. "
                        "Lower = stricter. Default: 0.75")
    g.add_argument("--ransac-thresh", type=float, default=4.0, metavar="T",
                   help="RANSAC reprojection threshold in pixels. Default: 4.0")
    g.add_argument("--fallback-no-align", action="store_true",
                   help="When homography estimation fails, include the raw "
                        "(unwarped) neighbour in the background stack as a "
                        "weak fallback. May help in scenes with low texture.")

    g2 = p.add_argument_group("anomaly detection")
    g2.add_argument("--threshold", type=float, default=30.0, metavar="T",
                    help="Absolute deviation from background median (0-255) above "
                         "which a pixel is flagged as a droplet. Lower catches "
                         "subtler droplets but risks false positives. Default: 30")
    g2.add_argument("--min-area", type=int, default=40, metavar="A",
                    help="Minimum connected blob area in pixels. Smaller blobs are "
                         "discarded as noise. Default: 40")
    g2.add_argument("--morph-close", type=int, default=7, metavar="K",
                    help="Ellipse kernel size for morphological closing "
                         "(fills intra-droplet holes). Default: 7")
    g2.add_argument("--morph-dilate", type=int, default=3, metavar="K",
                    help="Ellipse kernel size for final mask dilation "
                         "(covers droplet halos). Default: 3")

    g3 = p.add_argument_group("infill")
    g3.add_argument("--infill", choices=["median", "telea", "ns"], default="median",
                    help="How to fill detected droplet regions. "
                         "'median' substitutes directly from the aligned background "
                         "model (fast, usually best when BG model is clean). "
                         "'telea'/'ns' use OpenCV inpainting from surrounding pixels "
                         "(useful when few neighbours aligned successfully). "
                         "Default: median")
    g3.add_argument("--inpaint-radius", type=int, default=5, metavar="R",
                    help="Neighbourhood radius for cv2.inpaint. "
                         "Ignored for median infill. Default: 5")

    p.add_argument("--debug", action="store_true",
                   help="Write side-by-side debug images to <out>_debug/ showing "
                        "original | detection overlay | cleaned result.")

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.folder.exists() or not args.folder.is_dir():
        parser.error(f"Folder not found or not a directory: {args.folder}")
    if args.window < 2:
        parser.error("--window must be >= 2")

    process_folder(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

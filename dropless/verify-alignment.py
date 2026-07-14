#!/usr/bin/env python3
"""
verify-alignment.py -- Visualise homography alignment on the first N frames.

For each frame in the selected range, computes the homography to the reference
frame and saves the warped result to an output folder. Flip through the output
PNGs in any image viewer to confirm that the scene is correctly registered.

Also writes:
  overlay.png  -- all warped frames blended at low opacity over the reference
                  (a well-aligned stack looks sharp; misaligned frames smear)
  diff_NNN.png -- per-frame absolute difference from the reference after warping
                  (dark = well aligned, bright = residual error or droplets)

Usage
-----
  python verify-alignment.py FOLDER [options]

Examples
--------
  python verify-alignment.py frames/
  python verify-alignment.py frames/ --n 30 --ref 10 --out align_check
  python verify-alignment.py frames/ --detector sift
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def make_detector(name: str, max_features: int):
    if name == "sift":
        try:
            det = cv2.SIFT_create(nfeatures=max_features)
            mat = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
            return det, mat
        except AttributeError:
            print("[warn] SIFT unavailable, falling back to ORB.")
    det = cv2.ORB_create(nfeatures=max_features)
    mat = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return det, mat


def compute_homography(src_gray, dst_gray, detector, matcher,
                       ratio_thresh=0.75, ransac_thresh=4.0, min_inliers=8):
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
    H, mask = cv2.findHomography(pts_s, pts_d, cv2.RANSAC, ransac_thresh,
                                 maxIters=2000, confidence=0.995)
    if H is None or mask is None:
        return None, 0
    n_in = int(mask.sum())
    if n_in < min_inliers:
        return None, 0
    det = float(np.linalg.det(H[:2, :2]))
    if det < 0.1 or det > 10.0:
        return None, 0
    return H, n_in


def run(args):
    src = Path(args.folder)
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print(f"No PNGs found in: {src}")
        sys.exit(1)

    # Select range
    start = max(0, args.start)
    end   = min(len(pngs) - 1, start + args.n - 1)
    pngs  = pngs[start : end + 1]
    n     = len(pngs)

    ref_local = args.ref if args.ref is not None else n // 2
    if not (0 <= ref_local < n):
        print(f"--ref {ref_local} out of range [0, {n-1}]")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else src / "alignment_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source    : {src}")
    print(f"Frames    : {start} - {end}  ({n} total)")
    print(f"Reference : index {ref_local} -> {pngs[ref_local].name}")
    print(f"Output    : {out_dir}")
    print()

    detector, matcher = make_detector(args.detector, args.max_features)

    ref_bgr = cv2.imread(str(pngs[ref_local]))
    if ref_bgr is None:
        print(f"Cannot read reference: {pngs[ref_local]}")
        sys.exit(1)
    H_img, W_img = ref_bgr.shape[:2]
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    # Save reference with a clear label
    ref_out = ref_bgr.copy()
    cv2.putText(ref_out, f"REFERENCE ({pngs[ref_local].name})",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(str(out_dir / f"warped_{ref_local:04d}_{pngs[ref_local].stem}.png"),
                ref_out)

    # Accumulator for overlay
    overlay_acc = ref_bgr.astype(np.float64)
    overlay_count = 1

    results = []  # (local_idx, name, status, n_inliers, warped)

    for i, png in enumerate(pngs):
        if i == ref_local:
            results.append((i, png.name, "reference", 0, ref_bgr))
            continue

        frame = cv2.imread(str(png))
        if frame is None:
            print(f"  [{i:>3}] {png.name}  SKIP (unreadable)")
            results.append((i, png.name, "unreadable", 0, None))
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, n_inliers = compute_homography(
            gray, ref_gray, detector, matcher,
            ratio_thresh=args.ratio_thresh,
            ransac_thresh=args.ransac_thresh,
        )

        if H is None:
            print(f"  [{i:>3}] {png.name}  FAILED  (not enough matches)")
            results.append((i, png.name, "failed", 0, frame))
            # Save unwarped with red failure label
            fail_img = frame.copy()
            cv2.putText(fail_img, "ALIGN FAILED", (10, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imwrite(
                str(out_dir / f"warped_{i:04d}_{png.stem}.png"), fail_img)
            continue

        warped = cv2.warpPerspective(frame, H, (W_img, H_img),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        # Label with stats
        out_img = warped.copy()
        cv2.putText(out_img, f"frame {i}  inliers={n_inliers}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)
        cv2.imwrite(
            str(out_dir / f"warped_{i:04d}_{png.stem}.png"), out_img)

        # Diff image vs reference
        diff = cv2.absdiff(warped, ref_bgr)
        diff_bright = cv2.convertScaleAbs(diff, alpha=3.0)  # amplify for visibility
        cv2.putText(diff_bright, f"diff frame {i}  inliers={n_inliers}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.imwrite(
            str(out_dir / f"diff_{i:04d}_{png.stem}.png"), diff_bright)

        overlay_acc += warped.astype(np.float64)
        overlay_count += 1

        print(f"  [{i:>3}] {png.name}  OK  inliers={n_inliers}")
        results.append((i, png.name, "ok", n_inliers, warped))

    # --- Overlay image ---
    overlay = (overlay_acc / overlay_count).clip(0, 255).astype(np.uint8)
    cv2.putText(overlay, f"overlay: {overlay_count} frames aligned",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay)

    # --- Summary ---
    ok      = sum(1 for r in results if r[2] == "ok")
    failed  = sum(1 for r in results if r[2] == "failed")
    skipped = sum(1 for r in results if r[2] == "unreadable")
    avg_in  = (sum(r[3] for r in results if r[2] == "ok") / ok) if ok else 0

    print()
    print(f"Aligned  : {ok}/{n-1} neighbours  (avg inliers: {avg_in:.0f})")
    print(f"Failed   : {failed}")
    print(f"Skipped  : {skipped}")
    print(f"Output   : {out_dir}")
    print(f"  warped_NNNN_*.png  -- scene registered to reference")
    print(f"  diff_NNNN_*.png    -- residual error (bright = misalignment or artifact)")
    print(f"  overlay.png        -- all frames blended (sharp = good alignment)")


def build_parser():
    p = argparse.ArgumentParser(
        prog="verify-alignment",
        description="Visualise homography alignment on a range of PNG frames.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing PNG frames.")
    p.add_argument("--n", type=int, default=20, metavar="N",
                   help="Number of frames to process. Default: 20")
    p.add_argument("--start", type=int, default=0, metavar="N",
                   help="0-based index of first frame. Default: 0")
    p.add_argument("--ref", type=int, default=None, metavar="N",
                   help="0-based index within range to use as reference. "
                        "Default: middle of range")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output folder. Default: <folder>/alignment_check/")
    g = p.add_argument_group("feature alignment")
    g.add_argument("--detector", choices=["orb", "sift"], default="orb")
    g.add_argument("--max-features", type=int, default=2000)
    g.add_argument("--ratio-thresh", type=float, default=0.75)
    g.add_argument("--ransac-thresh", type=float, default=4.0)
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.folder.exists() or not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")
    run(args)


if __name__ == "__main__":
    main()

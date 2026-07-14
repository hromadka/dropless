#!/usr/bin/env python3
"""
horizon-stabilize-otsu.py -- Level marine dashcam frames via sky/water contrast.

Exploits two strong priors of the footage:
  1. Sky (bright) sits above the horizon; water (dark) sits below.
  2. The horizon extends continuously across the full image width.
  3. The horizon passes roughly through the image centroid.

Detection pipeline (per frame)
------------------------------
1. Optional Gaussian blur to suppress wave texture and spray.
2. Otsu threshold → binary sky (white) / water (black) mask.
   Otsu works well here because the bimodal histogram (bright sky / dark water)
   is exactly the distribution it is designed for.
3. Per-column optimal split: for each column x find the row y that maximises
       (sky pixels in rows [0 .. y])  +  (water pixels in rows [y .. H))
   This integral measure is robust to local noise because it uses all pixels in
   the column rather than a single edge response.
4. Confidence filter: keep only columns where the sky-above fraction AND the
   water-below fraction both exceed a threshold.  This rejects uniform columns
   (all sky, all water, or heavy glare) that would otherwise pass the score test.
5. RANSAC line fit through the surviving (x, y_split) pairs → slope + intercept
   → horizon angle = arctan(slope).

Fallback (row-gradient)
-----------------------
If Otsu yields too few confident column points (e.g. very overcast frames with
no clear contrast boundary) the script optionally falls back to comparing the
row-mean brightness gradient of the left and right image halves to estimate
tilt from the difference in their gradient peaks.

Smoothing / output
------------------
Angles are smoothed over a causal N-frame window to suppress per-frame jitter.
Each frame is rotated by -angle_deg so the horizon becomes level and written to
<folder>/leveled/.  In --debug mode the original frame is saved annotated
(no rotation): green = detected horizon, red = where it would land after leveling.

Usage
-----
  python horizon-stabilize-otsu.py FOLDER [options]

Examples
--------
  # Default run
  python horizon-stabilize-otsu.py frames/

  # Debug: see detected vs leveled horizon, no rotation applied
  python horizon-stabilize-otsu.py frames/ --debug

  # Tune sensitivity and smoothing
  python horizon-stabilize-otsu.py frames/ --min-confidence 0.8 --smooth 10

  # Crop black corners away after rotation
  python horizon-stabilize-otsu.py frames/ --crop
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):   # type: ignore[misc]
        desc = kwargs.get("desc", "")
        items = list(iterable)
        n = len(items)
        for i, item in enumerate(items):
            pct = int(100 * (i + 1) / n) if n else 0
            print(f"\r{desc}: {i+1}/{n} ({pct}%)", end="", flush=True)
            yield item
        print()


# ---------------------------------------------------------------------------
# Per-column sky/water transition
# ---------------------------------------------------------------------------

def column_transition_points(
    gray: np.ndarray,
    blur_ksize: int = 5,
    min_confidence: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the sky/water boundary row for each image column.

    Strategy
    --------
    For each column we find the row y* that maximises:

        score(y) = (sky pixels above y) + (water pixels below y)

    A perfect column with an ideal boundary scores H (the image height).
    We then compute a confidence value:

        confidence = (fraction of sky above y*) * (fraction of water below y*)

    A clean boundary → confidence near 1.0.
    A uniform column (all sky, all water, glare) → confidence near 0.0.

    Parameters
    ----------
    gray           : 2-D uint8 grayscale image (already cropped to search zone)
    blur_ksize     : Gaussian blur kernel before Otsu (odd; 1 = disabled)
    min_confidence : Discard columns below this confidence threshold

    Returns
    -------
    xs, ys : float32 arrays of valid (column_index, boundary_row) pairs
    """
    H, W = gray.shape

    if blur_ksize > 1:
        k = blur_ksize | 1          # ensure odd
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # Otsu threshold: sky → 255 (1.0), water → 0 (0.0)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    sky = binary.astype(np.float32) / 255.0   # (H, W)  1=sky, 0=water

    # Cumulative sky from top (row 0)
    sky_above = np.cumsum(sky, axis=0)                          # (H, W)

    # Cumulative water from bottom (row H-1)
    water_below = np.cumsum((1.0 - sky)[::-1], axis=0)[::-1]   # (H, W)

    # Best split row per column
    score     = sky_above + water_below            # (H, W)
    best_rows = np.argmax(score, axis=0)           # (W,)  in [0, H-1]
    col_idx   = np.arange(W)

    # --- Confidence filter ---
    sky_at    = sky_above[best_rows, col_idx]      # sky pixels above y*
    water_at  = water_below[best_rows, col_idx]    # water pixels below y*
    n_above   = (best_rows + 1).astype(np.float32)
    n_below   = (H - best_rows).astype(np.float32)

    sky_frac   = sky_at   / np.maximum(n_above, 1.0)
    water_frac = water_at / np.maximum(n_below, 1.0)
    confidence = sky_frac * water_frac             # ∈ [0, 1]

    valid = confidence >= min_confidence
    xs    = col_idx[valid].astype(np.float32)
    ys    = best_rows[valid].astype(np.float32)

    return xs, ys


# ---------------------------------------------------------------------------
# RANSAC line fit
# ---------------------------------------------------------------------------

def ransac_line(
    xs: np.ndarray,
    ys: np.ndarray,
    n_iter: int = 500,
    inlier_thresh: float = 5.0,
    min_inliers: int = 50,
) -> tuple[float | None, float | None, np.ndarray | None]:
    """
    Fit y = slope * x + intercept to (xs, ys) using RANSAC.

    Returns (slope, intercept, inlier_bool_mask) or (None, None, None).
    The returned line is refit by least squares on all final inliers.
    """
    n = len(xs)
    if n < max(2, min_inliers):
        return None, None, None

    rng = np.random.default_rng(0)
    best_mask = np.zeros(n, dtype=bool)

    for _ in range(n_iter):
        i, j = rng.choice(n, 2, replace=False)
        dx = xs[j] - xs[i]
        if abs(dx) < 1.0:
            continue
        s = (ys[j] - ys[i]) / dx
        b = ys[i] - s * xs[i]
        res  = np.abs(ys - (s * xs + b))
        mask = res < inlier_thresh
        if mask.sum() > best_mask.sum():
            best_mask = mask
            if best_mask.sum() > n * 0.9:
                break   # good enough — early exit

    if best_mask.sum() < min_inliers:
        return None, None, None

    # Least-squares refit on inliers
    coeffs = np.polyfit(xs[best_mask], ys[best_mask], 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    # Recompute final inlier mask with refined line
    res        = np.abs(ys - (slope * xs + intercept))
    final_mask = res < inlier_thresh

    if final_mask.sum() < min_inliers:
        return None, None, None

    return slope, intercept, final_mask


# ---------------------------------------------------------------------------
# Fallback: row-gradient tilt estimate
# ---------------------------------------------------------------------------

def gradient_fallback(gray: np.ndarray) -> float | None:
    """
    Estimate horizon tilt from row-mean brightness gradient.

    Splits the image into left and right halves, finds the row of the
    steepest downward gradient (sky→water transition) in each half, and
    computes the tilt angle from the difference.
    """
    H, W = gray.shape
    half = W // 2

    def peak_row(strip: np.ndarray) -> int:
        means = np.mean(strip, axis=1).astype(np.float32)
        return int(np.argmin(np.diff(means)))   # most negative gradient

    left_y  = peak_row(gray[:, :half])
    right_y = peak_row(gray[:, half:])
    return float(np.rad2deg(np.arctan2(float(right_y - left_y), float(half))))


# ---------------------------------------------------------------------------
# Top-level detection
# ---------------------------------------------------------------------------

def detect_horizon(
    gray: np.ndarray,
    blur_ksize: int,
    min_confidence: float,
    ransac_thresh: float,
    ransac_iters: int,
    min_columns: int,
    use_fallback: bool,
) -> tuple[float | None, tuple[float, float] | None]:
    """
    Detect the horizon angle and line equation.

    Returns
    -------
    angle_deg   : tilt to correct (positive = right side higher), or None
    line_params : (slope, intercept) of detected line, or None
    """
    xs, ys = column_transition_points(gray, blur_ksize, min_confidence)

    if len(xs) >= min_columns:
        slope, intercept, _ = ransac_line(
            xs, ys,
            n_iter=ransac_iters,
            inlier_thresh=ransac_thresh,
            min_inliers=min_columns,
        )
        if slope is not None:
            return float(np.rad2deg(np.arctan(slope))), (slope, intercept)

    # Primary method failed
    if use_fallback:
        angle = gradient_fallback(gray)
        return angle, None

    return None, None


# ---------------------------------------------------------------------------
# Angle smoother
# ---------------------------------------------------------------------------

class AngleSmoother:
    """Causal sliding-window mean of detected horizon angles."""

    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, angle: float | None) -> float | None:
        if angle is None:
            return float(np.mean(self._buf)) if self._buf else None
        self._buf.append(angle)
        return float(np.mean(self._buf))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotate_frame(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate img by -angle_deg around its centre to level the horizon."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def largest_inscribed_rect(
    w: int, h: int, angle_deg: float
) -> tuple[int, int, int, int]:
    """Return (x, y, crop_w, crop_h) of the largest axis-aligned rect with
    no black corners after rotating a w×h image by angle_deg degrees."""
    a = abs(np.deg2rad(angle_deg))
    if a == 0.0:
        return 0, 0, w, h
    sin_a, cos_a = np.sin(a), np.cos(a)
    if w <= h:
        hw = (w / 2) / (cos_a + sin_a * (h / w))
        hh = hw * h / w
    else:
        hh = (h / 2) / (cos_a + sin_a * (w / h))
        hw = hh * w / h
    cw, ch = int(2 * hw), int(2 * hh)
    return (w - cw) // 2, (h - ch) // 2, cw, ch


# ---------------------------------------------------------------------------
# Debug annotation
# ---------------------------------------------------------------------------

def annotate_frame(
    frame: np.ndarray,
    line_params: tuple[float, float] | None,
    raw_angle: float | None,
    used_angle: float | None,
) -> np.ndarray:
    """
    Return an annotated copy of the original frame (no rotation).

    Green line  = detected horizon (extended to full image width).
    Red line    = horizontal at the horizon's centre-column y, i.e. where the
                  horizon will sit in the image after leveling.
    Yellow text = detected / used angles.
    """
    out = frame.copy()
    H, W = frame.shape[:2]

    if line_params is not None:
        slope, intercept = line_params
        # Green: full-width line at detected slope
        y_left  = int(round(intercept))
        y_right = int(round(slope * W + intercept))
        cv2.line(out, (0, y_left), (W, y_right), (0, 255, 0), 2)

        # Red: horizontal through the centre-column y (post-leveling position)
        y_mid = int(round(slope * (W / 2.0) + intercept))
        cv2.line(out, (0, y_mid), (W, y_mid), (0, 0, 255), 2)

    label = (f"raw={raw_angle:.2f}  used={used_angle:.2f} deg"
             if used_angle is not None else "no horizon detected")
    cv2.putText(out, label, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    src  = Path(args.folder)
    pngs = sorted(src.glob("*.png"))
    if not pngs:
        print(f"No PNG files found in: {src}")
        sys.exit(1)

    total = len(pngs)
    start = max(0, args.start)
    end   = min(total - 1, args.end if args.end is not None else total - 1)
    pngs  = pngs[start : end + 1]
    n     = len(pngs)

    out_dir = Path(args.out) if args.out else src / "leveled"
    out_dir.mkdir(parents=True, exist_ok=True)

    smoother = AngleSmoother(args.smooth) if args.smooth > 1 else None

    sample = cv2.imread(str(pngs[0]))
    if sample is None:
        print(f"Could not read first frame: {pngs[0]}")
        sys.exit(1)
    H_img, W_img = sample.shape[:2]
    search_h = int(H_img * args.search_top)

    print(f"Source       : {src}  ({total} total, {n} selected)")
    print(f"Output       : {out_dir}")
    print(f"Frame size   : {W_img}x{H_img}")
    print(f"Search zone  : top {int(args.search_top * 100)}%  ({search_h}px)")
    print(f"Blur         : {args.blur}px kernel")
    print(f"Min confidence : {args.min_confidence}")
    print(f"RANSAC thresh  : {args.ransac_thresh}px  iters={args.ransac_iters}")
    print(f"Min columns  : {args.min_columns}")
    print(f"Smoothing    : {args.smooth}-frame window")
    print(f"Max angle    : {args.max_angle} deg")
    print(f"Fallback     : {args.fallback}")
    print(f"Debug mode   : {args.debug}")
    print()

    n_detected = n_skipped = n_clamped = 0

    for png in tqdm(pngs, desc="Processing", unit="frame"):
        frame = cv2.imread(str(png))
        if frame is None:
            n_skipped += 1
            continue

        gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_gray = gray[:search_h, :]

        raw_angle, line_params = detect_horizon(
            search_gray,
            blur_ksize     = args.blur,
            min_confidence = args.min_confidence,
            ransac_thresh  = args.ransac_thresh,
            ransac_iters   = args.ransac_iters,
            min_columns    = args.min_columns,
            use_fallback   = args.fallback,
        )

        used_angle = smoother.update(raw_angle) if smoother else raw_angle

        # ---- Debug mode: annotate original, do not rotate ----
        if args.debug:
            out = annotate_frame(frame, line_params, raw_angle, used_angle)
            cv2.imwrite(str(out_dir / png.name), out)
            if used_angle is None:
                n_skipped += 1
            else:
                n_detected += 1
            continue

        # ---- Normal mode: rotate to level horizon ----
        if used_angle is None:
            cv2.imwrite(str(out_dir / png.name), frame)
            n_skipped += 1
            continue

        n_detected += 1

        angle = used_angle
        if abs(angle) > args.max_angle:
            angle = float(np.sign(angle) * args.max_angle)
            n_clamped += 1

        rotated = rotate_frame(frame, angle)

        if args.crop:
            x, y, cw, ch = largest_inscribed_rect(W_img, H_img, angle)
            rotated = rotated[y:y+ch, x:x+cw]
            rotated = cv2.resize(rotated, (W_img, H_img),
                                 interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(out_dir / png.name), rotated)

    print(f"\nDone.")
    print(f"  Leveled  : {n_detected}")
    print(f"  Clamped  : {n_clamped}  (exceeded --max-angle {args.max_angle})")
    print(f"  Skipped  : {n_skipped}  (unreadable or no horizon found)")
    print(f"  Output   : {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="horizon-stabilize-otsu",
        description="Level marine dashcam frames using sky/water Otsu contrast.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing sequentially-named PNG frames.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output folder. Default: <folder>/leveled/")
    p.add_argument("--start", type=int, default=0,   metavar="N",
                   help="0-based index of first frame to process. Default: 0")
    p.add_argument("--end",   type=int, default=None, metavar="N",
                   help="0-based index of last frame to process. Default: last")

    g = p.add_argument_group("detection")
    g.add_argument("--blur", type=int, default=5, metavar="K",
                   help="Gaussian blur kernel size before Otsu (odd integer). "
                        "Suppresses wave texture. 1 = disabled. Default: 5")
    g.add_argument("--min-confidence", type=float, default=0.70, metavar="F",
                   help="Column confidence threshold (0–1). A column is used "
                        "only when its sky-above fraction × water-below fraction "
                        "exceeds this value. Default: 0.70")
    g.add_argument("--ransac-thresh", type=float, default=5.0, metavar="PX",
                   help="RANSAC inlier distance threshold in pixels. Default: 5.0")
    g.add_argument("--ransac-iters", type=int, default=500, metavar="N",
                   help="Maximum RANSAC iterations. Default: 500")
    g.add_argument("--min-columns", type=int, default=50, metavar="N",
                   help="Minimum number of confident column points required "
                        "to attempt a line fit. Default: 50")
    g.add_argument("--search-top", type=float, default=0.67, metavar="F",
                   help="Fraction of frame height to search (top portion only). "
                        "Default: 0.67  (top two-thirds)")
    g.add_argument("--fallback", action="store_true",
                   help="Fall back to row-gradient angle estimate when Otsu "
                        "yields too few column points.")

    g2 = p.add_argument_group("stabilization")
    g2.add_argument("--smooth", type=int, default=5, metavar="N",
                    help="Smooth detected angles over the last N frames. "
                         "1 = no smoothing. Default: 5")
    g2.add_argument("--max-angle", type=float, default=10.0, metavar="DEG",
                    help="Clamp correction to this many degrees. Prevents "
                         "over-correction on bad detections. Default: 10")
    g2.add_argument("--crop", action="store_true",
                    help="Crop to the largest all-content rectangle after "
                         "rotation, then rescale to original resolution.")

    p.add_argument("--debug", action="store_true",
                   help="Save annotated originals without rotating. "
                        "Green = detected horizon. Red = leveled position.")
    return p


def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)
    if not args.folder.exists() or not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")
    if args.smooth < 1:
        parser.error("--smooth must be >= 1")
    if args.max_angle <= 0:
        parser.error("--max-angle must be > 0")
    if not (0.0 < args.search_top <= 1.0):
        parser.error("--search-top must be in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()

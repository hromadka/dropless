#!/usr/bin/env python3
"""
horizon-stabilize.py -- Level each frame by detecting and aligning the horizon.

Detects the dominant near-horizontal line in each frame using the Hough
transform and rotates the image so that line becomes perfectly level.
Designed for marine footage where a clear sky/water horizon is visible.

The detected angles are optionally smoothed over time (--smooth N) to
suppress per-frame jitter caused by waves or noise in the horizon detection.

Output frames are the same resolution as input. Black triangular corners
introduced by rotation are filled using BORDER_REFLECT so there are no
hard edges.

Usage
-----
  python horizon-stabilize.py FOLDER [options]

Examples
--------
  # Default: detect horizon, level frames, write to frames/leveled/
  python horizon-stabilize.py frames/

  # Smooth detected angles over a 15-frame window to reduce jitter
  python horizon-stabilize.py frames/ --smooth 15

  # Save debug images showing the detected horizon line overlaid
  python horizon-stabilize.py frames/ --debug

  # Crop to the largest rectangle with no black corners after rotation
  python horizon-stabilize.py frames/ --crop

  # Clamp maximum rotation to avoid over-correction on bad detections
  python horizon-stabilize.py frames/ --max-angle 5.0
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
# Horizon detection
# ---------------------------------------------------------------------------

def detect_horizon_angle(
    gray: np.ndarray,
    canny_low: int = 30,
    canny_high: int = 100,
    hough_thresh: int = 80,
    angle_tolerance_deg: float = 20.0,
) -> float | None:
    """
    Detect the dominant near-horizontal line and return its tilt in degrees.

    A positive angle means the right side of the horizon is higher than the
    left (clockwise tilt); a negative angle means counter-clockwise tilt.
    Returns None if no reliable horizon is found.

    Strategy:
      1. Blur slightly to suppress wave texture.
      2. Canny edge detection.
      3. Probabilistic Hough line transform.
      4. Keep lines whose angle is within `angle_tolerance_deg` of horizontal.
      5. Weight each line by its length and return the length-weighted median
         angle.
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    # Probabilistic Hough: returns line segments (x1,y1,x2,y2)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=gray.shape[1] // 6,   # at least 1/6 of image width
        maxLineGap=gray.shape[1] // 10,
    )

    if lines is None:
        return None

    tol = np.deg2rad(angle_tolerance_deg)
    angles = []
    lengths = []

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.arctan2(dy, dx)   # radians, positive = clockwise tilt
        if abs(angle) > tol:
            continue
        length = np.hypot(dx, dy)
        angles.append(angle)
        lengths.append(length)

    if not angles:
        return None

    angles  = np.array(angles)
    lengths = np.array(lengths)

    # Length-weighted median via sorting
    order   = np.argsort(angles)
    angles  = angles[order]
    lengths = lengths[order]
    cumw    = np.cumsum(lengths)
    median_idx = np.searchsorted(cumw, cumw[-1] / 2)
    return float(np.rad2deg(angles[median_idx]))


def detect_horizon_line_for_debug(
    gray: np.ndarray,
    angle_deg: float,
    canny_low: int = 30,
    canny_high: int = 100,
    hough_thresh: int = 80,
    angle_tolerance_deg: float = 20.0,
) -> tuple[int, int, int, int] | None:
    """Return the longest near-horizontal Hough segment close to angle_deg."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hough_thresh,
        minLineLength=gray.shape[1] // 6,
        maxLineGap=gray.shape[1] // 10,
    )
    if lines is None:
        return None
    tol = np.deg2rad(angle_tolerance_deg)
    best_len = 0
    best_seg = None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = x2 - x1, y2 - y1
        if dx == 0:
            continue
        a = np.arctan2(dy, dx)
        if abs(a) > tol:
            continue
        ln = np.hypot(dx, dy)
        if ln > best_len:
            best_len = ln
            best_seg = (int(x1), int(y1), int(x2), int(y2))
    return best_seg


# ---------------------------------------------------------------------------
# Angle smoothing
# ---------------------------------------------------------------------------

class AngleSmoother:
    """
    Causal (one-sided) sliding-window mean for detected horizon angles.
    Keeps the last `window` valid angles and returns their mean.
    Falls back to the raw angle when the buffer is not yet full.
    """

    def __init__(self, window: int) -> None:
        self._buf: deque[float] = deque(maxlen=window)

    def update(self, angle: float | None) -> float | None:
        if angle is None:
            # Return last smoothed value if available, else None
            return float(np.mean(self._buf)) if self._buf else None
        self._buf.append(angle)
        return float(np.mean(self._buf))


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def rotate_frame(
    img: np.ndarray,
    angle_deg: float,
    border_mode: int = cv2.BORDER_REFLECT,
) -> np.ndarray:
    """Rotate img by -angle_deg around its centre (counter-clockwise correction)."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=border_mode)


def largest_inscribed_rect(w: int, h: int, angle_deg: float) -> tuple[int, int, int, int]:
    """
    Return (x, y, crop_w, crop_h) of the largest axis-aligned rectangle
    that fits inside a w x h image after rotation by angle_deg degrees.
    """
    angle = abs(np.deg2rad(angle_deg))
    if angle == 0:
        return 0, 0, w, h
    sin_a, cos_a = np.sin(angle), np.cos(angle)
    if w <= h:
        half_w = w / 2
        hw = half_w / (cos_a + sin_a * (h / w))
        hh = hw * h / w
    else:
        half_h = h / 2
        hh = half_h / (cos_a + sin_a * (w / h))
        hw = hh * w / h
    cw, ch = int(2 * hw), int(2 * hh)
    x = (w - cw) // 2
    y = (h - ch) // 2
    return x, y, cw, ch


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    src = Path(args.folder)
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

    print(f"Source    : {src}  ({total} total, {n} selected)")
    print(f"Output    : {out_dir}")
    print(f"Smoothing : {args.smooth}-frame window")
    print(f"Max angle : {args.max_angle} deg")
    print(f"Crop      : {args.crop}")
    print()

    detected = skipped = clamped = 0

    # Determine crop rect from the first frame's dimensions (assume all same size)
    sample = cv2.imread(str(pngs[0]))
    H_img, W_img = sample.shape[:2] if sample is not None else (0, 0)

    for png in tqdm(pngs, desc="Leveling", unit="frame"):
        frame = cv2.imread(str(png))
        if frame is None:
            print(f"\n[skip] {png.name}")
            skipped += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        search_gray = gray[: H_img * 2 // 3, :]   # top 2/3 only
        raw_angle = detect_horizon_angle(
            search_gray,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            hough_thresh=args.hough_thresh,
            angle_tolerance_deg=args.angle_tol,
        )

        # Smooth
        angle = smoother.update(raw_angle) if smoother else raw_angle

        if args.debug:
            # Debug mode: annotate the original frame, do NOT rotate.
            # Green line  = detected horizon (extended to full image width).
            # Red line    = where the horizon would be after leveling
            #               (horizontal line through the segment midpoint).
            dbg = frame.copy()
            seg = detect_horizon_line_for_debug(
                search_gray, raw_angle if raw_angle is not None else 0.0,
                canny_low=args.canny_low,
                canny_high=args.canny_high,
                hough_thresh=args.hough_thresh,
                angle_tolerance_deg=args.angle_tol,
            )
            if seg is not None:
                x1, y1, x2, y2 = seg
                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                # Green: extend detected line across the full width
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    left_y  = int(mid_y - slope * mid_x)
                    right_y = int(mid_y + slope * (W_img - mid_x))
                else:
                    left_y = right_y = int(mid_y)
                cv2.line(dbg, (0, left_y), (W_img, right_y), (0, 255, 0), 2)

                # Red: horizontal line at the midpoint y (post-leveling position)
                red_y = int(mid_y)
                cv2.line(dbg, (0, red_y), (W_img, red_y), (0, 0, 255), 2)

            if angle is not None:
                label = f"raw={raw_angle:.2f}  used={angle:.2f} deg"
            else:
                label = "no horizon detected"
            cv2.putText(dbg, label, (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)
            cv2.imwrite(str(out_dir / png.name), dbg)

            if angle is None:
                skipped += 1
            else:
                detected += 1
            continue

        if angle is None:
            # No horizon found: pass frame through unchanged
            cv2.imwrite(str(out_dir / png.name), frame)
            skipped += 1
            continue

        detected += 1

        # Clamp
        if abs(angle) > args.max_angle:
            angle = np.sign(angle) * args.max_angle
            clamped += 1

        rotated = rotate_frame(frame, angle)

        if args.crop:
            x, y, cw, ch = largest_inscribed_rect(W_img, H_img, angle)
            rotated = rotated[y:y+ch, x:x+cw]
            rotated = cv2.resize(rotated, (W_img, H_img), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(out_dir / png.name), rotated)

    print(f"\nDone.")
    print(f"  Leveled  : {detected}")
    print(f"  Clamped  : {clamped}  (exceeded --max-angle {args.max_angle})")
    print(f"  Skipped  : {skipped}  (unreadable or no horizon found)")
    print(f"  Output   : {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="horizon-stabilize",
        description="Level each frame by detecting and aligning the horizon.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folder", type=Path,
                   help="Folder containing PNG frames.")
    p.add_argument("--out", default=None, metavar="PATH",
                   help="Output folder. Default: <folder>/leveled/")
    p.add_argument("--start", type=int, default=0, metavar="N",
                   help="0-based index of first frame to process. Default: 0")
    p.add_argument("--end", type=int, default=None, metavar="N",
                   help="0-based index of last frame to process. Default: last")

    g = p.add_argument_group("horizon detection")
    g.add_argument("--canny-low", type=int, default=30, metavar="T",
                   help="Canny lower threshold. Default: 30")
    g.add_argument("--canny-high", type=int, default=100, metavar="T",
                   help="Canny upper threshold. Default: 100")
    g.add_argument("--hough-thresh", type=int, default=80, metavar="T",
                   help="Hough accumulator threshold. Lower = more sensitive. "
                        "Default: 80")
    g.add_argument("--angle-tol", type=float, default=20.0, metavar="DEG",
                   help="Max degrees from horizontal a line may be and still "
                        "count as a horizon candidate. Default: 20")

    g2 = p.add_argument_group("stabilization")
    g2.add_argument("--smooth", type=int, default=5, metavar="N",
                    help="Smooth detected angles over the last N frames to "
                         "reduce jitter. 1 = no smoothing. Default: 5")
    g2.add_argument("--max-angle", type=float, default=10.0, metavar="DEG",
                    help="Clamp correction to this many degrees. Protects "
                         "against bad horizon detections. Default: 10")
    g2.add_argument("--crop", action="store_true",
                    help="Crop to the largest rectangle with no black corners "
                         "after rotation, then rescale to original size.")

    p.add_argument("--debug", action="store_true",
                   help="Save annotated originals (no rotation applied). "
                        "Green line = detected horizon. "
                        "Red line = where horizon would land after leveling.")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.folder.exists() or not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")
    if args.smooth < 1:
        parser.error("--smooth must be >= 1")
    if args.max_angle <= 0:
        parser.error("--max-angle must be > 0")
    run(args)


if __name__ == "__main__":
    main()

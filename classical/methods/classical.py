"""
methods/classical.py
Classical OpenCV-based water droplet detection and removal.

Algorithm
---------
Phase 1 - Build temporal background model
    Load frames in a sliding window. Compute the per-pixel temporal
    median. Because droplets are stationary on the lens while the scene
    changes, the median converges to the clean background as long as a
    pixel is droplet-free for more than half the window.

Phase 2 - Detect droplets per frame
    Compute absolute per-channel difference from the temporal median,
    threshold, then apply morphological operations to suppress noise and
    fill gaps. Optional: secondary stationarity check via consecutive-
    frame differences to confirm a region truly does not move.

Phase 3 - Inpaint
    Apply cv2.inpaint (Telea or Navier-Stokes) over the mask, OR
    directly substitute the temporal-median pixel values.

Phase 4 - Write output
    Stream cleaned frames to the output MP4.
"""

from __future__ import annotations

import argparse
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

from .base import BaseMethod
from utils.video_io import VideoReader, VideoWriter


class ClassicalMethod(BaseMethod):
    name = "classical"

    # ------------------------------------------------------------------
    # CLI arguments
    # ------------------------------------------------------------------

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        grp = parser.add_argument_group("classical method options")
        grp.add_argument(
            "--window",
            type=int,
            default=31,
            metavar="N",
            help="Temporal window size (odd number of frames) for median "
                 "background model. Larger = more robust, higher memory. "
                 "Default: 31",
        )
        grp.add_argument(
            "--threshold",
            type=float,
            default=28.0,
            metavar="T",
            help="Pixel deviation threshold (0-255) above which a pixel "
                 "is a droplet candidate. Default: 28",
        )
        grp.add_argument(
            "--min-area",
            type=int,
            default=50,
            metavar="A",
            help="Minimum connected-component area (pixels) to keep in "
                 "the droplet mask. Default: 50",
        )
        grp.add_argument(
            "--inpaint-method",
            choices=["telea", "ns", "median"],
            default="telea",
            help="Inpainting strategy: telea (fast marching), "
                 "ns (Navier-Stokes), or median (direct substitution "
                 "from background model). Default: telea",
        )
        grp.add_argument(
            "--inpaint-radius",
            type=int,
            default=5,
            metavar="R",
            help="Neighbourhood radius for cv2.inpaint. "
                 "Ignored for median strategy. Default: 5",
        )
        grp.add_argument(
            "--use-frame-diff",
            action="store_true",
            help="Enable stationarity check: a pixel must also be stable "
                 "across consecutive frames to count as a droplet.",
        )
        grp.add_argument(
            "--morph-close",
            type=int,
            default=7,
            metavar="K",
            help="Kernel size for morphological closing on the mask "
                 "(fills gaps). Default: 7",
        )
        grp.add_argument(
            "--morph-dilate",
            type=int,
            default=3,
            metavar="K",
            help="Kernel size for mask dilation before inpainting. "
                 "Default: 3",
        )
        grp.add_argument(
            "--debug",
            action="store_true",
            help="Write a side-by-side debug video (input | mask | output).",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_median(self, frames: list[np.ndarray]) -> np.ndarray:
        """Per-pixel temporal median across a list of BGR frames."""
        stack = np.stack(frames, axis=0).astype(np.float32)
        return np.median(stack, axis=0).astype(np.uint8)

    def _detect_mask(
        self,
        frame: np.ndarray,
        median_bg: np.ndarray,
        prev_frame: np.ndarray | None,
        next_frame: np.ndarray | None,
    ) -> np.ndarray:
        """Return uint8 binary mask (255=droplet, 0=clean)."""
        args = self.args

        # 1. Median deviation
        diff = cv2.absdiff(frame, median_bg)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(
            diff_gray, args.threshold, 255, cv2.THRESH_BINARY
        )

        # 2. Optional stationarity check
        if args.use_frame_diff and prev_frame is not None and next_frame is not None:
            diff_prev = cv2.absdiff(frame, prev_frame)
            diff_next = cv2.absdiff(frame, next_frame)
            stable_prev = cv2.cvtColor(diff_prev, cv2.COLOR_BGR2GRAY)
            stable_next = cv2.cvtColor(diff_next, cv2.COLOR_BGR2GRAY)
            _, stable_mask = cv2.threshold(
                cv2.max(stable_prev, stable_next),
                args.threshold * 0.5,
                255,
                cv2.THRESH_BINARY_INV,
            )
            mask = cv2.bitwise_and(mask, stable_mask)

        # 3. Morphological closing to fill gaps within droplets
        if args.morph_close > 0:
            k_close = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.morph_close, args.morph_close)
            )
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

        # 4. Remove small blobs
        if args.min_area > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            clean_mask = np.zeros_like(mask)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= args.min_area:
                    clean_mask[labels == lbl] = 255
            mask = clean_mask

        # 5. Dilate to cover droplet edges before inpainting
        if args.morph_dilate > 0:
            k_dil = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (args.morph_dilate, args.morph_dilate)
            )
            mask = cv2.dilate(mask, k_dil)

        return mask

    def _inpaint(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        median_bg: np.ndarray,
    ) -> np.ndarray:
        """Apply the configured inpainting strategy."""
        if mask.max() == 0:
            return frame
        method = self.args.inpaint_method
        if method == "median":
            result = frame.copy()
            result[mask > 0] = median_bg[mask > 0]
            return result
        elif method == "telea":
            return cv2.inpaint(frame, mask, self.args.inpaint_radius, cv2.INPAINT_TELEA)
        else:
            return cv2.inpaint(frame, mask, self.args.inpaint_radius, cv2.INPAINT_NS)

    @staticmethod
    def _make_debug_frame(
        original: np.ndarray,
        mask: np.ndarray,
        result: np.ndarray,
    ) -> np.ndarray:
        """Stack original | mask-overlay | result side by side."""
        h = original.shape[0]
        overlay = original.copy()
        overlay[mask > 0] = (0, 0, 220)
        mask_vis = cv2.addWeighted(original, 0.5, overlay, 0.5, 0)
        sep = np.zeros((h, 3, 3), dtype=np.uint8)
        return np.hstack([original, sep, mask_vis, sep, result])

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process(self, input_path: Path, output_path: Path) -> None:
        args = self.args
        window = args.window
        if window % 2 == 0:
            window += 1
        half = window // 2

        with VideoReader(input_path) as reader:
            total = reader.frame_count
            fps = reader.fps
            W, H = reader.width, reader.height
            print(
                f"[classical] {total} frames  {W}x{H}  {fps:.2f} fps  "
                f"window={window}  threshold={args.threshold}  "
                f"inpaint={args.inpaint_method}"
            )
            print("Loading frames into memory...")
            all_frames: list[np.ndarray] = reader.read_all()

        if not all_frames:
            print("No frames read. Aborting.")
            return

        n = len(all_frames)

        with VideoWriter(output_path, W, H, fps) as writer:
            debug_writer: VideoWriter | None = None
            if args.debug:
                dbg_path = output_path.with_stem(output_path.stem + "_debug")
                debug_writer = VideoWriter(dbg_path, W * 3 + 6, H, fps)

            try:
                for i in tqdm(range(n), desc="Processing", unit="frame"):
                    w_start = max(0, i - half)
                    w_end = min(n, i + half + 1)
                    window_frames = all_frames[w_start:w_end]

                    median_bg = self._build_median(window_frames)

                    prev_f = all_frames[i - 1] if i > 0 else None
                    next_f = all_frames[i + 1] if i < n - 1 else None

                    mask = self._detect_mask(
                        all_frames[i], median_bg, prev_f, next_f
                    )

                    result = self._inpaint(all_frames[i], mask, median_bg)

                    writer.write(result)

                    if debug_writer is not None:
                        debug_writer.write(
                            self._make_debug_frame(all_frames[i], mask, result)
                        )
            finally:
                if debug_writer is not None:
                    debug_writer.close()

        print(f"Done. Output written to: {output_path}")
        if args.debug:
            print(f"Debug video: {output_path.with_stem(output_path.stem + '_debug')}")

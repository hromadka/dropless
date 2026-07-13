"""
utils/video_io.py
Frame-by-frame video reader and writer wrappers around OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple


class VideoReader:
    """Iterate over frames of an MP4 (or any OpenCV-readable video)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def read_all(self) -> list[np.ndarray]:
        """Load every frame into memory. Use only for short clips."""
        frames = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames

    def read_range(self, start: int, end: int) -> list[np.ndarray]:
        """Read frames [start, end) into memory."""
        frames = []
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(end - start):
            ok, frame = self._cap.read()
            if not ok:
                break
            frames.append(frame)
        return frames

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *_) -> None:
        self.close()


class VideoWriter:
    """Write BGR frames to an MP4 file."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        fourcc: str = "mp4v",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*fourcc)
        self._writer = cv2.VideoWriter(str(self.path), codec, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()

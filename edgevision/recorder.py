"""Write annotated frames to an MP4 file."""

from pathlib import Path

import cv2
import numpy as np


class VideoRecorder:
    """Opens the file on the first frame, using that frame's size.

    OpenCV silently drops frames whose size differs from the one the writer was
    opened with, which leaves an empty video; taking the size from the frame
    itself rules that out.
    """

    def __init__(self, path, fps: float = 30.0):
        self.path = Path(path)
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._size = (w, h)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                           self.fps, self._size)
            if not self._writer.isOpened():
                raise RuntimeError(f"Cannot write video: {self.path}")
        if (w, h) != self._size:
            raise ValueError(f"Frame size {(w, h)} differs from the video's {self._size}")
        self._writer.write(frame)
        self.frames += 1

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

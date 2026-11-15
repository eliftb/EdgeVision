"""Frame input: a camera or a video file behind one interface."""

import cv2
import numpy as np


class FrameSource:
    """Reads BGR frames from a camera index ("0") or a video file path."""

    def __init__(self, source: str = "0", width: int = 640, height: int = 480):
        self.is_camera = str(source).isdigit()
        self.cap = cv2.VideoCapture(int(source) if self.is_camera else str(source))
        if not self.cap.isOpened():
            kind = "camera" if self.is_camera else "video"
            raise RuntimeError(f"Cannot open {kind}: {source}")
        if self.is_camera:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self) -> np.ndarray | None:
        """Next frame, or None when the video ends or the camera stops."""
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()

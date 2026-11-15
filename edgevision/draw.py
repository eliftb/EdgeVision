"""Draw detections and a status line onto a frame."""

import cv2
import numpy as np

from .detector import Detection

# BGR colours; a class always gets the same one.
PALETTE = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (223, 107, 155),
           (60, 60, 220), (180, 180, 40), (150, 80, 200)]


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw a box and a "label 87%" tag for each detection (in place)."""
    h, w = frame.shape[:2]
    scale = max(0.45, w / 1300)
    thick = max(1, round(2 * scale))        # text
    line = max(2, round(3 * scale))         # box outline; 1 px disappears on video
    font = cv2.FONT_HERSHEY_SIMPLEX
    for d in detections:
        color = PALETTE[d.class_id % len(PALETTE)]
        x1, y1, x2, y2 = d.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
        text = f"{d.label} {d.confidence * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(text, font, scale, thick)
        # Put the tag above the box, or inside it when the box touches the top edge.
        top = y1 - th - base - 4 if y1 - th - base - 4 >= 0 else y1
        left = max(0, min(x1, w - tw - 6))   # slide left rather than run off the right edge
        cv2.rectangle(frame, (left, top), (left + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(frame, text, (left + 3, top + th + 2), font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return frame


def draw_status(frame: np.ndarray, text: str) -> np.ndarray:
    """Status line (backend, FPS, object count) in the bottom-left corner."""
    h, w = frame.shape[:2]
    scale = max(0.45, w / 1300)
    thick = max(1, round(1.5 * scale))
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(frame, (0, h - th - base - 12), (tw + 16, h), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, h - base - 6), font, scale, (235, 235, 235), thick, cv2.LINE_AA)
    return frame

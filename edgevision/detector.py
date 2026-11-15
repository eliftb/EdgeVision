"""YOLO detection behind one interface, whatever the runtime.

The same class runs PyTorch weights (.pt), an ONNX model (.onnx, through ONNX
Runtime) or a TensorRT engine (.engine); Ultralytics picks the runtime from
the file extension. The rest of the pipeline only ever sees Detection objects.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels of the input frame
    confidence: float
    class_id: int
    label: str


def detections_from_arrays(xyxy, conf, cls, names) -> list[Detection]:
    """Turn raw model output arrays into Detection objects.

    Kept separate from the model call so it can be tested without weights.
    """
    return [
        Detection(box=tuple(int(round(v)) for v in box), confidence=float(c),
                  class_id=int(k), label=str(names[int(k)]))
        for box, c, k in zip(np.asarray(xyxy), np.asarray(conf), np.asarray(cls))
    ]


class Detector:
    def __init__(self, weights="yolov8n.pt", device=None, conf: float = 0.25,
                 imgsz: int = 640, half: bool = False):
        import torch
        from ultralytics import YOLO   # imported here so the rest works without it

        self.weights = Path(weights)
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.imgsz = imgsz
        # Exported models have their precision baked in; `half` only applies to .pt.
        self.half = half and self.weights.suffix == ".pt" and self.device != "cpu"
        self.model = YOLO(str(self.weights), task="detect")

    @property
    def backend(self) -> str:
        return {".pt": "pytorch", ".onnx": "onnxruntime", ".engine": "tensorrt"}.get(
            self.weights.suffix, self.weights.suffix.lstrip("."))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run one BGR frame through preprocessing, the network and NMS."""
        result = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, half=self.half, verbose=False)[0]
        boxes = result.boxes
        # .cpu() waits for the GPU to finish, so timing detect() measures real latency.
        return detections_from_arrays(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                      boxes.cls.cpu().numpy(), result.names)

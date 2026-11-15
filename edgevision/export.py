#!/usr/bin/env python3
"""Export YOLO weights to ONNX or a TensorRT engine.

    python -m edgevision.export yolov8n.pt onnx
    python -m edgevision.export yolov8n.pt engine            # TensorRT, fp32
    python -m edgevision.export yolov8n.pt engine --half     # TensorRT, fp16

A TensorRT engine is built for the GPU and TensorRT version it was exported
on; build it on the device that will run it.
"""

import argparse
from pathlib import Path


def exported_name(weights: Path, fmt: str, half: bool) -> Path:
    """yolov8n.pt -> yolov8n.onnx, yolov8n-fp32.engine or yolov8n-fp16.engine.

    Ultralytics names every engine <stem>.engine, so fp32 and fp16 builds would
    overwrite each other; the precision goes into the file name instead.
    """
    if fmt == "onnx":
        return weights.with_suffix(".onnx")
    return weights.with_name(f"{weights.stem}-{'fp16' if half else 'fp32'}.engine")


def export(weights, fmt: str, half: bool = False, imgsz: int = 640, device=0) -> Path:
    from ultralytics import YOLO

    weights = Path(weights)
    target = exported_name(weights, fmt, half)
    if target.exists():
        return target
    made = Path(YOLO(str(weights)).export(format=fmt, half=half, imgsz=imgsz, device=device))
    if made != target:
        made.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path)
    parser.add_argument("format", choices=["onnx", "engine"])
    parser.add_argument("--half", action="store_true", help="fp16 engine")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    print(f"Exported {export(args.weights, args.format, args.half, args.imgsz)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare the same YOLO model across runtimes: speed, size and accuracy.

    python -m edgevision.benchmark                 # PyTorch, ONNX Runtime, TensorRT
    python -m edgevision.benchmark --accuracy      # also mAP on COCO128 (downloads 7 MB)
    python -m edgevision.benchmark --backends pytorch-fp32 tensorrt-fp16

Latency is one frame end to end: resize/letterbox, network, NMS and copying
the boxes back to the CPU. Missing exports are created first.
"""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .detector import Detector
from .export import export

# name -> (export format or None for the .pt itself, fp16?)
BACKENDS = {
    "pytorch-fp32": (None, False),
    "pytorch-fp16": (None, True),
    "onnxruntime": ("onnx", False),
    "tensorrt-fp32": ("engine", False),
    "tensorrt-fp16": ("engine", True),
}


@dataclass
class Result:
    backend: str
    file: str
    size_mb: float
    mean_ms: float                # whole frame: preprocessing + network + NMS
    p50_ms: float
    p95_ms: float
    fps: float
    network_ms: float             # network forward pass alone
    detections: int               # objects found in the test frame
    map50_95: float | None = None
    error: str | None = None


def summarise(times_ms: list[float]) -> dict:
    """Mean, median and 95th percentile of per-frame latencies."""
    ordered = sorted(times_ms)
    mean = statistics.fmean(ordered)
    return {"mean_ms": mean, "p50_ms": statistics.median(ordered),
            "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
            "fps": 1000 / mean}


def time_detector(detector, frame: np.ndarray, runs: int, warmup: int):
    """Per-frame latency in ms after `warmup` discarded calls, the mean time spent
    in the network alone, and the last output."""
    for _ in range(warmup):   # cuDNN autotuning, lazy CUDA init, engine context creation
        detector.detect(frame)
    times, network, detections = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter()
        detections = detector.detect(frame)
        times.append((time.perf_counter() - t0) * 1000)
        network.append(getattr(detector, "last_network_ms", 0.0))
    return times, statistics.fmean(network), detections


def run_backend(name: str, weights: Path, frame: np.ndarray, runs: int, warmup: int,
                accuracy: bool) -> Result:
    fmt, half = BACKENDS[name]
    try:
        path = export(weights, fmt, half) if fmt else weights
        detector = Detector(path, half=half)
        times, network_ms, detections = time_detector(detector, frame, runs, warmup)
        result = Result(backend=name, file=path.name, size_mb=path.stat().st_size / 2**20,
                        network_ms=network_ms, detections=len(detections), **summarise(times))
    except Exception as exc:   # a missing runtime should not sink the other backends
        return Result(name, "", 0, 0, 0, 0, 0, 0, 0, error=f"{type(exc).__name__}: {exc}")
    if accuracy:
        try:
            result.map50_95 = coco128_map(detector)
        except Exception as exc:   # keep the timings even if validation fails
            print(f"   accuracy check failed: {type(exc).__name__}: {exc}")
    return result


def coco128_map(detector: Detector) -> float:
    """mAP50-95 on the 128-image COCO128 set, same settings for every backend.

    rect=False feeds every backend a square 640x640 letterboxed image; that is
    the only shape the exported models accept, so PyTorch has to match it for
    the numbers to be comparable.
    """
    from ultralytics.nn.autobackend import AutoBackend

    # Ultralytics 8.3.170 reads AutoBackend.batch_size when validating a TensorRT
    # engine but does not set it with TensorRT 10; our engines are built for batch 1.
    if not hasattr(AutoBackend, "batch_size"):
        AutoBackend.batch_size = 1
    metrics = detector.model.val(data="coco128.yaml", imgsz=detector.imgsz, batch=1,
                                 rect=False, half=detector.half, device=detector.device,
                                 verbose=False, plots=False, project="outputs/val",
                                 name=detector.weights.stem, exist_ok=True)
    return float(metrics.box.map)


def format_table(results: list[Result]) -> str:
    """Markdown table; speed-up is relative to the first successful row."""
    ok = [r for r in results if not r.error]
    base = ok[0].mean_ms if ok else 1
    with_map = any(r.map50_95 is not None for r in ok)
    head = "| Backend | File size | Frame mean | p50 | p95 | FPS | Speed-up | Network only | Objects |"
    rule = "|---|---|---|---|---|---|---|---|---|"
    if with_map:
        head, rule = head + " mAP50-95 |", rule + "---|"
    lines = [head, rule]
    for r in results:
        if r.error:
            lines.append(f"| {r.backend} | failed: {r.error[:60]} |")
            continue
        row = (f"| {r.backend} | {r.size_mb:.1f} MB | {r.mean_ms:.1f} ms | {r.p50_ms:.1f} ms | "
               f"{r.p95_ms:.1f} ms | {r.fps:.0f} | {base / r.mean_ms:.2f}x | {r.network_ms:.1f} ms | "
               f"{r.detections} |")
        if with_map:
            row += f" {r.map50_95:.3f} |" if r.map50_95 is not None else " |"
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=Path("yolov8n.pt"))
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS), default=list(BACKENDS))
    parser.add_argument("--image", type=Path, help="test frame (default: Ultralytics' bus.jpg)")
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--accuracy", action="store_true", help="also run COCO128 validation")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark.json"))
    args = parser.parse_args()

    import cv2
    import torch
    from ultralytics.utils import ASSETS

    image = args.image or ASSETS / "bus.jpg"
    frame = cv2.imread(str(image))
    if frame is None:
        raise SystemExit(f"Cannot read image: {image}")

    results = []
    for name in args.backends:
        print(f"-- {name}")
        results.append(run_backend(name, args.weights, frame, args.runs, args.warmup,
                                   args.accuracy))
    table = format_table(results)
    print("\n" + table)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "image": f"{image.name} {frame.shape[1]}x{frame.shape[0]}",
        "runs": args.runs, "warmup": args.warmup,
        "results": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

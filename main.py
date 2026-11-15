#!/usr/bin/env python3
"""Real-time object detection on a camera or a video file.

    python main.py                                  # camera 0, PyTorch weights
    python main.py --weights yolov8n-fp16.engine    # TensorRT engine (see edgevision.export)
    python main.py --source video.mp4 --record outputs/result.mp4
    python main.py --source video.mp4 --no-display --max-frames 300

Press q or Esc to quit.
"""

import argparse
import sys

import cv2

from edgevision.detector import Detector
from edgevision.draw import draw_detections, draw_status
from edgevision.fps import FPSMeter
from edgevision.recorder import VideoRecorder
from edgevision.source import FrameSource

WINDOW = "EdgeVision (q to quit)"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="camera index or video file")
    parser.add_argument("--weights", default="yolov8n.pt",
                        help=".pt, .onnx or .engine; yolov8n.pt is downloaded if missing")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--half", action="store_true", help="fp16 for .pt weights on GPU")
    parser.add_argument("--device", default=None, help="0 for the first GPU, or cpu")
    parser.add_argument("--record", help="save the annotated video to this mp4 file")
    parser.add_argument("--no-display", action="store_true", help="run without a window")
    parser.add_argument("--max-frames", type=int, help="stop after this many frames")
    args = parser.parse_args()

    try:
        source = FrameSource(args.source)
    except RuntimeError as exc:
        sys.exit(str(exc))
    detector = Detector(args.weights, device=args.device, conf=args.conf, half=args.half)
    recorder = VideoRecorder(args.record, source.fps) if args.record else None
    meter = FPSMeter()
    print(f"{args.source}: {source.width}x{source.height}, backend {detector.backend}")

    try:
        while args.max_frames is None or meter.frames < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            detections = detector.detect(frame)
            meter.tick()
            draw_detections(frame, detections)
            draw_status(frame, f"{detector.backend}  {meter.fps:.0f} fps  "
                               f"{len(detections)} objects")
            if recorder:
                recorder.write(frame)
            if not args.no_display:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if recorder:
            recorder.release()
        cv2.destroyAllWindows()

    print(f"{meter.frames} frames, average {meter.average_fps:.1f} fps "
          "(capture + detection + drawing, model loading excluded)")
    if recorder:
        print(f"Saved {recorder.path}")


if __name__ == "__main__":
    main()

"""Tests for the pipeline pieces; no model weights, GPU or camera needed."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from edgevision.benchmark import Result, format_table, summarise, time_detector
from edgevision.detector import Detection, detections_from_arrays
from edgevision.draw import draw_detections, draw_status
from edgevision.export import exported_name
from edgevision.fps import FPSMeter
from edgevision.recorder import VideoRecorder
from edgevision.source import FrameSource


def grey_frame(h=240, w=320, value=128):
    return np.full((h, w, 3), value, dtype=np.uint8)


# --- detections -----------------------------------------------------------

def test_detections_are_built_from_model_arrays():
    dets = detections_from_arrays(
        xyxy=[[10.4, 20.6, 110.5, 220.2]], conf=[0.87], cls=[5.0], names={5: "bus"})
    assert dets == [Detection(box=(10, 21, 110, 220), confidence=pytest.approx(0.87),
                              class_id=5, label="bus")]


def test_no_detections_gives_empty_list():
    assert detections_from_arrays(np.empty((0, 4)), np.empty(0), np.empty(0), {}) == []


# --- fps ------------------------------------------------------------------

def test_fps_counts_from_first_frame_not_from_creation():
    meter = FPSMeter()
    # Frames arrive every 0.1 s, but only after a 5 s "model loading" pause.
    for i in range(11):
        meter.tick(now=5.0 + i * 0.1)
    assert meter.average_fps == pytest.approx(10.0)
    assert meter.fps == pytest.approx(10.0)


def test_fps_window_follows_recent_speed():
    meter = FPSMeter(window=5)
    t = 0.0
    for _ in range(10):      # slow start: 2 fps
        meter.tick(now=t)
        t += 0.5
    for _ in range(10):      # then 20 fps
        meter.tick(now=t)
        t += 0.05
    assert meter.fps == pytest.approx(20.0)
    assert 2.0 < meter.average_fps < 20.0


def test_fps_is_zero_before_two_frames():
    meter = FPSMeter()
    assert meter.fps == 0.0 and meter.average_fps == 0.0
    meter.tick(now=1.0)
    assert meter.fps == 0.0 and meter.average_fps == 0.0


# --- drawing --------------------------------------------------------------

def test_box_is_drawn_where_the_detection_is():
    frame = grey_frame()
    draw_detections(frame, [Detection((100, 100, 200, 180), 0.9, 0, "person")])
    assert (frame[100, 150] != 128).any()      # top edge
    assert (frame[140, 150] == 128).all()      # inside stays untouched
    assert (frame[220, 300] == 128).all()      # far away stays untouched


def test_label_stays_inside_frame_for_box_at_top_edge():
    frame = grey_frame()
    draw_detections(frame, [Detection((0, 0, 80, 60), 0.5, 3, "car")])
    assert frame.shape == (240, 320, 3)
    assert (frame[:20, :60] != 128).any()


def test_label_slides_left_for_box_at_right_edge():
    frame = grey_frame()
    draw_detections(frame, [Detection((300, 100, 319, 200), 0.9, 1, "bicycle")])
    # The tag is wider than the 20 px box, so it must extend to the left of it.
    assert (frame[80:100, 250:295] != 128).any()


def test_status_line_is_drawn_at_the_bottom():
    frame = grey_frame()
    draw_status(frame, "pytorch  30 fps  2 objects")
    assert (frame[-10:, :40] != 128).any()
    assert (frame[:100] == 128).all()


# --- recorder and source --------------------------------------------------

def test_recorded_video_can_be_read_back(tmp_path):
    path = tmp_path / "sub" / "clip.mp4"
    recorder = VideoRecorder(path, fps=10)
    for i in range(12):
        recorder.write(grey_frame(value=20 * i))
    recorder.release()

    source = FrameSource(str(path))
    assert (source.width, source.height) == (320, 240)
    count = 0
    while source.read() is not None:
        count += 1
    source.release()
    assert count == 12


def test_recorder_rejects_a_frame_of_another_size(tmp_path):
    recorder = VideoRecorder(tmp_path / "clip.mp4")
    recorder.write(grey_frame(240, 320))
    with pytest.raises(ValueError):
        recorder.write(grey_frame(480, 640))
    recorder.release()


def test_missing_video_raises_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="Cannot open video"):
        FrameSource(str(tmp_path / "nope.mp4"))


# --- export names and benchmark maths -------------------------------------

def test_engine_precisions_get_different_file_names():
    w = Path("models/yolov8n.pt")
    assert exported_name(w, "onnx", False) == Path("models/yolov8n.onnx")
    assert exported_name(w, "engine", False) == Path("models/yolov8n-fp32.engine")
    assert exported_name(w, "engine", True) == Path("models/yolov8n-fp16.engine")


def test_summarise_latencies():
    stats = summarise([10.0] * 95 + [50.0] * 5)
    assert stats["mean_ms"] == pytest.approx(12.0)
    assert stats["p50_ms"] == 10.0
    assert stats["p95_ms"] in (10.0, 50.0)
    assert stats["fps"] == pytest.approx(1000 / 12.0)


def test_warmup_calls_are_not_timed():
    class FakeDetector:
        calls = 0

        def detect(self, frame):
            self.calls += 1
            return ["x"]

    fake = FakeDetector()
    times, network_ms, detections = time_detector(fake, grey_frame(), runs=7, warmup=3)
    assert fake.calls == 10 and len(times) == 7 and detections == ["x"]
    assert network_ms == 0.0   # the fake reports no network time


def test_table_reports_speedup_and_failed_backends():
    table = format_table([
        Result("pytorch-fp32", "m.pt", 6.2, 10.0, 10.0, 11.0, 100.0, 7.0, 6),
        Result("tensorrt-fp16", "m.engine", 8.0, 2.5, 2.5, 3.0, 400.0, 1.2, 6),
        Result("onnxruntime", "", 0, 0, 0, 0, 0, 0, 0, error="ImportError: no onnxruntime"),
    ])
    assert "4.00x" in table and "1.00x" in table
    assert "failed: ImportError" in table

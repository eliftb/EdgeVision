"""Frame-rate measurement that starts counting at the first frame.

Counting from program start would fold model loading and GPU warm-up into the
average and report a rate far below what the pipeline actually sustains.
"""

import time
from collections import deque


class FPSMeter:
    def __init__(self, window: int = 30):
        self._recent: deque[float] = deque(maxlen=window)
        self._first: float | None = None
        self._last: float | None = None
        self.frames = 0

    def tick(self, now: float | None = None) -> None:
        """Call once per processed frame."""
        now = time.perf_counter() if now is None else now
        if self._first is None:
            self._first = now
        self._last = now
        self._recent.append(now)
        self.frames += 1

    @property
    def fps(self) -> float:
        """Rate over the last `window` frames; what to show on screen."""
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        return (len(self._recent) - 1) / span if span > 0 else 0.0

    @property
    def average_fps(self) -> float:
        """Rate over the whole run, from the first frame to the last."""
        if self.frames < 2 or self._last == self._first:
            return 0.0
        return (self.frames - 1) / (self._last - self._first)

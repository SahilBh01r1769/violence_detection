"""Shared temporal event state machine for live inference and trace replay."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalDecision:
    triggered: bool
    event_active: bool
    released: bool


class TemporalEventFilter:
    """Qualify an event with N positives and release it with K negatives."""

    def __init__(self, positive_frames: int, negative_frames: int = 1):
        if positive_frames < 1:
            raise ValueError("positive_frames must be at least 1")
        if negative_frames < 1:
            raise ValueError("negative_release_frames must be at least 1")
        self.positive_frames = int(positive_frames)
        self.negative_frames = int(negative_frames)
        self.reset()

    def reset(self) -> None:
        self.positive_run = 0
        self.negative_run = 0
        self.event_active = False

    def update(self, is_positive: bool) -> TemporalDecision:
        triggered = False
        released = False

        if is_positive:
            self.positive_run += 1
            self.negative_run = 0
            if not self.event_active and self.positive_run >= self.positive_frames:
                self.event_active = True
                triggered = True
        else:
            self.positive_run = 0
            if self.event_active:
                self.negative_run += 1
                if self.negative_run >= self.negative_frames:
                    self.event_active = False
                    self.negative_run = 0
                    released = True

        return TemporalDecision(triggered, self.event_active, released)


class RollingWindowEventFilter:
    """Qualify M positives within W frames and release with K negatives.

    The startup window may contain fewer than W observations, but it still
    requires all M positive observations before triggering. Releasing an event
    clears the window, so rearming always requires fresh post-release evidence.
    """

    def __init__(
        self,
        minimum_positives: int,
        window_size: int,
        negative_frames: int = 1,
    ):
        if minimum_positives < 1:
            raise ValueError("minimum_positives must be at least 1")
        if window_size < minimum_positives:
            raise ValueError("window_size must be at least minimum_positives")
        if negative_frames < 1:
            raise ValueError("negative_release_frames must be at least 1")
        self.minimum_positives = int(minimum_positives)
        self.window_size = int(window_size)
        self.negative_frames = int(negative_frames)
        self.reset()

    def reset(self) -> None:
        self._window: deque[bool] = deque(maxlen=self.window_size)
        self.positive_run = 0
        self.negative_run = 0
        self.event_active = False

    @property
    def positive_count(self) -> int:
        return sum(self._window)

    def update(self, is_positive: bool) -> TemporalDecision:
        triggered = False
        released = False
        self._window.append(is_positive)

        if is_positive:
            self.positive_run += 1
            self.negative_run = 0
        else:
            self.positive_run = 0
            if self.event_active:
                self.negative_run += 1

        if not self.event_active and self.positive_count >= self.minimum_positives:
            self.event_active = True
            triggered = True

        if self.event_active and not is_positive:
            if self.negative_run >= self.negative_frames:
                self.event_active = False
                self.negative_run = 0
                self._window.clear()
                released = True

        return TemporalDecision(triggered, self.event_active, released)


def build_temporal_filter(
    strategy: str,
    positive_frames: int,
    negative_frames: int,
    rolling_window_size: int,
):
    normalised = strategy.strip().lower()
    if normalised == "consecutive":
        return TemporalEventFilter(positive_frames, negative_frames)
    if normalised == "rolling_window":
        return RollingWindowEventFilter(
            positive_frames,
            rolling_window_size,
            negative_frames,
        )
    raise ValueError("temporal_strategy must be 'consecutive' or 'rolling_window'")

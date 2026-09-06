"""Shared temporal event state machine for live inference and trace replay."""

from __future__ import annotations

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

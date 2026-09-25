"""Deterministic sample allocation and capture-time windowing."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .common import Record


def allocate(capacities: list[int], total: int) -> list[int]:
    """Even recording coverage, redistributing shortages without duplicates."""
    if total < 0 or any(capacity < 0 for capacity in capacities):
        message = "Sample counts cannot be negative"
        raise ValueError(message)
    if not capacities:
        return []
    quotas = [
        min(capacity, total // len(capacities) + (i < total % len(capacities)))
        for i, capacity in enumerate(capacities)
    ]
    remaining = min(total, sum(capacities)) - sum(quotas)
    while remaining:
        for i, capacity in enumerate(capacities):
            if remaining and quotas[i] < capacity:
                quotas[i] += 1
                remaining -= 1
    return quotas


def spread_bursts(count: int, target: int, burst: int = 3) -> list[int]:
    """Spread short consecutive runs throughout the available sequence."""
    if min(count, target) < 0 or burst <= 0:
        message = "Invalid sample size or burst length"
        raise ValueError(message)
    target = min(count, target)
    if target == count:
        return list(range(count))
    starts = np.linspace(
        0, max(0, count - burst), (target + burst - 1) // burst
    ).astype(int)
    chosen: set[int] = set()
    for start in starts:
        for index in range(start, min(count, start + burst)):
            if len(chosen) < target:
                chosen.add(index)
    for index in range(count):
        if len(chosen) >= target:
            break
        chosen.add(index)
    return sorted(chosen)


@dataclass
class TimeWindows:
    fps: float
    gap_seconds: float
    origin: int | None = None
    previous: int | None = None
    clock: str | None = None
    segment: int = 0
    intervals: list[int] = field(default_factory=list)
    discontinuities: list[Record] = field(default_factory=list)

    def assign(self, timestamp_ns: int, clock: str) -> tuple[int, int] | None:
        if self.previous == timestamp_ns and self.clock == clock:
            return None
        discontinuity = self.previous is not None and (
            timestamp_ns < self.previous
            or timestamp_ns - self.previous > self.gap_seconds * 10**9
            or self.clock != clock
        )
        if discontinuity:
            self.discontinuities.append(
                {
                    "previous_ns": self.previous,
                    "current_ns": timestamp_ns,
                    "previous_clock": self.clock,
                    "current_clock": clock,
                }
            )
        elif self.previous is not None:
            self.intervals.append(timestamp_ns - self.previous)
        if self.origin is None or discontinuity:
            self.origin = timestamp_ns
            self.segment += int(discontinuity)
        self.previous, self.clock = timestamp_ns, clock
        rate = Fraction(str(self.fps))
        bucket = (
            (timestamp_ns - self.origin)
            * rate.numerator
            // (10**9 * rate.denominator)
        )
        return self.segment, bucket

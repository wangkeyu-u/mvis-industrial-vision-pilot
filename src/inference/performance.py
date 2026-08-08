"""Structured, framework-neutral inference performance sampling."""

from __future__ import annotations

import math
import platform
import resource
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class PerformanceSample:
    iteration: int
    latency_ms: float
    peak_memory_mb: float | None


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    available: bool
    reason: str | None
    warmup_runs: int
    requested_runs: int
    measured_runs: int
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    min_latency_ms: float | None
    max_latency_ms: float | None
    peak_memory_mb: float | None
    samples: tuple[PerformanceSample, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [asdict(sample) for sample in self.samples]
        return payload


class PerformanceSampler:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        memory_reader: Callable[[], float | None] | None = None,
    ) -> None:
        self._clock = clock
        self._memory_reader = memory_reader or process_peak_memory_mb

    def run(
        self,
        operation: Callable[[], object],
        *,
        warmup_runs: int = 1,
        measured_runs: int = 5,
    ) -> PerformanceSummary:
        if warmup_runs < 0:
            raise ValueError("warmup_runs must be non-negative")
        if measured_runs <= 0:
            raise ValueError("measured_runs must be positive")
        for _ in range(warmup_runs):
            operation()

        samples: list[PerformanceSample] = []
        for iteration in range(1, measured_runs + 1):
            started = self._clock()
            operation()
            latency_ms = max(0.0, (self._clock() - started) * 1000.0)
            samples.append(
                PerformanceSample(
                    iteration=iteration,
                    latency_ms=round(latency_ms, 3),
                    peak_memory_mb=self._memory_reader(),
                )
            )

        latencies = [sample.latency_ms for sample in samples]
        memory_values = [
            sample.peak_memory_mb
            for sample in samples
            if sample.peak_memory_mb is not None
        ]
        return PerformanceSummary(
            available=True,
            reason=None,
            warmup_runs=warmup_runs,
            requested_runs=measured_runs,
            measured_runs=len(samples),
            p50_latency_ms=_nearest_rank(latencies, 50),
            p95_latency_ms=_nearest_rank(latencies, 95),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            peak_memory_mb=max(memory_values) if memory_values else None,
            samples=tuple(samples),
        )


def unavailable_performance(
    reason: str, *, warmup_runs: int = 0, requested_runs: int = 0
) -> PerformanceSummary:
    if not reason.strip():
        raise ValueError("unavailable performance requires a reason")
    return PerformanceSummary(
        available=False,
        reason=reason,
        warmup_runs=warmup_runs,
        requested_runs=requested_runs,
        measured_runs=0,
        p50_latency_ms=None,
        p95_latency_ms=None,
        min_latency_ms=None,
        max_latency_ms=None,
        peak_memory_mb=None,
        samples=(),
    )


def process_peak_memory_mb() -> float | None:
    try:
        peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    divisor = 1024.0 * 1024.0 if platform.system() == "Darwin" else 1024.0
    return round(peak / divisor, 3)


def _nearest_rank(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil((percentile / 100.0) * len(ordered)))
    return ordered[rank - 1]

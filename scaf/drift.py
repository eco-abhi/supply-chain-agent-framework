"""Drift-triggered retraining (Section 3.3).

Two triggers:
  1. statistical drift (PSI) between a reference window and a live window
  2. repeated A2A negotiation disagreement, treated as evidence of
     model miscalibration

When either fires, the monitor flags drift; the router picks that flag up
via DecisionContext.model_drift_flag and becomes more cautious, while the
owning agent schedules a retrain.
"""
from __future__ import annotations

import math
from collections import deque


def psi(reference: list[float], live: list[float], bins: int = 10) -> float:
    """Population Stability Index between two samples."""
    if not reference or not live:
        return 0.0
    lo = min(min(reference), min(live))
    hi = max(max(reference), max(live))
    if hi == lo:
        return 0.0
    width = (hi - lo) / bins

    def bucket_frac(sample: list[float]) -> list[float]:
        counts = [0] * bins
        for x in sample:
            idx = min(int((x - lo) / width), bins - 1)
            counts[idx] += 1
        n = len(sample)
        # conventional smoothing floor to avoid log(0) without inflating PSI
        return [max(c / n, 1e-4) for c in counts]

    ref_f = bucket_frac(reference)
    live_f = bucket_frac(live)
    return sum((lf - rf) * math.log(lf / rf) for rf, lf in zip(ref_f, live_f))


class DriftMonitor:
    """Tracks prediction drift and A2A disagreement for one model."""

    PSI_THRESHOLD = 0.2          # conventional "significant shift" cutoff
    PSI_BINS = 3                 # fewer bins reduces small-sample noise
    PSI_BATCH_SIZE = 100         # non-overlapping batch; avoids sliding-window autocorrelation
    DISAGREEMENT_WINDOW = 20
    DISAGREEMENT_THRESHOLD = 0.5  # >50% failed negotiations in window

    def __init__(self, name: str, reference: list[float]):
        self.name = name
        self.reference = list(reference)
        self.live: deque[float] = deque(maxlen=self.PSI_BATCH_SIZE)
        self.negotiations: list[bool] = []
        self.retrain_count = 0
        self._last_batch_psi = 0.0
        self._last_disagreement_rate = 0.0
        self.last_batch_mean: float | None = None
        self._last_batch_data: list[float] | None = None

    def observe(self, value: float) -> None:
        self.live.append(value)
        # non-overlapping batches: only recompute PSI once the batch is full,
        # then clear it, so consecutive checks are independent draws rather
        # than a slowly-sliding, autocorrelated window
        if len(self.live) >= self.PSI_BATCH_SIZE:
            batch = list(self.live)
            self._last_batch_psi = psi(self.reference, batch, bins=self.PSI_BINS)
            self.last_batch_mean = sum(batch) / len(batch)
            self._last_batch_data = batch
            self.live.clear()

    def record_negotiation(self, agreed: bool) -> None:
        # same non-overlapping-batch logic as PSI, for the same reason:
        # a sliding window over-counts near-identical, autocorrelated
        # checks and inflates the false-positive rate
        self.negotiations.append(agreed)
        if len(self.negotiations) >= self.DISAGREEMENT_WINDOW:
            failures = sum(1 for a in self.negotiations if not a)
            self._last_disagreement_rate = failures / len(self.negotiations)
            self.negotiations.clear()

    @property
    def current_psi(self) -> float:
        return self._last_batch_psi

    @property
    def disagreement_rate(self) -> float:
        return self._last_disagreement_rate

    @property
    def drifting(self) -> bool:
        stat_drift = self._last_batch_psi > self.PSI_THRESHOLD
        nego_drift = self._last_disagreement_rate > self.DISAGREEMENT_THRESHOLD
        return stat_drift or nego_drift

    def retrain(self) -> None:
        """Retrain: in this reference implementation, re-baseline the
        reference distribution to the last completed batch of live data.
        A real deployment plugs an actual training pipeline in here."""
        if self._last_batch_data:
            self.reference = list(self._last_batch_data)
        self.live.clear()
        self.negotiations.clear()
        self._last_batch_psi = 0.0
        self._last_disagreement_rate = 0.0
        self.retrain_count += 1

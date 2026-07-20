"""Reference-speed profiles used by CARLA and offline controller tests."""

from __future__ import annotations

import random


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


class RandomReferenceSpeed:
    """Generate deterministic random step references for controller evaluation.

    The requested speed changes as a step. The speed planner then turns that step
    into a physically comfortable ramp, so the controller is tested against real
    changes without sending an impossible discontinuity directly to the pedals.
    """

    def __init__(
        self,
        dt: float,
        minimum_speed_kmh: float = 15.0,
        maximum_speed_kmh: float = 55.0,
        minimum_hold_seconds: float = 5.0,
        maximum_hold_seconds: float = 12.0,
        seed: int = 7,
        enabled: bool = True,
        initial_speed_kmh: float | None = None,
    ) -> None:
        self.dt = max(1e-3, float(dt))
        self.minimum_speed_mps = max(0.0, float(minimum_speed_kmh) / 3.6)
        self.maximum_speed_mps = max(
            self.minimum_speed_mps,
            float(maximum_speed_kmh) / 3.6,
        )
        self.minimum_hold_seconds = max(self.dt, float(minimum_hold_seconds))
        self.maximum_hold_seconds = max(
            self.minimum_hold_seconds,
            float(maximum_hold_seconds),
        )
        self.enabled = bool(enabled)
        self.random = random.Random(int(seed))

        if initial_speed_kmh is None:
            initial_speed_mps = 0.5 * (
                self.minimum_speed_mps + self.maximum_speed_mps
            )
        else:
            initial_speed_mps = clamp(
                float(initial_speed_kmh) / 3.6,
                self.minimum_speed_mps,
                self.maximum_speed_mps,
            )

        self.current_speed_mps = initial_speed_mps
        self.elapsed_in_segment_s = 0.0
        self.segment_duration_s = self._sample_duration()
        self.segment_index = 0

    def update(self, fallback_speed_mps: float) -> tuple[float, dict]:
        """Return the current reference and diagnostic information."""
        if not self.enabled:
            value = max(0.0, float(fallback_speed_mps))
            return value, {
                "enabled": False,
                "segment_index": self.segment_index,
                "remaining_seconds": None,
                "changed": False,
                "reference_speed_mps": value,
            }

        self.elapsed_in_segment_s += self.dt
        changed = False
        if self.elapsed_in_segment_s + 1e-9 >= self.segment_duration_s:
            self.current_speed_mps = self._sample_speed(self.current_speed_mps)
            self.elapsed_in_segment_s = 0.0
            self.segment_duration_s = self._sample_duration()
            self.segment_index += 1
            changed = True

        return self.current_speed_mps, {
            "enabled": True,
            "segment_index": self.segment_index,
            "remaining_seconds": max(
                0.0,
                self.segment_duration_s - self.elapsed_in_segment_s,
            ),
            "changed": changed,
            "reference_speed_mps": self.current_speed_mps,
        }

    def _sample_duration(self) -> float:
        return self.random.uniform(
            self.minimum_hold_seconds,
            self.maximum_hold_seconds,
        )

    def _sample_speed(self, previous_speed_mps: float) -> float:
        span = self.maximum_speed_mps - self.minimum_speed_mps
        if span <= 1e-6:
            return self.minimum_speed_mps

        # Avoid tiny changes that are not useful in a step-response experiment.
        minimum_change = min(8.0 / 3.6, 0.35 * span)
        for _ in range(20):
            candidate = self.random.uniform(
                self.minimum_speed_mps,
                self.maximum_speed_mps,
            )
            if abs(candidate - previous_speed_mps) >= minimum_change:
                return candidate

        # Deterministic fallback after unlikely repeated near-equal samples.
        midpoint = 0.5 * (self.minimum_speed_mps + self.maximum_speed_mps)
        return (
            self.maximum_speed_mps
            if previous_speed_mps <= midpoint
            else self.minimum_speed_mps
        )

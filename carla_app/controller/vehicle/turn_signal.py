"""Rota geometrisinden sağ/sol dönüş sinyali üretir."""

from __future__ import annotations

import math


class TurnSignalController:
    """Kavşağa yaklaşırken rotanın yönüne göre sinyal seçer.

    Pure Pursuit'in önündeki rota yön değişimini kullanır. Normal yol
    virajlarında sinyal yakmamak için yalnız kavşak yaklaşımında etkinleşir.
    """

    def __init__(self, dt: float = 0.05) -> None:
        self.dt = max(1e-3, float(dt))
        self.activation_angle_rad = math.radians(10.0)
        self.release_angle_rad = math.radians(3.5)
        self.activation_distance_m = 30.0
        self.minimum_on_ticks = max(1, int(round(1.2 / self.dt)))
        self.release_ticks_required = max(1, int(round(0.6 / self.dt)))
        self.maximum_on_ticks = max(1, int(round(12.0 / self.dt)))

        self.direction = "off"
        self.active_ticks = 0
        self.release_ticks = 0
        self.entered_junction = False

    def update(self, state: dict, lateral_info: dict) -> dict:
        path_turn_rad = float(lateral_info.get("path_turn_rad", 0.0))
        is_junction = bool(state.get("is_junction", False))
        junction_distance = state.get("junction_distance_m")
        approaching_junction = is_junction or (
            junction_distance is not None
            and 0.0 <= float(junction_distance) <= self.activation_distance_m
        )

        requested = "off"
        if approaching_junction:
            if path_turn_rad >= self.activation_angle_rad:
                requested = "right"
            elif path_turn_rad <= -self.activation_angle_rad:
                requested = "left"

        if self.direction == "off":
            if requested != "off":
                self._activate(requested)
        else:
            self.active_ticks += 1
            if is_junction:
                self.entered_junction = True

            if requested != "off" and requested != self.direction:
                self._activate(requested)
            else:
                straight_again = abs(path_turn_rad) <= self.release_angle_rad
                left_junction = not is_junction and (
                    self.entered_junction
                    or junction_distance is None
                    or float(junction_distance) > self.activation_distance_m
                )

                if (
                    self.active_ticks >= self.minimum_on_ticks
                    and straight_again
                    and left_junction
                ):
                    self.release_ticks += 1
                else:
                    self.release_ticks = 0

                if (
                    self.release_ticks >= self.release_ticks_required
                    or self.active_ticks >= self.maximum_on_ticks
                ):
                    self._deactivate()

        return {
            "direction": self.direction,
            "active": self.direction != "off",
            "path_turn_deg": math.degrees(path_turn_rad),
            "junction_distance_m": (
                float(junction_distance)
                if junction_distance is not None
                else None
            ),
            "is_junction": is_junction,
        }

    def _activate(self, direction: str) -> None:
        self.direction = direction
        self.active_ticks = 0
        self.release_ticks = 0
        self.entered_junction = False

    def _deactivate(self) -> None:
        self.direction = "off"
        self.active_ticks = 0
        self.release_ticks = 0
        self.entered_junction = False

"""Top-level vehicle controller."""

import math

import carla

from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import EmergencyBrakeSupervisor
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


class VehicleController:
    """Combine path tracking, speed planning, IDM and independent AEB."""

    def __init__(self, dt=0.05):
        self.lateral = StanleyController(dt)
        self.speed_planner = CurvatureSpeedPlanner(dt)
        self.longitudinal = LongitudinalController(dt)
        self.safety = EmergencyBrakeSupervisor()

    def run_step(self, state, lead_vehicle, emergency_obstacle=None):
        steer = self.lateral.run_step(state)
        lateral_info = self.lateral.last_info
        target_speed, speed_plan = self.speed_planner.run_step(
            state,
            lateral_info=lateral_info,
        )

        control_lead = self._longitudinal_lead(lead_vehicle, emergency_obstacle)
        throttle, brake, longitudinal_info = self.longitudinal.run_step(
            state,
            control_lead,
            target_speed,
        )

        emergency, safety_info = self.safety.evaluate_candidates(
            lead_vehicle,
            emergency_obstacle,
        )
        if emergency:
            self.longitudinal.notify_emergency_stop()
            throttle = 0.0
            brake = 1.0
            longitudinal_info["mode"] = "EMERGENCY"

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
        )
        info = {
            "mode": longitudinal_info["mode"],
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "target_speed_mps": target_speed,
            "lateral": lateral_info,
            "speed_plan": speed_plan,
            "longitudinal": longitudinal_info,
            "safety": safety_info,
            "emergency_obstacle": emergency_obstacle,
            "longitudinal_lead": control_lead,
        }
        return control, info

    @staticmethod
    def _longitudinal_lead(lead_vehicle, emergency_obstacle):
        """Use the nearest valid range while retaining the tracked identity."""
        lead = VehicleController._valid_obstacle(lead_vehicle)
        radar = VehicleController._valid_obstacle(emergency_obstacle)
        if radar is None:
            return lead_vehicle if lead is not None else None
        if lead is None:
            return emergency_obstacle
        if radar["distance_m"] >= lead["distance_m"]:
            return lead_vehicle

        merged = dict(lead_vehicle)
        merged["distance_m"] = radar["distance_m"]
        merged["relative_speed_mps"] = min(
            lead["relative_speed_mps"],
            radar["relative_speed_mps"],
        )
        merged["source"] = f"{merged.get('source', 'tracked')}+radar_near"
        return merged

    @staticmethod
    def _valid_obstacle(obstacle):
        if not isinstance(obstacle, dict):
            return None
        try:
            distance = float(obstacle["distance_m"])
            relative_speed = float(obstacle["relative_speed_mps"])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(distance) or not math.isfinite(relative_speed):
            return None
        if distance <= 0.0:
            return None
        return {
            "distance_m": distance,
            "relative_speed_mps": relative_speed,
        }

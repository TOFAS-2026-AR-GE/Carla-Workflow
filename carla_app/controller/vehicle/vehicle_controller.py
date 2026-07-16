"""Top-level vehicle controller."""

import carla

from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import (
    EmergencyBrakeSupervisor,
)
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


class VehicleController:
    """Combine lane tracking, speed planning, ACC and emergency braking."""

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
        longitudinal_lead = self._longitudinal_lead(
            lead_vehicle,
            emergency_obstacle,
        )
        throttle, brake, longitudinal_info = self.longitudinal.run_step(
            state,
            longitudinal_lead,
            target_speed,
        )
        emergency, safety_info = self.safety.evaluate_candidates(
            lead_vehicle,
            emergency_obstacle,
        )

        if emergency:
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
            "longitudinal_lead": longitudinal_lead,
        }
        return control, info

    @staticmethod
    def _longitudinal_lead(lead_vehicle, emergency_obstacle):
        """Use a closer raw-radar return without losing tracked identity."""
        if emergency_obstacle is None:
            return lead_vehicle
        if lead_vehicle is None:
            return emergency_obstacle

        try:
            lead_distance = float(lead_vehicle["distance_m"])
            radar_distance = float(emergency_obstacle["distance_m"])
            lead_relative_speed = float(lead_vehicle["relative_speed_mps"])
            radar_relative_speed = float(emergency_obstacle["relative_speed_mps"])
        except (KeyError, TypeError, ValueError):
            return lead_vehicle

        if radar_distance >= lead_distance:
            return lead_vehicle

        lead = dict(lead_vehicle)
        lead["distance_m"] = radar_distance
        lead["relative_speed_mps"] = min(
            lead_relative_speed,
            radar_relative_speed,
        )
        lead["source"] = f"{lead.get('source', 'tracked')}+radar_near"
        return lead

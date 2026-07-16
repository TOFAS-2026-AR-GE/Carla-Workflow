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

    def run_step(self, state, lead_vehicle):
        steer = self.lateral.run_step(state)
        lateral_info = self.lateral.last_info

        target_speed, speed_plan = self.speed_planner.run_step(
            state,
            lateral_info=lateral_info,
        )
        throttle, brake, longitudinal_info = self.longitudinal.run_step(
            state,
            lead_vehicle,
            target_speed,
        )
        emergency, safety_info = self.safety.evaluate(lead_vehicle)

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
        }
        return control, info

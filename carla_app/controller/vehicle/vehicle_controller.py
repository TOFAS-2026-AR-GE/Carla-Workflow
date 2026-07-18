"""Direksiyon, hedef hız, araç takibi ve acil freni tek komutta birleştirir.

Girdi olarak araç durumunu ve takip edilen ön aracı alır. Çıktı olarak
CARLA ``VehicleControl`` nesnesi ile anlaşılır tanı bilgilerini verir.
"""

import carla

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.behavior_planner import BehaviorPlanner
from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import EmergencyBrakeSupervisor
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


class VehicleController:
    """Dört bağımsız kontrol parçasını doğru sırayla çalıştırır."""

    def __init__(self, dt=0.05, cruise_speed_kmh=60.0, parameters=None):
        self.parameters = parameters or DrivingParameters(dt)
        self.lateral = StanleyController(dt)
        self.speed_planner = CurvatureSpeedPlanner(dt, cruise_speed_kmh)
        self.behavior = BehaviorPlanner(dt, self.parameters)
        self.longitudinal = LongitudinalController(dt)
        self.safety = EmergencyBrakeSupervisor(self.parameters)

    def run_step(
        self,
        state,
        lead_vehicle,
        emergency_obstacle=None,
        road_context=None,
    ):
        """Tek çevrim için direksiyon, gaz ve fren komutunu üretir."""
        steer = self.lateral.run_step(state)
        lateral_info = self.lateral.last_info
        target_speed, speed_plan = self.speed_planner.run_step(
            state,
            lateral_info=lateral_info,
        )
        behavior = self.behavior.plan(
            state=state,
            curve_target_speed_mps=target_speed,
            speed_plan=speed_plan,
            road_context=road_context,
            lead_vehicle=lead_vehicle,
        )
        target_speed = behavior["target_speed_mps"]

        # Tek bir ham radar noktası yalnızca acil frene gider. Normal takip,
        # kamera-radar takibini veya doğrulanmış radar kümesini kullanır.
        control_lead = self.choose_control_obstacle(
            lead_vehicle,
            behavior.get("control_obstacle"),
        )
        throttle, brake, longitudinal_info = self.longitudinal.run_step(
            state,
            control_lead,
            target_speed,
        )

        emergency, safety_info = self.safety.evaluate_candidates(
            lead_vehicle,
            emergency_obstacle,
            behavior.get("control_obstacle"),
        )
        if emergency:
            self.longitudinal.notify_emergency_stop()
            throttle = 0.0
            brake = 1.0
            longitudinal_info["mode"] = "EMERGENCY"

        throttle, brake, steer, command_info = self.safety.validate_control_command(
            throttle=throttle,
            brake=brake,
            steer=steer,
            target_speed_mps=target_speed,
        )
        safety_info["command_validation"] = command_info

        mode = behavior["mode"]
        if emergency:
            mode = "EMERGENCY"
        elif behavior["mode"] == "FOLLOW_VEHICLE":
            if longitudinal_info["mode"] == "HOLD":
                mode = "FOLLOW_VEHICLE"
        elif behavior["mode"] == "CRUISE" and longitudinal_info["mode"] == "RESTART":
            mode = "CRUISE"

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
        )
        info = {
            "mode": mode,
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
            "behavior": behavior,
            "road_context": road_context or {},
        }
        return control, info

    def choose_control_obstacle(self, lead_vehicle, rule_obstacle):
        """Ön araç ile kural kaynaklı durma noktasından yakın olanı seçer."""
        if lead_vehicle is None:
            return rule_obstacle
        if rule_obstacle is None:
            return lead_vehicle
        try:
            lead_distance = float(lead_vehicle["distance_m"])
            rule_distance = float(rule_obstacle["distance_m"])
        except (KeyError, TypeError, ValueError):
            return rule_obstacle
        return lead_vehicle if lead_distance <= rule_distance else rule_obstacle

"""Direksiyon, hedef hız, araç takibi ve acil freni tek komutta birleştirir.

Girdi olarak araç durumunu ve takip edilen ön aracı alır. Çıktı olarak
CARLA ``VehicleControl`` nesnesi ile anlaşılır tanı bilgilerini verir.
"""

import carla

from carla_app.controller.vehicle.longitudinal_controller import (
    LongitudinalController,
)
from carla_app.controller.vehicle.safety_supervisor import EmergencyBrakeSupervisor
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner
from carla_app.controller.vehicle.stanley_controller import StanleyController


class VehicleController:
    """Dört bağımsız kontrol parçasını doğru sırayla çalıştırır."""

    def __init__(self, dt=0.05, cruise_speed_kmh=60.0):
        self.lateral = StanleyController(dt)
        self.speed_planner = CurvatureSpeedPlanner(dt, cruise_speed_kmh)
        self.longitudinal = LongitudinalController(dt)
        self.safety = EmergencyBrakeSupervisor()

    def run_step(self, state, lead_vehicle, emergency_obstacle=None):
        """Tek çevrim için direksiyon, gaz ve fren komutunu üretir."""
        steer = self.lateral.run_step(state)
        lateral_info = self.lateral.last_info
        target_speed, speed_plan = self.speed_planner.run_step(
            state,
            lateral_info=lateral_info,
        )

        # Tek bir ham radar noktası yalnızca acil frene gider. Normal takip,
        # kamera-radar takibini veya doğrulanmış radar kümesini kullanır.
        control_lead = lead_vehicle
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

"""İsteğe bağlı gerçek CARLA kapalı çevrim sensör kabul testi.

Varsayılan test koşusunda atlanır. CARLA sunucusu açıkken
``RUN_CARLA_SENSOR_ACCEPTANCE=1`` ile çalıştırılır.
"""

import math
import os
import unittest


RUN_LIVE_TEST = os.getenv("RUN_CARLA_SENSOR_ACCEPTANCE", "0") == "1"


@unittest.skipUnless(
    RUN_LIVE_TEST,
    "Canlı CARLA kabul testi için RUN_CARLA_SENSOR_ACCEPTANCE=1 ayarlayın.",
)
class CarlaSensorClosedLoopAcceptanceTests(unittest.TestCase):
    def test_gnss_imu_localized_route_following(self):
        import carla

        from carla_app.config import DrivingParameters, Settings
        from carla_app.controller.vehicle.vehicle_controller import VehicleController
        from carla_app.core.client import CarlaSession
        from carla_app.core.route_manager import PersistentRouteManager
        from carla_app.core.scenario import load_scenario
        from carla_app.core.vehicle import spawn_ego_vehicle
        from carla_app.localization import LocalizationSystem
        from carla_app.sensors.manager import SensorManager
        from carla_app.validation import OracleValidator

        settings = Settings()
        scenario = load_scenario(
            settings.scenario_file,
            settings.fixed_delta_seconds,
        )
        dt = float(scenario.fixed_delta_seconds)
        session = CarlaSession(settings, scenario)
        sensors = None
        vehicle = None

        try:
            _, world = session.open()
            vehicle = spawn_ego_vehicle(
                world,
                settings.vehicle_name,
                role_name="sensor_acceptance_ego",
            )
            sensors = SensorManager(settings)
            sensors.start(world, vehicle, dt)

            parameters = DrivingParameters(dt)
            route_manager = PersistentRouteManager(
                world.get_map(),
                spacing_m=1.0,
                horizon_m=80.0,
                recovery_distance_m=8.0,
                recovery_ticks=max(10, int(round(1.0 / dt))),
            )
            localization = LocalizationSystem(
                world.get_map(),
                sensors.layout,
                dt,
                settings,
            )
            localization.wait_until_ready(
                world,
                sensors,
                settings.localization_startup_timeout_frames,
            )
            controller = VehicleController(
                dt,
                cruise_speed_kmh=30.0,
                parameters=parameters,
            )
            oracle = OracleValidator(enabled=True)

            first_location = None
            last_location = None
            maximum_speed_mps = 0.0
            lost_frames = 0
            last_validation = None

            # 12 saniyelik kapalı çevrim: kontrol durumu yalnız GNSS/IMU EKF'den
            # gelir; CARLA pose/hız yalnız oracle hata metriğinde okunur.
            for _ in range(max(80, int(round(12.0 / dt)))):
                frame_id = world.tick()
                snapshot = sensors.get_bev_snapshot(
                    frame_id,
                    max_age_frames=max(10, parameters.sensor_timeout_frames),
                )
                localization_result = localization.update(frame_id, snapshot)
                state = localization.build_vehicle_state(
                    frame_id,
                    route_manager,
                    vehicle,
                )
                state = sensors.enrich_vehicle_state(state, frame_id)
                road_context = {
                    "perception_age_frames": 0,
                    "sensor_fault": False,
                    "fresh": True,
                    "detections": [],
                    "lead_traffic_light": None,
                    "speed_limit_kmh": 30,
                }
                control, _ = controller.run_step(
                    state,
                    lead_vehicle=None,
                    emergency_obstacle=None,
                    road_context=road_context,
                    navigation_state={
                        "drive_enabled": True,
                        "status": "DRIVING",
                        "target_speed_mps": 8.0,
                    },
                )
                vehicle.apply_control(control)

                location = state["location"]
                if first_location is None:
                    first_location = (float(location.x), float(location.y))
                last_location = (float(location.x), float(location.y))
                maximum_speed_mps = max(maximum_speed_mps, float(state["speed_mps"]))
                if localization_result.get("status") == "LOST":
                    lost_frames += 1
                last_validation = oracle.observe(vehicle, state, road_context)

            vehicle.apply_control(carla.VehicleControl(brake=1.0))
            travelled = math.hypot(
                last_location[0] - first_location[0],
                last_location[1] - first_location[1],
            )
            self.assertGreater(travelled, 8.0)
            self.assertEqual(lost_frames, 0)
            self.assertLess(maximum_speed_mps, 11.0)
            self.assertFalse(last_validation["control_connected"])
            self.assertLess(last_validation["mean_position_error_m"], 4.0)
            self.assertLess(last_validation["mean_heading_error_deg"], 15.0)
        finally:
            if vehicle is not None:
                try:
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))
                except Exception:
                    pass
            if sensors is not None:
                sensors.stop()
            if vehicle is not None:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except Exception:
                    pass
            session.close()


if __name__ == "__main__":
    unittest.main()

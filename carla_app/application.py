"""CARLA application lifecycle and main simulation loop."""

import math

from carla_app.config import Settings
from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.vehicle_controller import VehicleController
from carla_app.core.client import CarlaSession
from carla_app.core.route_manager import PersistentRouteManager
from carla_app.core.scenario import load_scenario
from carla_app.core.spectator import update_spectator
from carla_app.core.state import read_vehicle_state, serializable_vehicle_state
from carla_app.core.traffic import Traffic
from carla_app.core.vehicle import spawn_ego_vehicle
from carla_app.perception.system import PerceptionSystem
from carla_app.perception.worker import PerceptionWorker
from carla_app.sensors.manager import SensorManager
from carla_app.visualization.viewer import PerceptionViewer


class CarlaApplication:
    """Run perception and control in CARLA synchronous mode."""

    def __init__(self):
        self.settings = Settings.load()

    def run(self):
        session = None
        traffic = None
        vehicle = None
        sensors = None
        worker = None
        viewer = None

        try:
            self.settings.check_models()
            scenario = load_scenario(
                self.settings.scenario_file,
                self.settings.fixed_delta_seconds,
            )
            dt = float(scenario.fixed_delta_seconds)
            if dt <= 0.0:
                raise ValueError("fixed_delta_seconds sifirdan buyuk olmali.")

            session = CarlaSession(self.settings, scenario)
            client, world = session.open()
            vehicle = spawn_ego_vehicle(
                world,
                self.settings.vehicle_name,
            )

            route_manager = PersistentRouteManager(
                world.get_map(),
                spacing_m=1.0,
                horizon_m=80.0,
                recovery_distance_m=8.0,
                recovery_ticks=max(10, int(round(1.0 / dt))),
            )

            traffic = Traffic(client, world, scenario)
            traffic.spawn()

            sensors = SensorManager(self.settings)
            sensors.start(world, vehicle)

            perception = PerceptionSystem(self.settings)
            worker = PerceptionWorker(perception)
            viewer = PerceptionViewer()
            controller = VehicleController(dt)
            lead_tracker = LeadVehicleTracker(
                dt=dt,
                image_width=self.settings.camera_width,
                camera_fov_deg=self.settings.camera_fov,
            )

            status_every_frames = max(
                1,
                int(round(self.settings.status_period_seconds / dt)),
            )
            perception_period = max(
                1,
                int(self.settings.perception_every_n_frames),
            )

            print("[INFO] Q, ESC veya pencerenin X dugmesi ile cikis.")
            print(
                "[INFO] Controller: PersistentRoute + filtered Stanley + "
                "time-headway ACC/CBF + AEB"
            )

            first_frame_id = None
            while True:
                frame_id = world.tick()
                if first_frame_id is None:
                    first_frame_id = frame_id

                state = read_vehicle_state(
                    world,
                    vehicle,
                    route_manager=route_manager,
                )
                rgb_image = sensors.get_rgb(frame_id)

                if rgb_image is not None and frame_id % perception_period == 0:
                    worker.submit(frame_id, rgb_image)

                perception_result = worker.get_latest()
                radar_frame_id, radar_points = sensors.get_radar("radar_front_long")
                radar_points = radar_points or []

                lead_vehicle = lead_tracker.update(
                    current_frame_id=frame_id,
                    state=state,
                    perception_result=perception_result,
                    radar_frame_id=radar_frame_id,
                    radar_points=radar_points,
                )
                control, control_info = controller.run_step(state, lead_vehicle)
                vehicle.apply_control(control)
                update_spectator(world, vehicle)

                if self.settings.enable_data_recording:
                    sensors.save_if_needed(
                        frame_id,
                        serializable_vehicle_state(state, control),
                    )

                if frame_id % status_every_frames == 0:
                    print(
                        self._status_message(
                            frame_id=frame_id,
                            state=state,
                            perception_result=perception_result,
                            radar_frame_id=radar_frame_id,
                            radar_points=radar_points,
                            lead_vehicle=lead_vehicle,
                            control_info=control_info,
                        )
                    )

                if not viewer.show(
                    perception_result,
                    fallback_image=rgb_image,
                    fallback_frame_id=frame_id,
                ):
                    break

                if self._runtime_limit_reached(first_frame_id, frame_id, dt):
                    print("[INFO] MAX_RUNTIME_SECONDS sinirina ulasildi.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Kullanici uygulamayi durdurdu.")
        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")
            raise
        finally:
            self._cleanup("viewer", viewer, "close")
            self._cleanup("perception worker", worker, "stop")
            self._cleanup("sensors", sensors, "stop")
            self._cleanup("traffic", traffic, "stop")

            if vehicle is not None:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except Exception as error:
                    print(f"[WARN] Ego arac silinemedi: {error}")

            self._cleanup("CARLA session", session, "close")

    def _runtime_limit_reached(self, first_frame_id, frame_id, dt):
        limit = self.settings.max_runtime_seconds
        if limit <= 0.0:
            return False
        elapsed_simulation_seconds = (int(frame_id) - int(first_frame_id)) * dt
        return elapsed_simulation_seconds >= limit

    @staticmethod
    def _status_message(
        frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
        lead_vehicle,
        control_info,
    ):
        perception_result = perception_result or {}
        vehicles = perception_result.get("vehicles", [])
        errors = perception_result.get("errors", {})
        perception_frame = perception_result.get("frame_id")
        perception_age = (
            int(frame_id) - int(perception_frame)
            if perception_frame is not None
            else "-"
        )

        message = (
            f"[STATUS] frame={frame_id} "
            f"speed={float(state['speed_kmh']):.1f}km/h "
            f"target={float(control_info['target_speed_mps']) * 3.6:.1f}km/h "
            f"mode={control_info['mode']} "
            f"steer={float(control_info['steer']):+.2f} "
            f"throttle={float(control_info['throttle']):.2f} "
            f"brake={float(control_info['brake']):.2f} "
            f"bbox={len(vehicles)} age={perception_age} "
            f"radar={len(radar_points)}@{radar_frame_id}"
        )

        lateral = control_info.get("lateral", {})
        cross_track_error = float(lateral.get("cross_track_error_m", 0.0))
        heading_error = float(lateral.get("heading_error_rad", 0.0))
        message += (
            f" cte={cross_track_error:+.2f}m"
            f" heading={math.degrees(heading_error):+.1f}deg"
        )

        if lead_vehicle is not None:
            message += (
                f" lead={float(lead_vehicle['distance_m']):.1f}m"
                f" rel_v={float(lead_vehicle['relative_speed_mps']):+.2f}m/s"
                f" source={lead_vehicle.get('source', 'unknown')}"
            )
        if errors:
            message += f" detector_errors={','.join(sorted(errors))}"

        return message

    @staticmethod
    def _cleanup(name, instance, method_name):
        if instance is None:
            return
        try:
            getattr(instance, method_name)()
        except Exception as error:
            print(f"[WARN] {name} kapatilamadi: {error}")

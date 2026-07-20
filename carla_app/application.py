"""CARLA application startup, synchronous loop and cleanup."""

from __future__ import annotations

import math
import traceback

from carla_app.config import Settings
from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.reference_speed import RandomReferenceSpeed
from carla_app.controller.vehicle.speed_sign import SpeedSignObserver
from carla_app.controller.vehicle.traffic_light import TrafficLightObserver
from carla_app.controller.vehicle.vehicle_controller import VehicleController
from carla_app.core.client import CarlaSession
from carla_app.core.route_manager import PersistentRouteManager
from carla_app.core.scenario import load_scenario
from carla_app.core.spectator import update_spectator
from carla_app.core.state import read_vehicle_state, serializable_vehicle_state
from carla_app.core.traffic import Traffic
from carla_app.core.vehicle import apply_vehicle_lights, spawn_ego_vehicle
from carla_app.perception.system import PerceptionSystem
from carla_app.perception.worker import PerceptionWorker
from carla_app.sensors.manager import SensorManager
from carla_app.visualization.viewer import PerceptionViewer


class CarlaApplication:
    """Run lane keeping, longitudinal control, perception and visualization."""

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
            vehicle = spawn_ego_vehicle(world, self.settings.vehicle_name)

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
            viewer = PerceptionViewer(
                history_seconds=self.settings.dashboard_history_seconds,
                fixed_delta_seconds=dt,
            )

            controller = VehicleController(
                dt,
                cruise_speed_kmh=self.settings.maximum_speed_kmh,
                follow_gap_m=self.settings.follow_gap_m,
                follow_gap_margin_m=self.settings.follow_gap_margin_m,
            )
            reference_profile = RandomReferenceSpeed(
                dt=dt,
                minimum_speed_kmh=self.settings.reference_min_speed_kmh,
                maximum_speed_kmh=min(
                    self.settings.reference_max_speed_kmh,
                    self.settings.maximum_speed_kmh,
                ),
                minimum_hold_seconds=self.settings.reference_min_hold_seconds,
                maximum_hold_seconds=self.settings.reference_max_hold_seconds,
                seed=self.settings.reference_seed,
                enabled=self.settings.enable_random_reference_speed,
                initial_speed_kmh=self.settings.reference_initial_speed_kmh,
            )

            radar_geometry = sensors.layout.front_radar_geometry
            lead_tracker = LeadVehicleTracker(
                dt=dt,
                image_width=self.settings.camera_width,
                camera_fov_deg=self.settings.camera_fov,
                radar_height_m=radar_geometry["height_above_ground_m"],
                radar_pitch_deg=radar_geometry["pitch_deg"],

                carla_map=world.get_map(),
            )
            traffic_light_observer = TrafficLightObserver(
                dt=dt,
                image_width=self.settings.camera_width,
                image_height=self.settings.camera_height,
                use_ground_truth_state_fallback=(
                    self.settings.traffic_light_ground_truth_fallback
                ),
                maximum_distance_m=self.settings.traffic_light_max_distance_m,
            )
            speed_sign_observer = SpeedSignObserver(
                dt=dt,
                image_width=self.settings.camera_width,
                image_height=self.settings.camera_height,
                maximum_speed_kmh=self.settings.maximum_speed_kmh,
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
                "[INFO] Controller: Pure Pursuit + CARLA şerit merkezi"
                "curve limit + PI/IDM 10m gap + traffic light + TTC/AEB"
            )

            first_frame_id = None
            previous_control_mode = None
            while True:
                frame_id = world.tick()
                if first_frame_id is None:
                    first_frame_id = frame_id

                state = read_vehicle_state(
                    world,
                    vehicle,
                    route_manager=route_manager,
                )
                camera_frame_id, rgb_image = sensors.get_rgb(frame_id)
                if (
                    rgb_image is not None
                    and camera_frame_id is not None
                    and camera_frame_id % perception_period == 0
                ):
                    worker.submit(camera_frame_id, rgb_image)

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
                radar_diagnostics = lead_tracker.get_radar_diagnostics()
                emergency_obstacle = lead_tracker.get_emergency_obstacle()

                profile_speed_mps, reference_info = reference_profile.update(
                    self.settings.maximum_speed_kmh / 3.6
                )
                speed_sign = speed_sign_observer.update(
                    current_frame_id=frame_id,
                    perception_result=perception_result,
                )
                sign_speed_mps = speed_sign.get("speed_limit_mps")
                if sign_speed_mps is not None:
                    reference_speed_mps = float(sign_speed_mps)
                    reference_info = dict(reference_info)
                    reference_info["source"] = "speed_sign"
                    reference_info["reference_speed_mps"] = reference_speed_mps
                else:
                    reference_speed_mps = profile_speed_mps
                    reference_info = dict(reference_info)
                    reference_info["source"] = (
                        "random_profile"
                        if reference_info.get("enabled")
                        else "maximum_speed"
                    )

                if speed_sign.get("changed"):
                    limit_kmh = speed_sign.get("speed_limit_kmh")
                    if limit_kmh is None:
                        print(f"[SPEED SIGN] frame={frame_id} limit cleared")
                    else:
                        print(
                            f"[SPEED SIGN] frame={frame_id} "
                            f"limit={float(limit_kmh):.0f}km/h "
                            f"class={speed_sign.get('class_name')}"
                        )
                elif reference_info.get("changed"):
                    print(
                        f"[REFERENCE] frame={frame_id} "
                        f"speed={reference_speed_mps * 3.6:.1f}km/h "
                        f"segment={reference_info['segment_index']}"
                    )

                traffic_light = traffic_light_observer.update(
                    current_frame_id=frame_id,
                    state=state,
                    vehicle=vehicle,
                    world=world,
                    perception_result=perception_result,
                )
                control, control_info = controller.run_step(
                    state,
                    lead_vehicle,
                    emergency_obstacle=emergency_obstacle,
                    requested_speed_mps=reference_speed_mps,
                    traffic_light=traffic_light,
                )
                control_info["reference_profile"] = reference_info
                control_info["speed_sign"] = speed_sign
                vehicle.apply_control(control)
                apply_vehicle_lights(vehicle, control, control_info)
                update_spectator(world, vehicle)

                current_mode = control_info["mode"]
                if current_mode != previous_control_mode:
                    print(self.control_event_message(frame_id, control_info))
                    previous_control_mode = current_mode

                if self.settings.enable_data_recording:
                    serializable = serializable_vehicle_state(state, control)
                    serializable["controller"] = self.serializable_control_info(
                        control_info
                    )
                    sensors.save_if_needed(frame_id, serializable)

                if frame_id % status_every_frames == 0:
                    print(
                        self.status_message(
                            frame_id=frame_id,
                            camera_frame_id=camera_frame_id,
                            state=state,
                            perception_result=perception_result,
                            radar_frame_id=radar_frame_id,
                            radar_points=radar_points,
                            radar_diagnostics=radar_diagnostics,
                            lead_vehicle=lead_vehicle,
                            control_info=control_info,
                        )
                    )

                if not viewer.show(
                    perception_result,
                    fallback_image=rgb_image,
                    fallback_frame_id=camera_frame_id,
                    current_frame_id=frame_id,
                    state=state,
                    control_info=control_info,
                    lead_vehicle=lead_vehicle,
                    traffic_light=traffic_light,
                ):
                    break

                if self.runtime_limit_reached(first_frame_id, frame_id, dt):
                    print("[INFO] MAX_RUNTIME_SECONDS sinirina ulasildi.")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Kullanici uygulamayi durdurdu.")
        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")
            # Print the original traceback before CARLA actors are destroyed.
            traceback.print_exc()
        finally:
            self.cleanup("viewer", viewer, "close")
            self.cleanup("perception worker", worker, "stop")
            self.cleanup("sensors", sensors, "stop")
            self.cleanup("traffic", traffic, "stop")

            if vehicle is not None:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except Exception as error:
                    print(f"[WARN] Ego arac silinemedi: {error}")
            self.cleanup("CARLA session", session, "close")

    def runtime_limit_reached(self, first_frame_id, frame_id, dt):
        limit = self.settings.max_runtime_seconds
        if limit <= 0.0:
            return False
        elapsed = (int(frame_id) - int(first_frame_id)) * dt
        return elapsed >= limit

    def status_message(
        self,
        frame_id,
        camera_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
        radar_diagnostics,
        lead_vehicle,
        control_info,
    ):
        perception_result = perception_result or {}
        vehicles = perception_result.get("vehicles", [])
        lights = perception_result.get("traffic_lights", [])
        signs = perception_result.get("signs", [])
        errors = perception_result.get("errors", {})
        perception_frame = perception_result.get("frame_id")
        perception_age = (
            int(frame_id) - int(perception_frame)
            if perception_frame is not None
            else "-"
        )

        longitudinal = control_info.get("longitudinal", {})
        speed_plan = control_info.get("speed_plan", {})
        traffic_light = control_info.get("traffic_light", {})
        message = (
            f"[STATUS] frame={frame_id} "
            f"speed={float(state['speed_kmh']):.1f}km/h "
            f"reference={float(control_info.get('requested_speed_mps', control_info.get('target_speed_mps', 0.0))) * 3.6:.1f}km/h "
            f"target={float(control_info['target_speed_mps']) * 3.6:.1f}km/h "
            f"mode={control_info['mode']} "
            f"steer={float(control_info['steer']):+.2f} "
            f"throttle={float(control_info['throttle']):.2f} "
            f"brake={float(control_info['brake']):.2f} "
            f"signal={control_info.get('turn_signal', {}).get('direction', 'off')} "
            f"a_cmd={float(longitudinal.get('acceleration_mps2', 0.0)):+.2f}m/s2 "
            f"camera={camera_frame_id} bbox={len(vehicles)} "
            f"lights={len(lights)} signs={len(signs)} age={perception_age} "
            f"radar={len(radar_points)}/"
            f"{int(radar_diagnostics.get('usable_points', 0))}@{radar_frame_id}"
        )

        lateral = control_info.get("lateral", {})
        message += (
            f" cte={float(lateral.get('cross_track_error_m', 0.0)):+.2f}m"
            f" heading={math.degrees(float(lateral.get('heading_error_rad', 0.0))):+.1f}deg"
            f" curve={float(speed_plan.get('curvature_1pm', 0.0)):.4f}1/m"
            f" speed_reason={speed_plan.get('speed_reason', 'unknown')}"
        )

        if lead_vehicle is not None:
            message += (
                f" lead={float(lead_vehicle['distance_m']):.1f}m"
                f" rel_v={float(lead_vehicle['relative_speed_mps']):+.2f}m/s"
                f" source={lead_vehicle.get('source', 'unknown')}"
            )
        filtered_distance = longitudinal.get("filtered_lead_distance_m")
        desired_gap = longitudinal.get("desired_gap_m")
        if filtered_distance is not None:
            message += f" ctrl_gap={float(filtered_distance):.1f}m"
        if desired_gap is not None:
            message += f" desired_gap={float(desired_gap):.1f}m"

        light_distance = traffic_light.get("distance_m")
        message += f" tl={traffic_light.get('state', 'unknown')}"
        if light_distance is not None:
            message += f"/{float(light_distance):.1f}m"
        if traffic_light.get("requires_stop"):
            message += " tl_stop"

        speed_sign = control_info.get("speed_sign", {})
        if speed_sign.get("speed_limit_kmh") is not None:
            message += f" sign_limit={float(speed_sign['speed_limit_kmh']):.0f}km/h"

        safety = control_info.get("safety", {})
        if safety.get("hazard_count", 0):
            ttc = safety.get("ttc_s")
            message += (
                f" aeb={safety.get('reason')}/{safety.get('target_source')}"
                f" count={safety.get('hazard_count')}"
            )
            if ttc is not None and math.isfinite(float(ttc)):
                message += f" ttc={float(ttc):.2f}s"
        if errors:
            message += f" detector_errors={','.join(sorted(errors))}"
        return message

    def control_event_message(self, frame_id, control_info):
        mode = control_info["mode"]
        message = f"[CONTROL] frame={frame_id} mode={mode}"
        if mode == "EMERGENCY":
            safety = control_info.get("safety", {})
            message += (
                f" reason={safety.get('reason')}"
                f" source={safety.get('target_source')}"
                f" distance={safety.get('distance_m')}m"
            )
        elif "RED" in mode or "YELLOW" in mode:
            light = control_info.get("traffic_light", {})
            message += (
                f" light={light.get('state')}"
                f" distance={light.get('distance_m')}m"
                f" source={light.get('source')}"
            )
        elif mode in ("FOLLOW", "HOLD", "LEAD_FAR", "RESTART"):
            lead = control_info.get("longitudinal_lead") or {}
            message += (
                f" source={lead.get('source', 'unknown')}"
                f" distance={lead.get('distance_m')}m"
            )
        else:
            speed_plan = control_info.get("speed_plan", {})
            message += f" speed_reason={speed_plan.get('speed_reason', 'unknown')}"
        return message

    @staticmethod
    def serializable_control_info(control_info):
        longitudinal = control_info.get("longitudinal", {})
        speed_plan = control_info.get("speed_plan", {})
        traffic_light = control_info.get("traffic_light", {})
        speed_sign = control_info.get("speed_sign", {})
        return {
            "mode": control_info.get("mode"),
            "reference_speed_mps": control_info.get("requested_speed_mps"),
            "target_speed_mps": control_info.get("target_speed_mps"),
            "speed_reason": speed_plan.get("speed_reason"),
            "curvature_1pm": speed_plan.get("curvature_1pm"),
            "desired_acceleration_mps2": longitudinal.get(
                "desired_acceleration_mps2"
            ),
            "acceleration_mps2": longitudinal.get("acceleration_mps2"),
            "desired_gap_m": longitudinal.get("desired_gap_m"),
            "filtered_lead_distance_m": longitudinal.get(
                "filtered_lead_distance_m"
            ),
            "traffic_light_state": traffic_light.get("state"),
            "traffic_light_distance_m": traffic_light.get("distance_m"),
            "speed_sign_limit_kmh": speed_sign.get("speed_limit_kmh"),
            "speed_sign_class_name": speed_sign.get("class_name"),
            "turn_signal": control_info.get("turn_signal", {}).get("direction"),
        }

    @staticmethod
    def cleanup(name, instance, method_name):
        if instance is None:
            return
        try:
            getattr(instance, method_name)()
        except Exception as error:
            print(f"[WARN] {name} kapatilamadi: {error}")

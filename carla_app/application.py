"""CARLA uygulamasını açar, ana döngüyü çalıştırır ve güvenli kapatır."""

import math
import time

from carla_app.bev import BevModule
from carla_app.config import DrivingParameters, Settings
from carla_app.controller.vehicle.lead_vehicle import LeadVehicleTracker
from carla_app.controller.vehicle.vehicle_controller import VehicleController
from carla_app.core.client import CarlaSession
from carla_app.core.performance import (
    AdaptivePerceptionScheduler,
    PerformanceMonitor,
)
from carla_app.core.route_manager import PersistentRouteManager
from carla_app.core.scenario import load_scenario
from carla_app.core.spectator import update_spectator
from carla_app.core.state import read_vehicle_state, serializable_vehicle_state
from carla_app.core.traffic import Traffic
from carla_app.core.vehicle import spawn_ego_vehicle
from carla_app.navigation import NavigationSystem
from carla_app.perception.road_context import RoadContextTracker
from carla_app.perception.system import PerceptionSystem
from carla_app.perception.worker import PerceptionWorker
from carla_app.sensors.manager import SensorManager
from carla_app.visualization.viewer import PerceptionViewer


class CarlaApplication:
    """Sensör, algılama ve kontrol parçalarını doğru sırayla çalıştırır."""

    def __init__(self):
        self.settings = Settings()
        self.dt = None
        self.session = None
        self.world = None
        self.vehicle = None
        self.route_manager = None
        self.navigation = None
        self.traffic = None
        self.sensors = None
        self.worker = None
        self.viewer = None
        self.controller = None
        self.lead_tracker = None
        self.bev_module = None
        self.road_context_tracker = None
        self.status_every_frames = 1
        self.perception_period = 1
        self.previous_control_mode = None
        self.previous_bev_validation_status = None
        self.previous_bev_contribution_track_id = None
        self.performance = None

    def run(self):
        """Uygulamayı açar, ana döngüyü çalıştırır ve her durumda temizler."""
        try:
            self.start()
            self.run_main_loop()
        except KeyboardInterrupt:
            print("\n[INFO] Kullanıcı uygulamayı durdurdu.")
        except Exception as error:
            print(f"[ERROR] {type(error).__name__}: {error}")
            raise
        finally:
            self.shutdown()

    def start(self):
        """Ayarları doğrular ve uygulamanın bütün parçalarını sırayla açar."""
        self.settings.check_models()
        scenario = load_scenario(
            self.settings.scenario_file,
            self.settings.fixed_delta_seconds,
        )
        self.dt = float(scenario.fixed_delta_seconds)
        if self.dt <= 0.0:
            raise ValueError("fixed_delta_seconds sıfırdan büyük olmalı.")

        self.session = CarlaSession(self.settings, scenario)
        client, self.world = self.session.open()
        self.vehicle = spawn_ego_vehicle(
            self.world,
            self.settings.vehicle_name,
            self.settings.ego_role_name,
        )

        self.route_manager = PersistentRouteManager(
            self.world.get_map(),
            spacing_m=1.0,
            horizon_m=110.0,
            recovery_distance_m=8.0,
            recovery_ticks=max(10, int(round(1.0 / self.dt))),
        )
        self.navigation = NavigationSystem(
            self.world.get_map(),
            self.route_manager,
            cruise_speed_kmh=self.settings.navigation_speed_kmh,
            arrival_distance_m=self.settings.navigation_arrival_distance_m,
        )

        self.traffic = Traffic(client, self.world, scenario)
        self.traffic.spawn()

        self.sensors = SensorManager(self.settings)
        self.sensors.start(self.world, self.vehicle, self.dt)

        if self.settings.enable_bev:
            self.bev_module = BevModule(
                self.sensors.layout,
                width=self.settings.camera_width,
                height=self.settings.camera_height,
                fixed_delta_seconds=self.dt,
                update_every_n_frames=(
                    self.settings.bev_update_every_n_frames
                ),
                asynchronous=True,
            )

        perception = PerceptionSystem(self.settings)
        self.worker = PerceptionWorker(perception)
        self.viewer = PerceptionViewer(
            carla_map=self.world.get_map(),
            navigation=self.navigation,
            dashboard_width=self.settings.dashboard_width,
            dashboard_height=self.settings.dashboard_height,
            navigation_render_every_n_frames=(
                self.settings.navigation_render_every_n_frames
            ),
        )
        self.controller = VehicleController(
            self.dt,
            cruise_speed_kmh=self.settings.maximum_speed_kmh,
            parameters=DrivingParameters(self.dt),
        )
        self.road_context_tracker = RoadContextTracker(
            layout=self.sensors.layout,
            parameters=self.controller.parameters,
        )

        radar_geometry = self.sensors.layout.front_radar_geometry
        self.lead_tracker = LeadVehicleTracker(
            dt=self.dt,
            image_width=self.settings.camera_width,
            camera_fov_deg=self.settings.camera_fov,
            radar_height_m=radar_geometry["height_above_ground_m"],
            radar_pitch_deg=radar_geometry["pitch_deg"],
        )

        self.status_every_frames = max(
            1,
            int(round(self.settings.status_period_seconds / self.dt)),
        )
        self.perception_period = max(
            1,
            int(self.settings.perception_every_n_frames),
        )
        self.performance = PerformanceMonitor(self.dt * 1000.0)
        self.perception_scheduler = AdaptivePerceptionScheduler(
            frame_budget_ms=self.dt * 1000.0,
            initial_period=self.perception_period,
            maximum_period=self.settings.maximum_perception_period,
        )

        print("[INFO] Haritada sol tık: hedef seç | ONAYLA: rotayı başlat")
        profile = self.settings.performance_profile
        print(
            f"[INFO] Performans profili: {profile.name} | "
            f"VRAM={profile.free_vram_mb:.0f}/"
            f"{profile.total_vram_mb:.0f} MB | "
            f"YOLO={self.settings.vehicle_image_size}px | "
            f"algilama={self.perception_period} karede bir"
        )
        print("[INFO] Q, ESC veya pencerenin X düğmesi ile çıkış.")
        print(
            "[INFO] Kontrol: Pure Pursuit warm-start + MPC direksiyon + "
            "IDM hız referansı + PID gaz-fren + bağımsız acil fren"
        )
        print(f"[INFO] Sensör modu: {self.settings.sensor_mode}")

    def run_main_loop(self):
        """CARLA dünyasını kare kare ilerletir."""
        first_frame_id = None
        self.previous_control_mode = None

        while True:
            frame_id = self.world.tick()
            if first_frame_id is None:
                first_frame_id = frame_id

            keep_running = self.process_frame(frame_id)
            if not keep_running:
                break

            if self.runtime_limit_reached(first_frame_id, frame_id, self.dt):
                print("[INFO] MAX_RUNTIME_SECONDS sınırına ulaşıldı.")
                break

    def process_frame(self, frame_id):
        """Tek dünya karesinin algılama, kontrol, kayıt ve gösterimini yapar."""
        process_started_at = time.perf_counter()
        state = read_vehicle_state(
            self.world,
            self.vehicle,
            route_manager=self.route_manager,
        )
        navigation_state = self.navigation.update(
            state["location"],
            state["speed_mps"],
        )
        camera_frame_id, rgb_image = self.sensors.get_rgb(frame_id)

        new_perception_frame = (
            rgb_image is not None
            and camera_frame_id % self.perception_period == 0
        )
        if new_perception_frame:
            if self.settings.enable_bev:
                camera_packet = self.sensors.get_bev_camera_packet(frame_id)
                primary_name = self.sensors.layout.primary_camera_name
                camera_packet[primary_name] = {
                    "frame_id": int(camera_frame_id),
                    "age_frames": int(frame_id) - int(camera_frame_id),
                    "data": rgb_image,
                }
                self.worker.submit_cameras(camera_packet, primary_name)
            else:
                self.worker.submit(camera_frame_id, rgb_image)

        perception_result = self.worker.get_latest()
        radar_frame_id, radar_points = self.sensors.get_radar("radar_front_long")
        radar_points = radar_points or []
        lidar_entry = self.sensors.get_lidar(
            frame_id,
            max_age_frames=self.controller.parameters.lidar_maximum_age_frames,
        )
        road_context = self.road_context_tracker.update(
            current_frame_id=frame_id,
            perception_result=perception_result,
            lidar_entry=lidar_entry,
            state=state,
        )

        lead_vehicle = self.lead_tracker.update(
            current_frame_id=frame_id,
            state=state,
            perception_result=perception_result,
            radar_frame_id=radar_frame_id,
            radar_points=radar_points,
        )
        radar_diagnostics = self.lead_tracker.get_radar_diagnostics()
        emergency_obstacle = self.lead_tracker.get_emergency_obstacle()
        bev_validation = None
        bev_contribution = None
        if self.bev_module is not None:
            bev_validation = self.bev_module.validate(
                frame_id,
                lead_vehicle=lead_vehicle,
                emergency_obstacle=emergency_obstacle,
            )
            bev_contribution = self.bev_module.contribute(
                frame_id,
                state,
                lead_vehicle=lead_vehicle,
            )
            if bev_contribution["applied"]:
                lead_vehicle = bev_contribution["lead_vehicle"]

        control, control_info = self.controller.run_step(
            state,
            lead_vehicle,
            emergency_obstacle=emergency_obstacle,
            road_context=road_context,
            navigation_state=navigation_state,
        )
        if bev_validation is not None:
            control_info["bev_validation"] = bev_validation
        if bev_contribution is not None:
            control_info["bev_contribution"] = bev_contribution
        self.vehicle.apply_control(control)
        update_spectator(self.world, self.vehicle)

        current_mode = control_info["mode"]
        if current_mode != self.previous_control_mode:
            print(self.control_event_message(frame_id, control_info))
            self.previous_control_mode = current_mode

        if bev_validation is not None:
            validation_status = bev_validation["status"]
            if validation_status != self.previous_bev_validation_status:
                print(
                    f"[BEV-VALIDATION] frame={frame_id} "
                    f"status={validation_status} "
                    f"health={bev_validation['health']['status']} "
                    f"lead={bev_validation['lead']['status']}"
                )
                self.previous_bev_validation_status = validation_status

        contribution_track_id = None
        if bev_contribution is not None and bev_contribution["applied"]:
            contribution_track_id = bev_contribution["track_id"]
        if contribution_track_id != self.previous_bev_contribution_track_id:
            if contribution_track_id is None:
                print(f"[BEV-CONTROL] frame={frame_id} recovery=released")
            else:
                recovered_lead = bev_contribution["lead_vehicle"]
                print(
                    f"[BEV-CONTROL] frame={frame_id} recovery=active "
                    f"track={contribution_track_id} "
                    f"distance={recovered_lead['distance_m']:.1f}m"
                )
            self.previous_bev_contribution_track_id = contribution_track_id

        if self.settings.enable_data_recording:
            self.sensors.save_if_needed(
                frame_id,
                serializable_vehicle_state(state, control),
            )

        if frame_id % self.status_every_frames == 0:
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
                    road_context=road_context,
                )
            )

        bev_image = None
        if self.bev_module is not None:
            sensor_snapshot = self.sensors.get_bev_snapshot(frame_id)
            self.bev_module.submit(
                sensor_snapshot=sensor_snapshot,
                perception_result=perception_result,
                vehicle_state=state,
                current_frame_id=frame_id,
                driving_state=control_info,
                display_mode=self.viewer.bev_mode,
            )
            bev_image = self.bev_module.get_latest()

        viewer_started_at = time.perf_counter()
        keep_running = self.viewer.show(
            perception_result,
            fallback_image=rgb_image,
            fallback_frame_id=camera_frame_id,
            current_frame_id=frame_id,
            bev_image=bev_image,
            road_context=road_context,
            vehicle_state=state,
            navigation_state=navigation_state,
        )
        viewer_ms = (time.perf_counter() - viewer_started_at) * 1000.0
        process_ms = (time.perf_counter() - process_started_at) * 1000.0
        worker_diagnostics = self.worker.get_diagnostics()
        self.performance.update(
            process_ms=process_ms,
            viewer_ms=viewer_ms,
            camera_wait_ms=self.sensors.last_camera_wait_ms,
            perception_result=perception_result,
            worker_diagnostics=worker_diagnostics,
        )
        new_period = self.perception_scheduler.update(
            inference_ms=self.performance.values.get("inference_ms", 0.0),
            worker_diagnostics=worker_diagnostics,
        )
        if new_period is not None and new_period != self.perception_period:
            self.perception_period = new_period
            print(
                "[PERFORMANCE] Algilama araligi "
                f"{self.perception_period} karede bir olarak ayarlandi."
            )
        return keep_running

    def shutdown(self):
        """Açılmış parçaları ters sırayla kapatır."""
        self.close_component("görüntü penceresi", self.viewer)
        self.stop_component("BEV işçisi", self.bev_module)
        self.stop_component("algılama işçisi", self.worker)
        self.stop_component("sensörler", self.sensors)
        self.stop_component("trafik", self.traffic)

        if self.vehicle is not None:
            try:
                if self.vehicle.is_alive:
                    self.vehicle.destroy()
            except Exception as error:
                print(f"[WARN] Ego araç silinemedi: {error}")

        self.close_component("CARLA oturumu", self.session)

    def runtime_limit_reached(self, first_frame_id, frame_id, dt):
        limit = self.settings.max_runtime_seconds
        if limit <= 0.0:
            return False
        elapsed_simulation_seconds = (int(frame_id) - int(first_frame_id)) * dt
        return elapsed_simulation_seconds >= limit

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
        road_context=None,
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
            f"camera={camera_frame_id} "
            f"bbox={len(vehicles)} age={perception_age} "
            f"radar={len(radar_points)}/"
            f"{int(radar_diagnostics.get('usable_points', 0))}@{radar_frame_id}"
        )

        camera_wait_ms = float(
            getattr(self.sensors, "last_camera_wait_ms", 0.0)
        )
        message += (
            f" camera_wait={camera_wait_ms:.1f}ms"
            f" perception={float(perception_result.get('elapsed_ms', 0.0)):.1f}ms"
            f" queue={float(perception_result.get('queue_delay_ms', 0.0)):.1f}ms"
        )
        if self.performance is not None:
            message += self.performance.summary()
        lane_detection = perception_result.get("lane_detection", {})
        if lane_detection.get("available"):
            message += (
                f" lanes={int(lane_detection.get('detected_count', 0))}"
                f" lane_ms={float(lane_detection.get('elapsed_ms', 0.0)):.1f}"
            )

        radar_age = radar_diagnostics.get("frame_age")
        message += f" radar_age={radar_age if radar_age is not None else '-'}"
        if not radar_diagnostics.get("fresh", False):
            message += " radar_stale"

        ground_rejected = int(radar_diagnostics.get("ground_rejected", 0))
        if ground_rejected:
            message += f" ground={ground_rejected}"

        lateral = control_info.get("lateral", {})
        cross_track_error = float(lateral.get("cross_track_error_m", 0.0))
        heading_error = float(lateral.get("heading_error_rad", 0.0))
        message += (
            f" cte={cross_track_error:+.2f}m"
            f" heading={math.degrees(heading_error):+.1f}deg"
        )
        lateral_controller = lateral.get("controller", "pure_pursuit")
        message += f" lateral={lateral_controller}"
        if lateral.get("mpc_active"):
            message += (
                f" mpc={float(lateral.get('mpc_solve_ms', 0.0)):.1f}ms"
                f" mpc_iter={int(lateral.get('mpc_iterations', 0))}"
            )
        elif lateral.get("fallback_reason"):
            message += f" mpc_fallback={lateral.get('fallback_reason')}"
        lookahead = lateral.get("lookahead_m")
        if lookahead is not None:
            message += f" lookahead={float(lookahead):.1f}m"

        speed_plan = control_info.get("speed_plan", {})
        speed_reason = speed_plan.get("speed_reason", "unknown")
        message += f" speed_reason={speed_reason}"
        idm = control_info.get("idm", {})
        if idm:
            message += (
                f" idm_ref={float(idm.get('reference_speed_mps', 0.0)) * 3.6:.1f}km/h"
                f" idm_a={float(idm.get('idm_acceleration_mps2', 0.0)):+.2f}m/s2"
            )
            desired_gap = idm.get("desired_gap_m")
            if desired_gap is not None:
                message += f" idm_gap={float(desired_gap):.1f}m"
        if speed_reason in {"curve", "lane_recovery"}:
            curvature = float(speed_plan.get("curvature_1pm", 0.0))
            curve_distance = float(speed_plan.get("curve_distance_m", 0.0))
            yaw_rate = float(speed_plan.get("predicted_yaw_rate_radps", 0.0))
            lateral_acceleration = float(
                speed_plan.get("predicted_lateral_acceleration_mps2", 0.0)
            )
            longitudinal_acceleration = float(
                speed_plan.get("planned_longitudinal_acceleration_mps2", 0.0)
            )
            message += (
                f" curve_k={curvature:.4f}/m"
                f" curve_at={curve_distance:.1f}m"
                f" yaw_rate={yaw_rate:.2f}rad/s"
                f" lat_a={lateral_acceleration:.2f}m/s2"
                f" long_a={longitudinal_acceleration:+.2f}m/s2"
            )

        road_context = road_context or {}
        light = road_context.get("lead_traffic_light")
        primary = control_info.get("behavior", {}).get("primary_detection")
        if isinstance(primary, dict) and primary.get("state_source") == (
            "carla_vehicle_state"
        ):
            light = primary
        elif light is None:
            if isinstance(primary, dict) and primary.get("color") in {
                "red",
                "orange",
                "green",
            }:
                light = primary
        if light is not None:
            message += (
                f" light={light.get('color', 'unknown')}"
                f"@{light.get('estimated_distance_m')}m"
                f" light_source={light.get('state_source', 'unknown')}"
            )
            observed_color = light.get("observed_color")
            candidate_color = light.get("candidate_color")
            candidate_hits = light.get("candidate_hits")
            if observed_color and observed_color != light.get("color"):
                message += f" light_seen={observed_color}"
            if candidate_color and candidate_color != light.get("color"):
                message += f" light_candidate={candidate_color}/{candidate_hits}"
        simulator_light = state.get("simulator_traffic_light", {})
        if simulator_light.get("affected"):
            message += f" sim_light={simulator_light.get('color', 'unknown')}"
        speed_limit = road_context.get("speed_limit_kmh")
        if speed_limit is not None:
            message += f" limit={speed_limit}km/h"
        pedestrian_risk = road_context.get("pedestrian_risk", "NONE")
        if pedestrian_risk != "NONE":
            message += f" pedestrian={pedestrian_risk}"
        lidar = road_context.get("lidar", {})
        if lidar.get("available"):
            message += f" lidar_age={lidar.get('age_frames')}"

        if lead_vehicle is not None:
            message += (
                f" lead={float(lead_vehicle['distance_m']):.1f}m"
                f" rel_v={float(lead_vehicle['relative_speed_mps']):+.2f}m/s"
                f" source={lead_vehicle.get('source', 'unknown')}"
            )
        filtered_distance = control_info.get("longitudinal", {}).get(
            "filtered_lead_distance_m"
        )
        if filtered_distance is not None:
            message += f" ctrl_gap={float(filtered_distance):.1f}m"
        emergency_obstacle = control_info.get("emergency_obstacle")
        if emergency_obstacle is not None:
            message += f" aeb_radar={float(emergency_obstacle['distance_m']):.1f}m"
        safety = control_info.get("safety", {})
        if safety.get("hazard_count", 0):
            ttc = safety.get("ttc_s")
            message += (
                f" aeb={safety.get('reason')}"
                f"/{safety.get('target_source')}"
                f" count={safety.get('hazard_count')}"
                f" sensor_frame={safety.get('measurement_frame_id')}"
            )
            if ttc is not None and math.isfinite(float(ttc)):
                message += f" ttc={float(ttc):.2f}s"
        bev_validation = control_info.get("bev_validation")
        if bev_validation is not None:
            message += (
                f" bev={bev_validation.get('status', 'UNAVAILABLE')}"
                f" bev_lead={bev_validation.get('lead', {}).get('status', 'UNKNOWN')}"
            )
        bev_contribution = control_info.get("bev_contribution")
        if bev_contribution is not None and bev_contribution.get("applied"):
            message += (
                f" bev_ctrl=track#{bev_contribution.get('track_id')}"
            )
        if errors:
            message += f" detector_errors={','.join(sorted(errors))}"

        return message

    def control_event_message(self, frame_id, control_info):
        """Kontrol modunun neden değiştiğini tek, okunaklı satırda anlatır."""
        mode = control_info["mode"]
        message = f"[CONTROL] frame={frame_id} mode={mode}"
        if mode in ("EMERGENCY", "EMERGENCY_BRAKE"):
            safety = control_info.get("safety", {})
            message += (
                f" reason={safety.get('reason')}"
                f" source={safety.get('target_source')}"
                f" distance={safety.get('distance_m')}m"
                f" sensor_frame={safety.get('measurement_frame_id')}"
            )
        elif mode in ("FOLLOW", "FOLLOW_VEHICLE", "HOLD", "LEAD_FAR", "RESTART"):
            lead = control_info.get("longitudinal_lead") or {}
            message += (
                f" source={lead.get('source', 'unknown')}"
                f" distance={lead.get('distance_m')}m"
            )
        else:
            speed_plan = control_info.get("speed_plan", {})
            message += f" speed_reason={speed_plan.get('speed_reason', 'unknown')}"
        return message

    def stop_component(self, name, component):
        """`stop` metodu olan bir parçayı hata yaymadan durdurur."""
        if component is None:
            return
        try:
            component.stop()
        except Exception as error:
            print(f"[WARN] {name} durdurulamadı: {error}")

    def close_component(self, name, component):
        """`close` metodu olan bir parçayı hata yaymadan kapatır."""
        if component is None:
            return
        try:
            component.close()
        except Exception as error:
            print(f"[WARN] {name} kapatılamadı: {error}")

import math
import traceback

from carla_app.config import Settings
from carla_app.controller.vehicle.lead_vehicle import (
    LeadVehicleTracker,
)
from carla_app.controller.vehicle.vehicle_controller import (
    VehicleController,
)
from carla_app.core.client import CarlaSession
from carla_app.core.route_manager import (
    PersistentRouteManager,
)
from carla_app.core.scenario import load_scenario
from carla_app.core.spectator import (
    update_spectator,
)
from carla_app.core.state import (
    read_vehicle_state,
)
from carla_app.core.traffic import Traffic
from carla_app.core.vehicle import (
    spawn_ego_vehicle,
)
from carla_app.perception.system import (
    PerceptionSystem,
)
from carla_app.perception.worker import (
    PerceptionWorker,
)
from carla_app.sensors.manager import (
    SensorManager,
)
from carla_app.visualization.viewer import (
    PerceptionViewer,
)


class CarlaApplication:
    """
    CARLA ana uygulama dongusu.

    Akis:
        world.tick()
        -> state ve sensor verileri
        -> kamera/radar lead vehicle secimi
        -> yanal + boylamsal kontrol
        -> vehicle.apply_control()
        -> goruntuleme ve diagnostik

    Lead vehicle kontrol komutundan once hesaplanir.
    Boylece kontrolcu mevcut tick'in radar bilgisini kullanir.
    """

    CONTROL_STATUS_PERIOD_FRAMES = 20

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

            dt = float(
                scenario.fixed_delta_seconds
            )

            if dt <= 0.0:
                raise ValueError(
                    "scenario.fixed_delta_seconds "
                    "sifirdan buyuk olmali."
                )

            session = CarlaSession(
                self.settings,
                scenario,
            )

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
                recovery_ticks=max(
                    10,
                    int(
                        round(
                            1.0 / dt
                        )
                    ),
                ),
            )

            traffic = Traffic(
                client,
                world,
                scenario,
            )

            traffic.spawn()

            sensors = SensorManager(
                self.settings
            )

            sensors.start(
                world,
                vehicle,
            )

            controller = VehicleController(
                dt
            )

            lead_tracker = LeadVehicleTracker(
                dt=dt,
                image_width=(
                    self.settings.camera_width
                ),
                camera_fov_deg=(
                    self.settings.camera_fov
                ),
            )

            perception = PerceptionSystem(
                self.settings
            )

            worker = PerceptionWorker(
                perception
            )

            viewer = PerceptionViewer()

            perception_every_n_frames = max(
                1,
                int(
                    self.settings
                    .perception_every_n_frames
                ),
            )

            print(
                "[INFO] Q veya ESC ile cikis."
            )

            print(
                "[INFO] Vehicle device: "
                f"{self.settings.vehicle_device}"
            )

            print(
                "[INFO] Sign device: "
                f"{self.settings.sign_device}"
            )

            print(
                "[INFO] Controller: "
                "PersistentRoute + Stanley + ACC/CBF"
            )

            print(
                "[INFO] Simulation dt: "
                f"{dt:.3f} s"
            )

            while True:
                frame_id = world.tick()

                state = read_vehicle_state(
                    world,
                    vehicle,
                    route_manager=route_manager,
                )

                # Sensor verilerini kontrol komutundan
                # once al.
                rgb_image = sensors.get_rgb(
                    frame_id
                )

                if (
                    rgb_image is not None
                    and frame_id
                    % perception_every_n_frames
                    == 0
                ):
                    worker.submit(
                        frame_id,
                        rgb_image,
                    )

                # Worker asenkron calisir.
                # Burada son tamamlanmis sonuc alinir.
                perception_result = (
                    worker.get_latest()
                )

                (
                    radar_frame_id,
                    radar_points,
                ) = sensors.get_radar(
                    "radar_front_long"
                )

                radar_points = (
                    radar_points or []
                )

                # Lead arac, kontrol komutundan
                # once hesaplanir.
                #
                # Guncellenmis LeadVehicleTracker,
                # bbox bulunmasa bile radar_direct
                # yolu ile lead uretebilir.
                lead_vehicle = (
                    lead_tracker.update(
                        current_frame_id=(
                            frame_id
                        ),
                        state=state,
                        perception_result=(
                            perception_result
                        ),
                        radar_frame_id=(
                            radar_frame_id
                        ),
                        radar_points=(
                            radar_points
                        ),
                    )
                )

                control, control_info = (
                    controller.run_step(
                        state,
                        lead_vehicle,
                    )
                )

                vehicle.apply_control(
                    control
                )

                update_spectator(
                    world,
                    vehicle,
                )

                if (
                    frame_id
                    % self
                    .CONTROL_STATUS_PERIOD_FRAMES
                    == 0
                ):
                    self._print_sensor_status(
                        frame_id=frame_id,
                        perception_result=(
                            perception_result
                        ),
                        radar_frame_id=(
                            radar_frame_id
                        ),
                        radar_points=(
                            radar_points
                        ),
                        lead_vehicle=(
                            lead_vehicle
                        ),
                    )

                    self._print_control_status(
                        state=state,
                        lead_vehicle=(
                            lead_vehicle
                        ),
                        control_info=(
                            control_info
                        ),
                    )

                if not viewer.show(
                    perception_result,
                    rgb_image,
                    frame_id,
                ):
                    break

        except KeyboardInterrupt:
            print(
                "\n[INFO] Kullanici durdurdu."
            )

        except Exception as error:
            print(
                "[ERROR] "
                f"{type(error).__name__}: "
                f"{error}"
            )

            traceback.print_exc()

        finally:
            self._safe_close_viewer(
                viewer
            )

            self._safe_stop_worker(
                worker
            )

            self._safe_stop_sensors(
                sensors
            )

            self._safe_stop_traffic(
                traffic
            )

            self._safe_destroy_vehicle(
                vehicle
            )

            self._safe_close_session(
                session
            )

    @staticmethod
    def _print_sensor_status(
        frame_id,
        perception_result,
        radar_frame_id,
        radar_points,
        lead_vehicle,
    ):
        vehicle_count = 0
        sign_count = 0
        perception_age = None
        elapsed_ms = None

        if perception_result is not None:
            vehicle_count = len(
                perception_result.get(
                    "vehicles",
                    [],
                )
            )

            sign_count = len(
                perception_result.get(
                    "signs",
                    [],
                )
            )

            perception_frame = (
                perception_result.get(
                    "frame_id"
                )
            )

            if perception_frame is not None:
                perception_age = (
                    int(frame_id)
                    - int(perception_frame)
                )

            elapsed_ms = (
                perception_result.get(
                    "elapsed_ms"
                )
            )

        lead_source = "none"

        if lead_vehicle is not None:
            lead_source = str(
                lead_vehicle.get(
                    "source",
                    "unknown",
                )
            )

        message = (
            "[SENSORS] "
            f"frame={frame_id} "
            f"bbox={vehicle_count} "
            f"signs={sign_count} "
            f"perception_age="
            f"{perception_age} "
            f"radar_frame="
            f"{radar_frame_id} "
            f"radar_points="
            f"{len(radar_points)} "
            f"lead_source="
            f"{lead_source}"
        )

        if elapsed_ms is not None:
            message += (
                f" inference="
                f"{float(elapsed_ms):.1f}ms"
            )

        print(message)

    @staticmethod
    def _print_control_status(
        state,
        lead_vehicle,
        control_info,
    ):
        if control_info is None:
            return

        lateral = control_info.get(
            "lateral",
            {},
        )

        longitudinal = control_info.get(
            "longitudinal",
            {},
        )

        safety = control_info.get(
            "safety",
            {},
        )

        cross_track_error = float(
            lateral.get(
                "cross_track_error_m",
                0.0,
            )
        )

        heading_error_rad = float(
            lateral.get(
                "heading_error_rad",
                0.0,
            )
        )

        curvature = float(
            lateral.get(
                "curvature_1pm",
                0.0,
            )
        )

        target_speed_mps = float(
            control_info.get(
                "target_speed_mps",
                0.0,
            )
        )

        mode = str(
            control_info.get(
                "mode",
                "UNKNOWN",
            )
        )

        steer = float(
            control_info.get(
                "steer",
                0.0,
            )
        )

        throttle = float(
            control_info.get(
                "throttle",
                0.0,
            )
        )

        brake = float(
            control_info.get(
                "brake",
                0.0,
            )
        )

        message = (
            "[CONTROL] "
            f"mode={mode} "
            f"speed="
            f"{float(state['speed_kmh']):.1f}"
            f"km/h "
            f"target="
            f"{target_speed_mps * 3.6:.1f}"
            f"km/h "
            f"steer={steer:+.2f} "
            f"throttle={throttle:.2f} "
            f"brake={brake:.2f} "
            f"cte="
            f"{cross_track_error:+.2f}m "
            f"heading="
            f"{math.degrees(heading_error_rad):+.1f}"
            f"deg "
            f"curve={curvature:+.4f}1/m "
            f"lane="
            f"{state.get('road_id', '-')}:"
            f"{state.get('lane_id', '-')}"
        )

        if lead_vehicle is not None:
            lead_distance = float(
                lead_vehicle.get(
                    "distance_m",
                    0.0,
                )
            )

            lead_lateral = float(
                lead_vehicle.get(
                    "lateral_m",
                    0.0,
                )
            )

            relative_speed = float(
                lead_vehicle.get(
                    "relative_speed_mps",
                    0.0,
                )
            )

            message += (
                f" lead_id="
                f"{lead_vehicle.get('track_id', '-')}"
                f" lead={lead_distance:.1f}m"
                f" lead_lat="
                f"{lead_lateral:+.2f}m"
                f" rel_v="
                f"{relative_speed:+.2f}m/s"
                f" source="
                f"{lead_vehicle.get('source', 'unknown')}"
            )

        desired_gap = (
            longitudinal.get(
                "desired_gap_m"
            )
        )

        gap_error = (
            longitudinal.get(
                "gap_error_m"
            )
        )

        activation_distance = (
            longitudinal.get(
                "activation_distance_m"
            )
        )

        if desired_gap is not None:
            message += (
                f" gap_ref="
                f"{float(desired_gap):.1f}m"
            )

        if gap_error is not None:
            message += (
                f" gap_err="
                f"{float(gap_error):+.1f}m"
            )

        if activation_distance is not None:
            message += (
                f" acc_on<="
                f"{float(activation_distance):.1f}m"
            )

        ttc = safety.get(
            "ttc_s"
        )

        if (
            ttc is not None
            and math.isfinite(
                float(ttc)
            )
        ):
            message += (
                f" ttc="
                f"{float(ttc):.2f}s"
            )

        print(message)

    @staticmethod
    def _safe_close_viewer(
        viewer,
    ):
        if viewer is None:
            return

        try:
            viewer.close()

        except Exception as error:
            print(
                "[WARN] Viewer kapatilamadi: "
                f"{error}"
            )

    @staticmethod
    def _safe_stop_worker(
        worker,
    ):
        if worker is None:
            return

        try:
            worker.stop()

        except Exception as error:
            print(
                "[WARN] Perception worker "
                "durdurulamadi: "
                f"{error}"
            )

    @staticmethod
    def _safe_stop_sensors(
        sensors,
    ):
        if sensors is None:
            return

        try:
            sensors.stop()

        except Exception as error:
            print(
                "[WARN] Sensorler "
                "durdurulamadi: "
                f"{error}"
            )

    @staticmethod
    def _safe_stop_traffic(
        traffic,
    ):
        if traffic is None:
            return

        try:
            traffic.stop()

        except Exception as error:
            print(
                "[WARN] Trafik "
                "temizlenemedi: "
                f"{error}"
            )

    @staticmethod
    def _safe_destroy_vehicle(
        vehicle,
    ):
        if vehicle is None:
            return

        try:
            if vehicle.is_alive:
                vehicle.destroy()

        except Exception as error:
            print(
                "[WARN] Ego arac "
                "silinemedi: "
                f"{error}"
            )

    @staticmethod
    def _safe_close_session(
        session,
    ):
        if session is None:
            return

        try:
            session.close()

        except Exception as error:
            print(
                "[WARN] CARLA oturumu "
                "kapatilamadi: "
                f"{error}"
            )
from carla_app.config import Settings
from carla_app.controller.vehicle.lead_vehicle import (
    LeadVehicleTracker,
)
from carla_app.controller.vehicle.vehicle_controller import (
    VehicleController,
)
from carla_app.core.client import CarlaSession
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

            session = CarlaSession(
                self.settings,
                scenario,
            )

            client, world = session.open()

            vehicle = spawn_ego_vehicle(
                world,
                self.settings.vehicle_name,
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
                scenario.fixed_delta_seconds
            )

            lead_tracker = LeadVehicleTracker(
                dt=(
                    scenario
                    .fixed_delta_seconds
                ),
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

            # İlk tick'te henüz öndeki araç bilgisi yok.
            lead_vehicle = None
            control_info = None

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

            while True:
                frame_id = world.tick()

                state = read_vehicle_state(
                    world,
                    vehicle,
                )

                # Bir önceki tick'te hesaplanan
                # lead_vehicle bilgisi kullanılır.
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

                rgb_image = sensors.get_rgb(
                    frame_id
                )

                if (
                    rgb_image is not None

                    and frame_id
                    % (
                        self.settings
                        .perception_every_n_frames
                    )
                    == 0
                ):
                    worker.submit(
                        frame_id,
                        rgb_image,
                    )

                perception_result = (
                    worker.get_latest()
                )

                radar_frame_id, radar_points = (
                    sensors.get_radar(
                        "radar_front_long"
                    )
                )

                # Burada üretilen araç bilgisi
                # bir sonraki kontrol tick'inde kullanılır.
                lead_vehicle = lead_tracker.update(
                    current_frame_id=frame_id,
                    state=state,
                    perception_result=(
                        perception_result
                    ),
                    radar_frame_id=(
                        radar_frame_id
                    ),
                    radar_points=radar_points,
                )

                if frame_id % 20 == 0:
                    self._print_control_status(
                        state,
                        lead_vehicle,
                        control_info,
                    )

                if not viewer.show(
                    worker.get_latest(),
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
                f"[ERROR] {error}"
            )

        finally:
            if viewer is not None:
                viewer.close()

            if worker is not None:
                worker.stop()

            if sensors is not None:
                sensors.stop()

            if traffic is not None:
                traffic.stop()

            if vehicle is not None:
                try:
                    vehicle.destroy()
                except Exception:
                    pass

            if session is not None:
                session.close()

    @staticmethod
    def _print_control_status(
        state,
        lead_vehicle,
        control_info,
    ):
        if control_info is None:
            return

        message = (
            f"[CONTROL] "
            f"mode={control_info['mode']} "

            f"speed="
            f"{state['speed_kmh']:.1f} km/h "

            f"target="
            f"{control_info['target_speed_mps'] * 3.6:.1f} km/h "

            f"steer="
            f"{control_info['steer']:+.2f} "

            f"throttle="
            f"{control_info['throttle']:.2f} "

            f"brake="
            f"{control_info['brake']:.2f}"
        )

        if lead_vehicle is not None:
            message += (
                f" lead="
                f"{lead_vehicle['distance_m']:.1f} m"

                f" rel_v="
                f"{lead_vehicle['relative_speed_mps']:+.1f} m/s"
            )

        print(message)
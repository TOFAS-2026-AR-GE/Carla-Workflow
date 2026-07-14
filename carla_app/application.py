from carla_app.config import Settings
from carla_app.core.client import CarlaSession
from carla_app.core.controller import LaneFollowController
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

            session = CarlaSession(self.settings, scenario)
            client, world = session.open()
            vehicle = spawn_ego_vehicle(world, self.settings.vehicle_name)

            traffic = Traffic(client, world, scenario)
            traffic.spawn()

            sensors = SensorManager(self.settings)
            sensors.start(world, vehicle)

            controller = LaneFollowController(scenario.fixed_delta_seconds)
            perception = PerceptionSystem(self.settings)
            worker = PerceptionWorker(perception)
            viewer = PerceptionViewer()

            print("[INFO] Q veya ESC ile cikis.")
            print(f"[INFO] Vehicle device: {self.settings.vehicle_device}")
            print(f"[INFO] Sign device: {self.settings.sign_device}")

            while True:
                frame_id = world.tick()
                state = read_vehicle_state(world, vehicle)
                control = controller.run_step(state)
                vehicle.apply_control(control)
                update_spectator(world, vehicle)

                rgb_image = sensors.get_rgb(frame_id)
                if (
                    rgb_image is not None
                    and frame_id % self.settings.perception_every_n_frames == 0
                ):
                    worker.submit(frame_id, rgb_image)

                sensors.save_if_needed(
                    frame_id,
                    serializable_vehicle_state(state, control),
                )

                if not viewer.show(worker.get_latest(), rgb_image, frame_id):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Kullanici durdurdu.")
        except Exception as error:
            print(f"[ERROR] {error}")
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

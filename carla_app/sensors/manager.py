from carla_app.sensors.factory import (
    spawn_camera,
    spawn_gnss,
    spawn_imu,
    spawn_lidar,
    spawn_radars,
)
from carla_app.sensors.processors import process_packet
from carla_app.sensors.stream import CameraStream
from carla_app.sensors.sync import SensorSync
from carla_app.sensors.writer import DatasetWriter


SENSOR_NAMES = [
    "rgb_camera",
    "lidar",
    "gnss",
    "imu",
    "radar_front",
    "radar_rear",
    "radar_left",
    "radar_right",
]


class SensorManager:
    def __init__(self, settings):
        self.settings = settings
        self.sync = SensorSync(SENSOR_NAMES)
        self.camera_stream = CameraStream()
        self.writer = DatasetWriter(settings.output_folder)
        self.actors = []

    def start(self, world, vehicle):
        camera = spawn_camera(
            world,
            vehicle,
            self.sync,
            self.camera_stream,
            self.settings.camera_width,
            self.settings.camera_height,
            self.settings.camera_fov,
        )
        lidar = spawn_lidar(world, vehicle, self.sync)
        gnss = spawn_gnss(world, vehicle, self.sync)
        imu = spawn_imu(world, vehicle, self.sync)
        radars = spawn_radars(world, vehicle, self.sync)
        self.actors = [camera, lidar, gnss, imu, *radars.values()]
        print("[OK] Sensorler baslatildi.")

    def get_rgb(self, frame_id):
        return self.camera_stream.wait(frame_id, timeout=0.5)

    def save_if_needed(self, frame_id, vehicle_data):
        if frame_id % self.settings.save_every_n_frames != 0:
            return

        packet = self.sync.wait(frame_id, timeout=1.0)
        if packet is None:
            print(f"[WARN] Sensor paketi eksik, frame atlandi: {frame_id}")
            return

        self.writer.save(frame_id, process_packet(packet), vehicle_data)

    def stop(self):
        for actor in self.actors:
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()
        self.sync.clear()
        self.camera_stream.clear()
        print("[OK] Sensorler kapatildi.")

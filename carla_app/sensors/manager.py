from carla_app.sensors.factory import (
    spawn_layout,
)
from carla_app.sensors.layout import (
    build_sensor_layout,
)
from carla_app.sensors.processors import (
    process_packet,
)
from carla_app.sensors.stream import (
    CameraStream,
)
from carla_app.sensors.sync import (
    SensorSync,
)
from carla_app.sensors.writer import (
    DatasetWriter,
)


class SensorManager:
    def __init__(
        self,
        settings,
    ):
        self.settings = settings

        # Araç bounding-box bilgisi start() sırasında
        # mevcut olduğu için sync ve layout daha sonra
        # oluşturuluyor.
        self.sync = None
        self.layout = None

        self.camera_stream = CameraStream(
            max_frames=12
        )

        self.writer = DatasetWriter(
            settings.output_folder
        )

        self.actors = []

    def start(
        self,
        world,
        vehicle,
    ):
        self.layout = build_sensor_layout(
            vehicle=vehicle,
            camera_width=(
                self.settings.camera_width
            ),
            camera_height=(
                self.settings.camera_height
            ),
            front_wide_fov=(
                self.settings.camera_fov
            ),
            fixed_delta_seconds=(
                self.settings.fixed_delta_seconds
            ),
        )

        self.sync = SensorSync(
            self.layout.sensor_names,
            max_frames=30,
        )

        self.actors = spawn_layout(
            world=world,
            vehicle=vehicle,
            layout=self.layout,
            sync=self.sync,
            camera_stream=self.camera_stream,
        )

        manifest = self.layout.to_manifest(
            vehicle.type_id
        )

        self.writer.write_manifest(
            manifest
        )

        counts = manifest["sensor_count"]

        print(
            "[OK] Sensor setupi baslatildi: "
            f"{counts['cameras']} kamera, "
            f"{counts['automotive_radars']} radar, "
            f"{counts['lidars']} lidar, "
            f"{counts['ultrasonics']} ultrasonik, "
            "GNSS ve IMU."
        )

        print(
            "[INFO] Arac boyutlari: "
            f"L={self.layout.vehicle_geometry['length_m']:.2f} m, "
            f"W={self.layout.vehicle_geometry['width_m']:.2f} m, "
            f"H={self.layout.vehicle_geometry['height_m']:.2f} m"
        )

    def get_rgb(
        self,
        frame_id,
    ):
        # Mevcut perception sistemi bu metodu
        # kullanmaya devam eder.
        # Dönen görüntü camera_front_wide görüntüsüdür.
        return self.camera_stream.wait(
            frame_id,
            timeout=0.5,
        )

    def save_if_needed(
        self,
        frame_id,
        vehicle_data,
    ):
        if (
            frame_id
            % self.settings.save_every_n_frames
            != 0
        ):
            return

        if (
            self.sync is None
            or self.layout is None
        ):
            raise RuntimeError(
                "SensorManager.start() cagrilmadan "
                "veri kaydi yapilamaz."
            )

        packet = self.sync.wait(
            frame_id,
            timeout=1.5,
        )

        if packet is None:
            print(
                "[WARN] Sensor paketi eksik, "
                f"frame atlandi: {frame_id}"
            )
            return

        sensor_data = process_packet(
            packet,
            self.layout,
        )

        self.writer.save(
            frame_id,
            sensor_data,
            vehicle_data,
        )

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

        if self.sync is not None:
            self.sync.clear()

        self.camera_stream.clear()

        print("[OK] Sensorler kapatildi.")
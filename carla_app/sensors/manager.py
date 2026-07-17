"""Uygulamanın kullandığı sensörleri açar, okur ve güvenli biçimde kapatır."""

from carla_app.sensors.factory import spawn_layout
from carla_app.sensors.layout import build_sensor_layout
from carla_app.sensors.processors import process_packet
from carla_app.sensors.stream import CameraStream, RadarStream
from carla_app.sensors.sync import SensorSync
from carla_app.sensors.writer import DatasetWriter


class SensorManager:
    """Normal kullanımda yalnızca ön kamera ile ön radarı çalıştırır."""

    def __init__(self, settings):
        self.settings = settings
        self.recording_enabled = bool(settings.enable_data_recording)
        self.layout = None
        self.sync = None
        self.writer = None
        self.actors = []

        self.camera_stream = CameraStream(max_frames=8)
        self.radar_stream = RadarStream()

        if self.recording_enabled:
            self.writer = DatasetWriter(settings.output_folder)

    def start(self, world, vehicle, fixed_delta_seconds):
        """Yerleşimi hesaplar ve seçilen çalışma modunun sensörlerini açar."""
        self.layout = build_sensor_layout(
            vehicle=vehicle,
            camera_width=self.settings.camera_width,
            camera_height=self.settings.camera_height,
            front_wide_fov=self.settings.camera_fov,
            fixed_delta_seconds=fixed_delta_seconds,
        )

        if self.recording_enabled:
            active_specs = self.layout.all_specs
            self.sync = SensorSync(self.layout.sensor_names, max_frames=8)
        else:
            active_specs = self.layout.control_specs

        self.actors = spawn_layout(
            world=world,
            vehicle=vehicle,
            layout=self.layout,
            sync=self.sync,
            camera_stream=self.camera_stream,
            radar_stream=self.radar_stream,
            specs=active_specs,
        )

        if self.recording_enabled:
            manifest = self.layout.to_manifest(vehicle.type_id)
            self.writer.write_manifest(manifest)
            print(f"[OK] Veri kaydı sensörleri aktif: {len(active_specs)} sensör")
        else:
            names = ", ".join(spec.name for spec in active_specs)
            print(f"[OK] Yalnızca kontrol sensörleri aktif: {names}")

    def get_rgb(self, frame_id):
        return self.camera_stream.wait_latest(frame_id, timeout=0.5)

    def get_radar(self, sensor_name="radar_front_long"):
        return self.radar_stream.get_latest(sensor_name)

    def save_if_needed(self, frame_id, vehicle_data):
        """Kayıt açıksa belirlenen aralıkta tam sensör paketini kaydeder."""
        if not self.recording_enabled:
            return
        if frame_id % self.settings.save_every_n_frames != 0:
            return
        if self.sync is None or self.layout is None or self.writer is None:
            raise RuntimeError(
                "SensorManager.start() çağrılmadan veri kaydı yapılamaz."
            )

        packet = self.sync.wait(frame_id, timeout=1.5)
        if packet is None:
            print(f"[WARN] Sensör paketi eksik; kare atlandı: {frame_id}")
            return

        sensor_data = process_packet(packet, self.layout)
        self.writer.save(frame_id, sensor_data, vehicle_data)

    def stop(self):
        """Oluşturulan bütün CARLA sensörlerini ve geçici veriyi temizler."""
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
        self.radar_stream.clear()
        print("[OK] Sensörler kapatıldı.")

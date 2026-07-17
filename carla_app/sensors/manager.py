"""Uygulamanın kullandığı sensörleri açar, okur ve güvenli biçimde kapatır."""

from carla_app.sensors.factory import spawn_layout
from carla_app.sensors.layout import build_sensor_layout
from carla_app.sensors.processors import process_packet
from carla_app.sensors.stream import (
    CameraStream,
    LatestSensorStream,
    RadarStream,
)
from carla_app.sensors.sync import SensorSync
from carla_app.sensors.writer import DatasetWriter


class SensorManager:
    """Seçilen moda göre kontrol, BEV veya kayıt sensörlerini çalıştırır."""

    def __init__(self, settings):
        self.settings = settings
        self.recording_enabled = bool(settings.enable_data_recording)
        self.bev_enabled = bool(getattr(settings, "enable_bev", False))
        self.sensor_mode = getattr(settings, "sensor_mode", "control")
        self.layout = None
        self.sync = None
        self.writer = None
        self.actors = []

        self.camera_stream = CameraStream(max_frames=8)
        self.radar_stream = RadarStream()
        self.live_stream = LatestSensorStream() if self.bev_enabled else None

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
        elif self.bev_enabled:
            active_specs = self.layout.all_specs
        else:
            active_specs = self.layout.control_specs

        self.actors = spawn_layout(
            world=world,
            vehicle=vehicle,
            layout=self.layout,
            sync=self.sync,
            camera_stream=self.camera_stream,
            radar_stream=self.radar_stream,
            live_stream=self.live_stream,
            specs=active_specs,
        )

        if self.recording_enabled:
            manifest = self.layout.to_manifest(vehicle.type_id)
            self.writer.write_manifest(manifest)
            print(f"[OK] Veri kaydı sensörleri aktif: {len(active_specs)} sensör")
        elif self.bev_enabled:
            print(f"[OK] BEV sensörleri aktif: {len(active_specs)} sensör")
        else:
            sensor_names = []
            for spec in active_specs:
                sensor_names.append(spec.name)
            names = ", ".join(sensor_names)
            print(f"[OK] Yalnızca kontrol sensörleri aktif: {names}")

    def get_rgb(self, frame_id):
        return self.camera_stream.wait_latest(frame_id, timeout=0.5)

    def get_radar(self, sensor_name="radar_front_long"):
        return self.radar_stream.get_latest(sensor_name)

    def get_bev_snapshot(self, frame_id, max_age_frames=10):
        """Sensörlerin en yeni geçerli verisini ana döngüyü bekletmeden verir."""
        if self.live_stream is None:
            return {}
        return self.live_stream.get_snapshot(frame_id, max_age_frames)

    def get_bev_camera_packet(self, frame_id, max_age_frames=10):
        """Yedi kameranın mevcut en yeni görüntülerini adlarıyla verir."""
        snapshot = self.get_bev_snapshot(frame_id, max_age_frames)
        packet = {}
        if self.layout is None:
            return packet

        for camera in self.layout.cameras:
            entry = snapshot.get(camera.name)
            if entry is not None:
                packet[camera.name] = entry
        return packet

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
        if self.live_stream is not None:
            self.live_stream.clear()
        print("[OK] Sensörler kapatıldı.")

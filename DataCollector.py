from collection.DatasetWriter import DatasetWriter
from processing.SensorSync import SensorSync
from processing.sensor_processors import process_frame
from sensors import(
    spawn_gnss,
    spawn_imu,
    spawn_lidar,
    spawn_rgb_camera,
)

class DataCollector:
    def __init__(self, output_folder="data/runs", save_every_n_frames=5):
        self.sensor_names = [
            "rgb_camera",
            "lidar",
            "gnss",
            "imu",
        ]
        
        self.sync = SensorSync(
            sensors=self.sensor_names
        )
        self.dataset_writer = DatasetWriter(
            output_folder=output_folder
        )
        
        self.save_every_n_frames = max(
            1,
            int(save_every_n_frames),
        )
        
        self.sensor_actors = []
    
    def start(self, world, vehicle):
        rgb_camera = spawn_rgb_camera(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
            width=800,
            height=600,
            fov=90,
        )
        
        lidar = spawn_lidar(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
            channels=32,
            range_m=50,
            points_per_second=200000,
            rotation_frequency=20,
        )
        gnss = spawn_gnss(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
        )
        
        imu = spawn_imu(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
        )
        self.sensor_actors = [
            rgb_camera,
            lidar,
            gnss,
            imu,
        ]
        
        print("[OK] Veri toplama sensörleri başlatıldı.")
        print(
            f"[INFO] Her {self.save_every_n_frames} frame'de "
            "bir veri kaydedilecek."
        )
        
    def collect(self, frame_id, vehicle_data):
        """Bir CARLA frame'ine ait sensör verilerini toplar.

        Args:
            frame_id:
                world.tick() tarafından döndürülen CARLA frame numarası.

            vehicle_data:
                Araç hızı, konumu ve kontrol komutları gibi bilgiler.
        """

        # Her frame'i kaydetmek istemiyorsak bazılarını geçiyoruz.
        if frame_id % self.save_every_n_frames != 0:
            return

        # Kamera, LiDAR, GNSS ve IMU'nun aynı frame verisini bekle.
        packet = self.sync.wait_for_frame(
            frame_id=frame_id,
            timeout=1.0,
            allow_partial=False,
        )

        # Sensörlerden biri gelmediyse bu frame'i kaydetme.
        if packet is None:
            print(
                f"[WARN] Frame {frame_id} tamamlanamadı, "
                "kayıt yapılmadı."
            )
            return

        try:
            # Ham CARLA sensör verilerini NumPy veya sözlüğe çevir.
            processed_data = process_frame(packet)

            # İşlenmiş verileri diske kaydet.
            self.dataset_writer.save_frame(
                packet=packet,
                processed_data=processed_data,
                vehicle_data=vehicle_data,
            )

        except Exception as error:
            print(
                f"[ERROR] Frame {frame_id} kaydedilemedi: {error}"
            )

    def stop(self):
        """Bütün sensörleri durdurur ve siler."""

        print("[INFO] Sensörler kapatılıyor...")

        for sensor in self.sensor_actors:
            if sensor is None:
                continue

            try:
                sensor.stop()
            except Exception:
                pass

            try:
                sensor.destroy()
            except Exception:
                pass

        self.sensor_actors.clear()
        self.sync.clear()

        print("[OK] Bütün sensörler kapatıldı.")


from collection.DatasetWriter import DatasetWriter
from processing.SensorSync import SensorSync
from processing.sensor_processors import process_frame

from sensors import (
    spawn_rgb_camera,
    spawn_lidar,
    spawn_gnss,
    spawn_imu,
    spawn_four_radars,
)


class DataCollector:
    """
    Ego araca sensörleri bağlar.

    Aynı CARLA frame'ine ait sensör verilerini toplar,
    işler ve belirli aralıklarla diske kaydeder.
    """

    def __init__(
        self,
        output_folder="data/runs",
        save_every_n_frames=5,
    ):
        # SensorSync bu sensörlerin aynı frame'e ait
        # verilerinin tamamını bekleyecek.
        self.sensor_names = [
            "rgb_camera",
            "lidar",
            "gnss",
            "imu",
            "radar_front",
            "radar_rear",
            "radar_left",
            "radar_right",
        ]

        self.sync = SensorSync(
            sensors=self.sensor_names,
        )

        self.dataset_writer = DatasetWriter(
            output_folder=output_folder,
        )

        # En az 1 frame aralıkla kayıt yapılmasına izin ver.
        self.save_every_n_frames = max(
            1,
            int(save_every_n_frames),
        )

        # Program kapanırken sensörleri durdurabilmek için
        # bütün CARLA sensör aktörlerini burada tutuyoruz.
        self.sensor_actors = []

    def start(self, world, vehicle):
        """
        Bütün sensörleri oluşturur ve ego araca bağlar.
        """

        # Ön kamera
        rgb_camera = spawn_rgb_camera(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
            width=800,
            height=600,
            fov=90,
        )

        # LiDAR
        lidar = spawn_lidar(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
            channels=32,
            range_m=50,
            points_per_second=200000,
            rotation_frequency=20,
        )

        # Konum sensörü
        gnss = spawn_gnss(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
        )

        # İvme ve açısal hız sensörü
        imu = spawn_imu(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
        )

        # Ön, arka, sol ve sağ radarlar
        radars = spawn_four_radars(
            world=world,
            vehicle=vehicle,
            sync=self.sync,
        )

        # Oluşturulan bütün sensörleri sakla.
        self.sensor_actors = [
            rgb_camera,
            lidar,
            gnss,
            imu,
            radars["radar_front"],
            radars["radar_rear"],
            radars["radar_left"],
            radars["radar_right"],
        ]

        print("[OK] Veri toplama sensörleri başlatıldı.")

        print(
            f"[INFO] Her {self.save_every_n_frames} frame'de "
            "bir veri kaydedilecek."
        )

    def collect(self, frame_id, vehicle_data):
        """
        Belirtilen CARLA frame'ine ait sensör verilerini toplar.

        Args:
            frame_id:
                world.tick() tarafından döndürülen frame numarası.

            vehicle_data:
                Ego aracın hız, konum ve kontrol bilgileri.

        Returns:
            İşlenmiş sensör verileri.

            Bu frame kaydedilmeyecekse veya sensör verileri
            tamamlanamazsa None döner.
        """

        # Örneğin save_every_n_frames=5 ise:
        # yalnızca 5, 10, 15, 20... frame'leri işle.
        if frame_id % self.save_every_n_frames != 0:
            return None

        # Bütün sensörlerin aynı frame'e ait verisini bekle.
        packet = self.sync.wait_for_frame(
            frame_id=frame_id,
            timeout=1.0,
            allow_partial=False,
        )

        if packet is None:
            print(
                f"[WARN] Frame {frame_id} tamamlanamadı. "
                "Kayıt yapılmadı."
            )

            return None

        try:
            # Kamera, LiDAR, GNSS, IMU ve dört radarın
            # tamamı burada işlenir.
            processed_data = process_frame(packet)

            # İşlenmiş sensör verilerini diske kaydet.
            self.dataset_writer.save_frame(
                packet=packet,
                processed_data=processed_data,
                vehicle_data=vehicle_data,
            )

            # Daha sonra main.py içinde YOLO ve kontrol sistemi
            # tarafından kullanılabilmesi için veriyi döndür.
            return processed_data

        except Exception as error:
            print(
                f"[ERROR] Frame {frame_id} işlenemedi: "
                f"{error}"
            )

            return None

    def stop(self):
        """
        Bütün sensörleri durdurur ve CARLA dünyasından siler.
        """

        print("[INFO] Sensörler kapatılıyor...")

        for sensor in self.sensor_actors:
            if sensor is None:
                continue

            # Önce sensörün veri üretmesini durdur.
            try:
                sensor.stop()
            except Exception:
                pass

            # Sonra sensörü CARLA dünyasından sil.
            try:
                sensor.destroy()
            except Exception:
                pass

        self.sensor_actors.clear()
        self.sync.clear()

        print("[OK] Bütün sensörler kapatıldı.")
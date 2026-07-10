"""CARLA sensör verilerini frame numarasına göre eşleştirir."""

import threading
import time

from processing.FramePacket import FramePacket


class SensorSync:
    """Sensör verilerini CARLA frame numarasına göre toplar."""

    def __init__(self, sensors, max_buffer_frames=100):
        # Örnek:
        # ["rgb_camera", "lidar", "gnss", "imu"]
        self.sensors = list(sensors)

        if len(self.sensors) == 0:
            raise ValueError("En az bir sensör ismi verilmelidir.")

        self.max_buffer_frames = int(max_buffer_frames)

        # Buffer yapısı:
        #
        # {
        #     100: {
        #         "rgb_camera": image,
        #         "lidar": lidar_data
        #     },
        #     101: {
        #         "rgb_camera": image
        #     }
        # }
        self.buffer = {}

        # CARLA sensör callback'leri farklı thread'lerden çalışabilir.
        # Lock, aynı anda buffer'a yazılmasını güvenli hale getirir.
        self.lock = threading.Lock()

    def push(self, sensor_name, frame, data):
        """Sensörden gelen veriyi buffer'a ekler.

        sensors.py içindeki callback'ler bu fonksiyonu çağırır.

        Örnek:

            sync.push("rgb_camera", image.frame, image)
        """

        if sensor_name not in self.sensors:
            raise ValueError(
                f"Beklenmeyen sensör: {sensor_name}. "
                f"Beklenen sensörler: {self.sensors}"
            )

        frame = int(frame)

        with self.lock:
            if frame not in self.buffer:
                self.buffer[frame] = {}

            self.buffer[frame][sensor_name] = data

            self._limit_buffer_size()

    def wait_for_frame(
        self,
        frame_id,
        timeout=2.0,
        allow_partial=False,
    ):
        """Bir frame için bütün sensörlerin gelmesini bekler.

        Args:
            frame_id:
                Beklenen CARLA frame numarası.

            timeout:
                En fazla kaç saniye bekleyeceğimiz.

            allow_partial:
                True ise bazı sensörler eksik olsa da mevcut verileri döndürür.
                False ise eksik sensör varsa None döndürür.
        """

        frame_id = int(frame_id)
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.lock:
                frame_data = self.buffer.get(frame_id, {})

                if self._all_sensors_arrived(frame_data):
                    completed_data = dict(frame_data)

                    del self.buffer[frame_id]
                    self._remove_old_frames(frame_id)

                    return FramePacket(
                        frame_id=frame_id,
                        sensor_data=completed_data,
                        expected_sensors=self.sensors,
                    )

            # CPU'yu boşuna yormamak için kısa süre bekliyoruz.
            time.sleep(0.005)

        # Timeout oldu.
        with self.lock:
            frame_data = dict(
                self.buffer.pop(frame_id, {})
            )

            self._remove_old_frames(frame_id)

        if len(frame_data) == 0:
            print(
                f"[WARN] Frame {frame_id} için hiçbir sensör verisi gelmedi."
            )
            return None

        packet = FramePacket(
            frame_id=frame_id,
            sensor_data=frame_data,
            expected_sensors=self.sensors,
        )

        if packet.is_complete():
            return packet

        print(
            f"[WARN] Frame {frame_id} eksik. "
            f"Eksik sensörler: {packet.missing_sensors()}"
        )

        if allow_partial:
            return packet

        return None

    def _all_sensors_arrived(self, frame_data):
        """Bütün beklenen sensörler geldi mi?"""

        for sensor_name in self.sensors:
            if sensor_name not in frame_data:
                return False

        return True

    def _remove_old_frames(self, current_frame):
        """Kullanılmayacak eski frame'leri temizler.

        Bu fonksiyon çağrılırken lock açık olmalıdır.
        """

        old_frames = []

        for frame_id in self.buffer:
            if frame_id < current_frame:
                old_frames.append(frame_id)

        for frame_id in old_frames:
            del self.buffer[frame_id]

    def _limit_buffer_size(self):
        """Buffer'ın bellekte sınırsız büyümesini engeller.

        Bu fonksiyon çağrılırken lock açık olmalıdır.
        """

        if len(self.buffer) <= self.max_buffer_frames:
            return

        frame_ids = sorted(self.buffer.keys())

        while len(frame_ids) > self.max_buffer_frames:
            oldest_frame = frame_ids.pop(0)

            if oldest_frame in self.buffer:
                del self.buffer[oldest_frame]

    def cleanup(self, upto_frame):
        """Belirtilen frame ve öncesindeki verileri temizler."""

        upto_frame = int(upto_frame)

        with self.lock:
            frames_to_delete = []

            for frame_id in self.buffer:
                if frame_id <= upto_frame:
                    frames_to_delete.append(frame_id)

            for frame_id in frames_to_delete:
                del self.buffer[frame_id]

    def clear(self):
        """Bütün buffer'ı temizler."""

        with self.lock:
            self.buffer.clear()

    def buffered_frame_count(self):
        """Buffer içinde kaç frame bulunduğunu döndürür."""

        with self.lock:
            return len(self.buffer)
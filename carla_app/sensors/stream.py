"""Kamera ve radarın en güncel verisini iş parçacıkları arasında saklar."""

import threading
import time


class CameraStream:
    def __init__(self, max_frames=10):
        self.max_frames = max_frames
        self.frames = {}
        self.condition = threading.Condition()

    def push(self, frame_id, rgb_image):
        with self.condition:
            self.frames[int(frame_id)] = rgb_image
            while len(self.frames) > self.max_frames:
                del self.frames[min(self.frames)]
            self.condition.notify_all()

    def wait_latest(self, world_frame_id, timeout=0.5):
        """Dünya karesine göre kullanılabilir en güncel görüntüyü verir.

        RGB kamera GPU üzerinde çizildiği için callback birkaç dünya karesi
        gecikebilir. Yalnızca tam kare numarasını beklemek geç gelen bütün
        görüntüleri kaybettirir. Bu nedenle görüntü kendi gerçek kare
        numarasıyla birlikte döndürülür.
        """
        world_frame_id = int(world_frame_id)
        end_time = time.monotonic() + timeout

        with self.condition:
            while True:
                available = []
                for frame_id in self.frames:
                    if frame_id <= world_frame_id:
                        available.append(frame_id)
                if available:
                    camera_frame_id = max(available)
                    image = self.frames[camera_frame_id]
                    old_frame_ids = []
                    for frame_id in self.frames:
                        if frame_id <= camera_frame_id:
                            old_frame_ids.append(frame_id)
                    for old_id in old_frame_ids:
                        del self.frames[old_id]
                    return camera_frame_id, image

                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return None, None
                self.condition.wait(remaining)

    def clear(self):
        with self.condition:
            self.frames.clear()


class RadarStream:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}

    def push(self, sensor_name, frame_id, points):
        with self.lock:
            self.latest[sensor_name] = (int(frame_id), points)

    def get_latest(self, sensor_name):
        with self.lock:
            entry = self.latest.get(sensor_name)
            return entry if entry is not None else (None, [])

    def clear(self):
        with self.lock:
            self.latest.clear()


class LatestSensorStream:
    """BEV için her sensörün en yeni verisini beklemeden saklar.

    Bir sensör geç kaldığında ana araç döngüsü durmaz. Her verinin gerçek
    kare numarası ve mevcut dünya karesine göre yaşı birlikte döndürülür.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.latest = {}

    def push(self, sensor_name, frame_id, data):
        with self.lock:
            self.latest[sensor_name] = (int(frame_id), data)

    def get_snapshot(self, world_frame_id=None, max_age_frames=None):
        snapshot = {}

        with self.lock:
            for sensor_name, entry in self.latest.items():
                frame_id, data = entry
                age = None
                if world_frame_id is not None:
                    age = int(world_frame_id) - frame_id
                    if age < 0:
                        continue
                    if max_age_frames is not None and age > max_age_frames:
                        continue

                snapshot[sensor_name] = {
                    "frame_id": frame_id,
                    "age_frames": age,
                    "data": data,
                }

        return snapshot

    def clear(self):
        with self.lock:
            self.latest.clear()

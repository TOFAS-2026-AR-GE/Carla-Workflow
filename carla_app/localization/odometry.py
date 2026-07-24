"""Gerçek araç teker/CAN odometrisini EKF'ye bağlayan sade adaptör."""

import math


class OdometryAdapter:
    """Sensör paketindeki standart odometri kaydını doğrular.

    CARLA'nın yerleşik teker enkoder sensörü yoktur. Bu adaptör gerçek araçta
    CAN/teker hızının veya harici odometri düğümünün aynı sensör snapshot'ına
    ``wheel_odometry`` adıyla eklenmesini bekler. Ground-truth ``get_velocity``
    hiçbir zaman odometri yerine kullanılmaz.
    """

    def __init__(self, maximum_speed_mps=70.0):
        self.maximum_speed_mps = max(1.0, float(maximum_speed_mps))
        self.names = (
            "wheel_odometry",
            "can_wheel_odometry",
            "vehicle_odometry",
        )
        self.last_frame_id = None

    def read(self, sensor_snapshot, current_frame_id):
        sensor_snapshot = sensor_snapshot or {}
        for name in self.names:
            entry = sensor_snapshot.get(name)
            result = self.standardize(entry, current_frame_id, name)
            if result is not None:
                return result
        return None

    def standardize(self, entry, current_frame_id, source):
        if entry is None:
            return None
        data = entry.get("data", entry)
        try:
            frame_id = int(entry.get("frame_id", data.get("frame_id")))
            speed_mps = float(data.get("speed_mps"))
        except (AttributeError, TypeError, ValueError):
            return None
        if frame_id > int(current_frame_id):
            return None
        if not math.isfinite(speed_mps) or not 0.0 <= speed_mps <= self.maximum_speed_mps:
            return None
        if self.last_frame_id is not None and frame_id <= self.last_frame_id:
            return None
        self.last_frame_id = frame_id
        return {
            "frame_id": frame_id,
            "speed_mps": speed_mps,
            "source": source,
        }

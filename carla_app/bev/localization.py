"""GNSS ve IMU canlılığını sade ve açıklanabilir biçimde doğrular."""

import math


class LocalizationHealth:
    """GNSS/IMU ölçümlerinin taze ve fiziksel olarak geçerli olduğunu denetler."""

    def __init__(self, layout, maximum_age_frames=4):
        self.gnss_name = layout.gnss.name
        self.imu_name = layout.imu.name
        self.maximum_age_frames = max(0, int(maximum_age_frames))

    def evaluate(self, sensor_snapshot, current_frame_id):
        gnss = self._validate_gnss(
            sensor_snapshot.get(self.gnss_name),
            current_frame_id,
        )
        imu = self._validate_imu(
            sensor_snapshot.get(self.imu_name),
            current_frame_id,
        )
        available = int(gnss["valid"]) + int(imu["valid"])
        if available == 2:
            status = "HEALTHY"
        elif available == 1:
            status = "DEGRADED"
        else:
            status = "UNAVAILABLE"
        return {
            "status": status,
            "gnss": gnss,
            "imu": imu,
            "available_sources": available,
        }

    def _validate_gnss(self, entry, current_frame_id):
        base = self._entry_status(entry, current_frame_id)
        if not base["fresh"]:
            return base
        data = entry.get("data", {})
        try:
            latitude = float(data["latitude"])
            longitude = float(data["longitude"])
            altitude = float(data["altitude"])
        except (KeyError, TypeError, ValueError):
            base["reason"] = "invalid_values"
            return base

        valid = (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and math.isfinite(altitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
        )
        base.update(
            {
                "valid": valid,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "reason": None if valid else "invalid_values",
            }
        )
        return base

    def _validate_imu(self, entry, current_frame_id):
        base = self._entry_status(entry, current_frame_id)
        if not base["fresh"]:
            return base
        data = entry.get("data", {})
        try:
            acceleration = data["accelerometer"]
            gyroscope = data["gyroscope"]
            values = (
                float(acceleration["x"]),
                float(acceleration["y"]),
                float(acceleration["z"]),
                float(gyroscope["x"]),
                float(gyroscope["y"]),
                float(gyroscope["z"]),
                float(data["compass"]),
            )
        except (KeyError, TypeError, ValueError):
            base["reason"] = "invalid_values"
            return base

        valid = all(math.isfinite(value) for value in values)
        valid = valid and max(abs(value) for value in values[:3]) <= 100.0
        valid = valid and max(abs(value) for value in values[3:6]) <= 20.0
        base.update(
            {
                "valid": valid,
                "lateral_acceleration_mps2": values[1],
                "yaw_rate_radps": values[5],
                "compass_rad": values[6],
                "reason": None if valid else "invalid_values",
            }
        )
        return base

    def _entry_status(self, entry, current_frame_id):
        if entry is None or entry.get("frame_id") is None:
            return {
                "valid": False,
                "fresh": False,
                "age_frames": None,
                "reason": "missing",
            }
        age = int(current_frame_id) - int(entry["frame_id"])
        fresh = 0 <= age <= self.maximum_age_frames
        return {
            "valid": False,
            "fresh": fresh,
            "age_frames": age,
            "reason": None if fresh else "stale",
        }

"""Sensör füzyonunu CARLA rota durumuna bağlayan lokalizasyon sistemi."""

import math
from types import SimpleNamespace

try:
    import carla
except ImportError:  # Birim testlerinde gerçek CARLA paketi zorunlu değildir.
    carla = None

from carla_app.localization.ekf_localizer import (
    ExtendedKalmanLocalizer,
    wrap_angle,
)
from carla_app.localization.odometry import OdometryAdapter


class GnssMapProjector:
    """GNSS ölçümünü CARLA dünya koordinatına dönüştürür."""

    def __init__(self, carla_map):
        self.carla_map = carla_map

    def project(self, gnss):
        latitude = float(gnss["latitude"])
        longitude = float(gnss["longitude"])
        altitude = float(gnss.get("altitude", 0.0))

        if not hasattr(self.carla_map, "geolocation_to_transform"):
            raise RuntimeError(
                "CARLA haritası geolocation_to_transform() sağlamıyor; "
                "oracle konum fallback'i bilinçli olarak kullanılmadı."
            )

        if carla is not None and hasattr(carla, "GeoLocation"):
            geolocation = carla.GeoLocation(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
            )
        else:
            geolocation = SimpleNamespace(
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
            )
        location = self.carla_map.geolocation_to_transform(geolocation)
        return float(location.x), float(location.y), float(location.z)


def compass_to_carla_yaw(compass_rad):
    """CARLA IMU pusulasını CARLA dünya yaw açısına dönüştürür."""
    # Pusula: 0 kuzey, pi/2 doğu. CARLA yaw: 0 +X/doğu yönüdür.
    return wrap_angle(math.pi / 2.0 - float(compass_rad))


class LocalizationSystem:
    """GNSS/IMU EKF sonucundan kontrolcünün araç durumunu üretir."""

    def __init__(self, carla_map, layout, dt, settings):
        self.carla_map = carla_map
        self.layout = layout
        self.dt = max(0.01, float(dt))
        self.settings = settings
        self.projector = GnssMapProjector(carla_map)
        self.odometry_adapter = OdometryAdapter()
        self.filter = ExtendedKalmanLocalizer(
            dt=self.dt,
            gnss_position_std_m=float(
                getattr(settings, "localization_gnss_std_m", 1.5)
            ),
            compass_std_rad=math.radians(
                float(getattr(settings, "localization_compass_std_deg", 6.0))
            ),
            odometry_speed_std_mps=float(
                getattr(settings, "localization_odometry_std_mps", 0.45)
            ),
        )
        self.last_result = {"available": False, "status": "UNINITIALIZED"}
        self.last_gnss_frame_id = None
        self.last_seen_gnss_frame_id = None
        self.last_imu_frame_id = None
        self.last_valid_frame_id = None
        self.last_altitude_m = 0.0

    def wait_until_ready(self, world, sensors, maximum_frames=40):
        """İlk geçerli GNSS ve IMU gelene kadar dünyayı güvenli biçimde ilerletir."""
        maximum_frames = max(1, int(maximum_frames))
        for _ in range(maximum_frames):
            frame_id = world.tick()
            snapshot = sensors.get_bev_snapshot(
                frame_id,
                max_age_frames=max(2, int(round(0.25 / self.dt))),
            )
            result = self.update(frame_id, snapshot)
            if result.get("available"):
                return frame_id
        raise RuntimeError(
            "GNSS+IMU lokalizasyonu başlatılamadı. Oracle pose fallback kapalıdır."
        )

    def update(self, frame_id, sensor_snapshot, odometry=None):
        """En yeni sensör ölçümleriyle EKF'yi bir kare günceller."""
        frame_id = int(frame_id)
        sensor_snapshot = sensor_snapshot or {}
        gnss_entry = sensor_snapshot.get(self.layout.gnss.name)
        imu_entry = sensor_snapshot.get(self.layout.imu.name)
        gnss = None if gnss_entry is None else gnss_entry.get("data")
        imu = None if imu_entry is None else imu_entry.get("data")
        if odometry is None:
            odometry = self.odometry_adapter.read(sensor_snapshot, frame_id)

        if not self.filter.initialized:
            if gnss_entry is None or imu_entry is None or gnss is None or imu is None:
                self.last_result = {
                    "available": False,
                    "status": "UNINITIALIZED",
                    "reason": "waiting_for_gnss_and_imu",
                }
                return dict(self.last_result)
            x_m, y_m, z_m = self.projector.project(gnss)
            yaw_rad = compass_to_carla_yaw(imu.get("compass", 0.0))
            initial_speed = 0.0
            if odometry is not None:
                initial_speed = max(0.0, float(odometry.get("speed_mps", 0.0)))
            self.filter.initialize(x_m, y_m, yaw_rad, frame_id, initial_speed)
            self.last_altitude_m = z_m

        self.filter.predict(frame_id, imu)

        if gnss_entry is not None and gnss is not None:
            gnss_frame_id = int(gnss_entry.get("frame_id", frame_id))
            if gnss_frame_id != self.last_seen_gnss_frame_id:
                x_m, y_m, z_m = self.projector.project(gnss)
                accepted = self.filter.update_gnss(x_m, y_m, gnss_frame_id)
                if accepted:
                    self.last_altitude_m = z_m
                    self.last_valid_frame_id = frame_id
                    self.last_gnss_frame_id = gnss_frame_id
                self.last_seen_gnss_frame_id = gnss_frame_id

        if imu_entry is not None and imu is not None:
            imu_frame_id = int(imu_entry.get("frame_id", frame_id))
            if imu_frame_id != self.last_imu_frame_id:
                yaw_rad = compass_to_carla_yaw(imu.get("compass", 0.0))
                self.filter.update_compass(yaw_rad, imu_frame_id)
                self.last_imu_frame_id = imu_frame_id
                self.last_valid_frame_id = frame_id

        if odometry is not None:
            odometry_frame_id = int(odometry.get("frame_id", frame_id))
            speed_mps = odometry.get("speed_mps")
            if speed_mps is not None:
                self.filter.update_odometry_speed(speed_mps, odometry_frame_id)

        result = self.filter.result()
        sensor_age_frames = self.sensor_age_frames(frame_id)
        sensor_age_s = sensor_age_frames * self.dt
        if sensor_age_frames > 0:
            # Process modeli zaten her kare covariance büyütür. Eksik ölçüm
            # payını toplam yaşla tekrar tekrar eklemek yerine yalnız bu çevrimin
            # ek belirsizliğini ekleriz.
            self.filter.inflate_for_missing_measurement(
                min(self.dt, sensor_age_s)
            )
            result = self.filter.result()

        status = self.health_status(result, sensor_age_s)
        result.update(
            {
                "status": status,
                "frame_id": frame_id,
                "sensor_age_frames": sensor_age_frames,
                "sensor_age_s": sensor_age_s,
                "z_m": float(self.last_altitude_m),
                "source": "gnss_imu_ekf",
                "odometry_used": bool(odometry is not None),
            }
        )
        self.last_result = result
        return dict(result)

    def sensor_age_frames(self, current_frame_id):
        frames = []
        if self.last_gnss_frame_id is not None:
            frames.append(int(current_frame_id) - int(self.last_gnss_frame_id))
        if self.last_imu_frame_id is not None:
            frames.append(int(current_frame_id) - int(self.last_imu_frame_id))
        if not frames:
            return 10**9
        return max(0, max(frames))

    def health_status(self, result, sensor_age_s):
        if not result.get("available"):
            return "UNAVAILABLE"
        stop_age = float(getattr(self.settings, "localization_stop_age_s", 0.60))
        degraded_age = float(
            getattr(self.settings, "localization_degraded_age_s", 0.25)
        )
        maximum_position_std = float(
            getattr(self.settings, "localization_max_position_std_m", 3.0)
        )
        maximum_yaw_std = math.radians(
            float(getattr(self.settings, "localization_max_yaw_std_deg", 12.0))
        )
        if sensor_age_s >= stop_age:
            return "LOST"
        if (
            sensor_age_s >= degraded_age
            or float(result.get("position_std_m", math.inf)) > maximum_position_std
            or float(result.get("yaw_std_rad", math.inf)) > maximum_yaw_std
        ):
            return "DEGRADED"
        return "NOMINAL"

    def build_vehicle_state(self, frame_id, route_manager, vehicle):
        """EKF pozunu rota ve statik araç geometrisiyle kontrol durumuna çevirir."""
        result = self.last_result
        if not result.get("available"):
            raise RuntimeError("Lokalizasyon hazır olmadan araç durumu üretilemez.")

        location = self.make_location(
            result["x_m"],
            result["y_m"],
            result.get("z_m", 0.0),
        )
        waypoint, reference_path = route_manager.update(location)
        if waypoint is None:
            raise RuntimeError("EKF konumu için kalıcı rota waypoint'i bulunamadı.")

        next_location = reference_path[0] if reference_path else None
        speed_mps = max(0.0, float(result["speed_mps"]))
        half_width = 0.95
        try:
            half_width = float(vehicle.bounding_box.extent.y)
        except (AttributeError, TypeError, ValueError):
            pass

        return {
            "frame_id": int(frame_id),
            "location": location,
            "yaw": float(result["yaw_deg"]),
            "speed_mps": speed_mps,
            "speed_kmh": speed_mps * 3.6,
            "next_location": next_location,
            "reference_path": reference_path,
            "road_id": waypoint.road_id,
            "lane_id": waypoint.lane_id,
            "lane_width": waypoint.lane_width,
            "vehicle_half_width_m": half_width,
            "is_junction": waypoint.is_junction,
            "localization": dict(result),
            # Normal kontrol akışında simulator_traffic_light bilinçli olarak yoktur.
        }

    def make_location(self, x_m, y_m, z_m):
        if carla is not None and hasattr(carla, "Location"):
            return carla.Location(x=float(x_m), y=float(y_m), z=float(z_m))
        return SimpleNamespace(x=float(x_m), y=float(y_m), z=float(z_m))

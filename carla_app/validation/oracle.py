"""CARLA ground-truth bilgisini yalnız doğrulama için izole eder."""

import math

from carla_app.core.state import read_simulator_traffic_light


class OracleValidator:
    """Lokalizasyon ve trafik ışığı çıktısını CARLA oracle'ıyla karşılaştırır.

    Bu sınıfın çıktısı hiçbir kontrol kararına verilmez. Yalnız log, test ve
    mühendislik doğrulaması için kullanılır.
    """

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self.sample_count = 0
        self.position_error_sum_m = 0.0
        self.heading_error_sum_deg = 0.0
        self.speed_error_sum_mps = 0.0
        self.light_sample_count = 0
        self.light_match_count = 0

    def observe(self, vehicle, localized_state, road_context=None):
        if not self.enabled:
            return {"enabled": False, "control_connected": False}

        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        oracle_speed = math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )
        location = localized_state["location"]
        position_error = math.hypot(
            float(location.x) - float(transform.location.x),
            float(location.y) - float(transform.location.y),
        )
        heading_error = self.angle_difference_deg(
            float(localized_state.get("yaw", 0.0)),
            float(transform.rotation.yaw),
        )
        speed_error = abs(float(localized_state.get("speed_mps", 0.0)) - oracle_speed)

        self.sample_count += 1
        self.position_error_sum_m += position_error
        self.heading_error_sum_deg += heading_error
        self.speed_error_sum_mps += speed_error

        simulator_light = read_simulator_traffic_light(vehicle)
        camera_light = (road_context or {}).get("lead_traffic_light")
        light_match = None
        if simulator_light.get("affected"):
            oracle_color = simulator_light.get("color")
            camera_color = None if camera_light is None else camera_light.get("color")
            light_match = camera_color == oracle_color
            self.light_sample_count += 1
            if light_match:
                self.light_match_count += 1

        return {
            "enabled": True,
            "control_connected": False,
            "position_error_m": float(position_error),
            "heading_error_deg": float(heading_error),
            "speed_error_mps": float(speed_error),
            "mean_position_error_m": self.position_error_sum_m / self.sample_count,
            "mean_heading_error_deg": self.heading_error_sum_deg / self.sample_count,
            "mean_speed_error_mps": self.speed_error_sum_mps / self.sample_count,
            "traffic_light": {
                "oracle": simulator_light,
                "camera_color": (
                    None if camera_light is None else camera_light.get("color")
                ),
                "match": light_match,
                "accuracy": (
                    self.light_match_count / self.light_sample_count
                    if self.light_sample_count
                    else None
                ),
            },
        }

    def angle_difference_deg(self, first_deg, second_deg):
        difference = math.radians(float(first_deg) - float(second_deg))
        wrapped = math.atan2(math.sin(difference), math.cos(difference))
        return abs(math.degrees(wrapped))

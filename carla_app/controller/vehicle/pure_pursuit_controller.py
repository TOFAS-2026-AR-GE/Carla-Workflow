"""Referans rotayı Pure Pursuit ile izleyen tek yanal kontrolcü."""

import math


def clamp(value, minimum, maximum):
    """Bir sayıyı verilen alt ve üst sınırlar içinde tutar."""
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    """Açıyı -pi ile +pi aralığına getirir."""
    return math.atan2(math.sin(angle), math.cos(angle))


class PurePursuitController:
    """Aracın önündeki bir rota noktasına doğru yumuşak direksiyon üretir."""

    def __init__(self, dt=0.05):
        self.dt = max(0.01, float(dt))

        # Tesla Model 3 için kullanılan yaklaşık dingil mesafesi.
        self.wheelbase_m = 2.85
        self.maximum_wheel_angle_rad = math.radians(70.0)

        # Hız arttıkça daha uzağa bakmak direksiyon hareketini sakinleştirir.
        self.minimum_lookahead_m = 4.5
        self.maximum_lookahead_m = 22.0
        self.lookahead_speed_gain_s = 0.65

        # Direksiyonun tek çevrimde sıçramasını önleyen konfor değerleri.
        self.steering_filter_ratio = 0.45
        self.previous_steer = 0.0
        self.last_info = self.empty_info()

    def run_step(self, state):
        """Araç durumu ve referans rotadan -1 ile +1 arası direksiyon verir."""
        path = state.get("reference_path", [])
        if len(path) < 2:
            self.previous_steer *= 0.90
            self.last_info = self.empty_info()
            self.last_info["steer"] = self.previous_steer
            self.last_info["reason"] = "short_path"
            return self.previous_steer

        location = state["location"]
        vehicle_x = float(location.x)
        vehicle_y = float(location.y)
        vehicle_yaw = math.radians(float(state.get("yaw", 0.0)))
        speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))

        projection = self.find_nearest_projection(vehicle_x, vehicle_y, path)
        lookahead_m = self.calculate_lookahead(speed_mps)
        target = self.find_target_point(path, projection, lookahead_m)

        delta_x = target["x"] - vehicle_x
        delta_y = target["y"] - vehicle_y

        # Hedefi araç koordinatına çeviririz. local_y hedefin yan konumudur.
        local_y = -math.sin(vehicle_yaw) * delta_x + math.cos(vehicle_yaw) * delta_y
        target_distance_squared = max(0.25, delta_x**2 + delta_y**2)

        # Pure Pursuit eğriliği ve bisiklet modeli direksiyon açısı.
        pursuit_curvature = 2.0 * local_y / target_distance_squared
        wheel_angle = math.atan(self.wheelbase_m * pursuit_curvature)
        desired_steer = wheel_angle / self.maximum_wheel_angle_rad

        steering_limit = self.calculate_steering_limit(speed_mps)
        desired_steer = clamp(desired_steer, -steering_limit, steering_limit)
        steer = self.smooth_steering(desired_steer, speed_mps)

        path_heading = projection["path_heading_rad"]
        heading_error = normalize_angle(path_heading - vehicle_yaw)
        self.previous_steer = steer
        self.last_info = {
            "controller": "pure_pursuit",
            "reason": "tracking",
            "steer": float(steer),
            "desired_steer": float(desired_steer),
            "lookahead_m": float(lookahead_m),
            "target_x": float(target["x"]),
            "target_y": float(target["y"]),
            "target_index": int(target["segment_index"]),
            "cross_track_error_m": float(projection["cross_track_error_m"]),
            "heading_error_rad": float(heading_error),
            "curvature_1pm": float(pursuit_curvature),
            "steer_limit": float(steering_limit),
        }
        return steer

    def calculate_lookahead(self, speed_mps):
        """Düşük hızda çevik, yüksek hızda sakin bir bakış mesafesi seçer."""
        lookahead = self.minimum_lookahead_m + self.lookahead_speed_gain_s * speed_mps
        return clamp(lookahead, self.minimum_lookahead_m, self.maximum_lookahead_m)

    def find_nearest_projection(self, vehicle_x, vehicle_y, path):
        """Aracın en yakın rota parçası üzerindeki izdüşümünü bulur."""
        best = None
        best_distance_squared = math.inf

        # Rota yöneticisi aracın arkasında yalnızca birkaç nokta bırakır.
        # İlk 80 parça kontrol için yeterlidir ve hesabı sabit sürede tutar.
        segment_count = min(len(path) - 1, 80)
        for index in range(segment_count):
            start = path[index]
            end = path[index + 1]
            segment_x = float(end.x) - float(start.x)
            segment_y = float(end.y) - float(start.y)
            length_squared = segment_x**2 + segment_y**2
            if length_squared < 1e-8:
                continue

            fraction = (
                (vehicle_x - float(start.x)) * segment_x
                + (vehicle_y - float(start.y)) * segment_y
            ) / length_squared
            fraction = clamp(fraction, 0.0, 1.0)
            projection_x = float(start.x) + fraction * segment_x
            projection_y = float(start.y) + fraction * segment_y
            error_x = vehicle_x - projection_x
            error_y = vehicle_y - projection_y
            distance_squared = error_x**2 + error_y**2

            if distance_squared < best_distance_squared:
                segment_length = math.sqrt(length_squared)
                signed_error = (
                    segment_x * error_y - segment_y * error_x
                ) / segment_length
                best_distance_squared = distance_squared
                best = {
                    "segment_index": index,
                    "fraction": fraction,
                    "x": projection_x,
                    "y": projection_y,
                    "cross_track_error_m": signed_error,
                    "path_heading_rad": math.atan2(segment_y, segment_x),
                }

        if best is None:
            first = path[0]
            second = path[1]
            best = {
                "segment_index": 0,
                "fraction": 0.0,
                "x": float(first.x),
                "y": float(first.y),
                "cross_track_error_m": 0.0,
                "path_heading_rad": math.atan2(
                    float(second.y) - float(first.y),
                    float(second.x) - float(first.x),
                ),
            }
        return best

    def find_target_point(self, path, projection, lookahead_m):
        """İzdüşümden bakış mesafesi kadar ilerideki noktayı bulur."""
        segment_index = int(projection["segment_index"])
        current_x = float(projection["x"])
        current_y = float(projection["y"])
        remaining = float(lookahead_m)

        for index in range(segment_index, len(path) - 1):
            end = path[index + 1]
            end_x = float(end.x)
            end_y = float(end.y)
            segment_x = end_x - current_x
            segment_y = end_y - current_y
            segment_length = math.hypot(segment_x, segment_y)

            if segment_length >= remaining and segment_length > 1e-6:
                ratio = remaining / segment_length
                return {
                    "x": current_x + ratio * segment_x,
                    "y": current_y + ratio * segment_y,
                    "segment_index": index,
                }

            remaining -= segment_length
            current_x = end_x
            current_y = end_y

        last = path[-1]
        return {
            "x": float(last.x),
            "y": float(last.y),
            "segment_index": len(path) - 2,
        }

    def calculate_steering_limit(self, speed_mps):
        """Yüksek hızda aşırı direksiyon komutunu sınırlar."""
        return clamp(1.0 - 0.018 * speed_mps, 0.55, 1.0)

    def smooth_steering(self, desired_steer, speed_mps):
        """Alçak geçiren filtre ve değişim sınırını birlikte uygular."""
        filtered = self.previous_steer + self.steering_filter_ratio * (
            desired_steer - self.previous_steer
        )
        maximum_rate = clamp(0.90 - 0.020 * speed_mps, 0.45, 0.90)
        maximum_change = maximum_rate * self.dt
        change = clamp(
            filtered - self.previous_steer,
            -maximum_change,
            maximum_change,
        )
        return clamp(self.previous_steer + change, -1.0, 1.0)

    def empty_info(self):
        return {
            "controller": "pure_pursuit",
            "reason": "not_started",
            "steer": 0.0,
            "desired_steer": 0.0,
            "lookahead_m": 0.0,
            "target_x": None,
            "target_y": None,
            "target_index": 0,
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
            "curvature_1pm": 0.0,
            "steer_limit": 0.0,
        }

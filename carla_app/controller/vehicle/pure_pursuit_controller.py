"""CARLA şerit merkezini izleyen sade Pure Pursuit kontrolcüsü."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PurePursuitController:
    """CARLA waypoint merkezlerinden seçilen hedef noktaya yönelir."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        # Tesla Model 3 için yaklaşık geometrik değerler.
        self.wheelbase_m = 2.87
        self.maximum_wheel_angle_rad = math.radians(35.0)

        # Hız yükseldikçe daha uzağa bakılır.
        self.minimum_lookahead_m = 4.0
        self.maximum_lookahead_m = 16.0
        self.base_lookahead_m = 4.0
        self.speed_lookahead_gain = 0.55

        # Direksiyon komutunun fiziksel olarak akıcı değişmesi için.
        self.steering_filter_time_s = 0.16
        self.maximum_steer_rate_per_s = 1.15
        self.previous_steer = 0.0

        self.last_info = self.empty_info()

    def run_step(self, state):
        """Referans şerit merkezine göre bu çevrimin direksiyonunu üretir."""
        path = state.get("reference_path", [])
        speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))

        if len(path) < 2:
            steer = self.smooth_steering(0.0)
            self.last_info = self.empty_info()
            return steer

        location = state["location"]
        yaw_rad = math.radians(float(state["yaw"]))

        projection = self.project_to_path(location.x, location.y, path)
        if projection is None:
            steer = self.smooth_steering(0.0)
            self.last_info = self.empty_info()
            return steer

        path_turn = self.preview_turn_amount(
            path,
            projection["segment_index"],
            preview_distance_m=18.0,
        )
        lookahead_m = self.calculate_lookahead(speed_mps, path_turn)

        target = self.target_at_distance(
            path,
            projection["segment_index"],
            projection["segment_fraction"],
            lookahead_m,
        )

        local_x, local_y = self.world_to_vehicle(
            target["x"],
            target["y"],
            location.x,
            location.y,
            yaw_rad,
        )

        # Hedef araç arkasına düşerse en küçük pozitif ileri mesafeyi kullan.
        local_x = max(0.20, local_x)
        target_distance_m = max(
            0.50,
            math.hypot(local_x, local_y),
        )

        # Pure Pursuit eğriliği: kappa = 2*y / Ld^2
        curvature = 2.0 * local_y / (target_distance_m**2)
        wheel_angle_rad = math.atan(self.wheelbase_m * curvature)
        desired_steer = clamp(
            wheel_angle_rad / self.maximum_wheel_angle_rad,
            -1.0,
            1.0,
        )

        # Tam düz yolda çok küçük rota sayısal gürültüsünü sıfırla.
        if abs(path_turn) < math.radians(0.8) and abs(desired_steer) < 0.012:
            desired_steer = 0.0

        steer = self.smooth_steering(desired_steer)

        self.last_info = {
            "controller": "pure_pursuit",
            "cross_track_error_m": float(projection["lateral_m"]),
            "heading_error_rad": float(
                normalize_angle(projection["path_heading_rad"] - yaw_rad)
            ),
            "raw_cross_track_error_m": float(projection["lateral_m"]),
            "raw_heading_error_rad": float(
                normalize_angle(projection["path_heading_rad"] - yaw_rad)
            ),
            "curvature_1pm": float(curvature),
            "target_index": int(target["segment_index"]),
            "target_x": float(target["x"]),
            "target_y": float(target["y"]),
            "target_local_x_m": float(local_x),
            "target_local_y_m": float(local_y),
            "target_distance_m": float(target_distance_m),
            "lookahead_m": float(lookahead_m),
            "path_turn_rad": float(path_turn),
            "raw_steer": float(desired_steer),
            "steer_limit": 1.0,
            # Eski gösterim ve test arayüzleri için sıfır değerler.
            "cross_track_term_rad": 0.0,
            "curve_feedforward_rad": 0.0,
            "straight_mode": abs(path_turn) < math.radians(1.2),
        }
        return steer

    def calculate_lookahead(self, speed_mps, path_turn_rad):
        """Düz yolda uzağa, keskin virajda daha yakına bakar."""
        lookahead = (
            self.base_lookahead_m
            + self.speed_lookahead_gain * max(0.0, float(speed_mps))
        )

        # Bu bir ek direksiyon kontrolü değildir.
        # Yalnızca Pure Pursuit hedef noktasının uzaklığını seçer.
        turn_ratio = clamp(
            abs(path_turn_rad) / math.radians(50.0),
            0.0,
            1.0,
        )
        lookahead *= 1.0 - 0.42 * turn_ratio

        return clamp(
            lookahead,
            self.minimum_lookahead_m,
            self.maximum_lookahead_m,
        )

    def smooth_steering(self, desired_steer):
        """Pure Pursuit çıktısını direksiyon aktüatörüne akıcı uygular."""
        alpha = clamp(
            self.dt / (self.steering_filter_time_s + self.dt),
            0.05,
            1.0,
        )
        filtered_target = self.previous_steer + alpha * (
            desired_steer - self.previous_steer
        )

        maximum_change = self.maximum_steer_rate_per_s * self.dt
        change = clamp(
            filtered_target - self.previous_steer,
            -maximum_change,
            maximum_change,
        )

        self.previous_steer = clamp(
            self.previous_steer + change,
            -1.0,
            1.0,
        )
        return self.previous_steer

    def project_to_path(self, point_x, point_y, path):
        """Aracı CARLA şerit merkez çizgisinin en yakın parçasına izdüşürür."""
        best = None

        # Rota yöneticisi arkada az sayıda nokta bırakır.
        # İlk 60 metre kontrol için yeterlidir ve yanlış uzak eşleşmeyi önler.
        search_path = path[:60]
        for index in range(len(search_path) - 1):
            start = search_path[index]
            end = search_path[index + 1]

            segment_x = end.x - start.x
            segment_y = end.y - start.y
            length_squared = segment_x**2 + segment_y**2
            if length_squared < 1e-8:
                continue

            fraction = clamp(
                (
                    (point_x - start.x) * segment_x
                    + (point_y - start.y) * segment_y
                )
                / length_squared,
                0.0,
                1.0,
            )

            projection_x = start.x + fraction * segment_x
            projection_y = start.y + fraction * segment_y
            error_x = point_x - projection_x
            error_y = point_y - projection_y
            distance_squared = error_x**2 + error_y**2

            if best is not None and distance_squared >= best["distance_squared"]:
                continue

            segment_length = math.sqrt(length_squared)
            lateral_m = (
                -segment_y * error_x + segment_x * error_y
            ) / segment_length

            best = {
                "distance_squared": distance_squared,
                "segment_index": index,
                "segment_fraction": fraction,
                "lateral_m": lateral_m,
                "path_heading_rad": math.atan2(segment_y, segment_x),
            }

        return best

    def target_at_distance(
        self,
        path,
        start_index,
        start_fraction,
        lookahead_m,
    ):
        """İzdüşümden lookahead kadar ilerideki şerit merkezini bulur."""
        start = path[start_index]
        end = path[start_index + 1]

        current_x = start.x + start_fraction * (end.x - start.x)
        current_y = start.y + start_fraction * (end.y - start.y)
        remaining = float(lookahead_m)

        for index in range(start_index, len(path) - 1):
            segment_start_x = current_x if index == start_index else path[index].x
            segment_start_y = current_y if index == start_index else path[index].y
            segment_end_x = path[index + 1].x
            segment_end_y = path[index + 1].y

            segment_length = math.hypot(
                segment_end_x - segment_start_x,
                segment_end_y - segment_start_y,
            )
            if segment_length < 1e-6:
                continue

            if remaining <= segment_length:
                fraction = remaining / segment_length
                return {
                    "x": segment_start_x
                    + fraction * (segment_end_x - segment_start_x),
                    "y": segment_start_y
                    + fraction * (segment_end_y - segment_start_y),
                    "segment_index": index,
                }

            remaining -= segment_length

        last = path[-1]
        return {
            "x": last.x,
            "y": last.y,
            "segment_index": len(path) - 2,
        }

    def preview_turn_amount(self, path, start_index, preview_distance_m):
        """Öndeki CARLA merkez yolunun toplam yön değişimini hesaplar."""
        if len(path) < 3:
            return 0.0

        first_heading = None
        last_heading = None
        travelled = 0.0

        for index in range(start_index, len(path) - 1):
            start = path[index]
            end = path[index + 1]
            segment_length = math.hypot(end.x - start.x, end.y - start.y)
            if segment_length < 1e-6:
                continue

            heading = math.atan2(end.y - start.y, end.x - start.x)
            if first_heading is None:
                first_heading = heading
            last_heading = heading
            travelled += segment_length

            if travelled >= preview_distance_m:
                break

        if first_heading is None or last_heading is None:
            return 0.0
        return normalize_angle(last_heading - first_heading)

    def world_to_vehicle(
        self,
        world_x,
        world_y,
        vehicle_x,
        vehicle_y,
        vehicle_yaw_rad,
    ):
        dx = world_x - vehicle_x
        dy = world_y - vehicle_y

        local_x = (
            dx * math.cos(vehicle_yaw_rad)
            + dy * math.sin(vehicle_yaw_rad)
        )
        local_y = (
            -dx * math.sin(vehicle_yaw_rad)
            + dy * math.cos(vehicle_yaw_rad)
        )
        return local_x, local_y

    def empty_info(self):
        return {
            "controller": "pure_pursuit",
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
            "raw_cross_track_error_m": 0.0,
            "raw_heading_error_rad": 0.0,
            "curvature_1pm": 0.0,
            "target_index": 0,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_local_x_m": 0.0,
            "target_local_y_m": 0.0,
            "target_distance_m": 0.0,
            "lookahead_m": 0.0,
            "path_turn_rad": 0.0,
            "raw_steer": 0.0,
            "steer_limit": 1.0,
            "cross_track_term_rad": 0.0,
            "curve_feedforward_rad": 0.0,
            "straight_mode": True,
        }

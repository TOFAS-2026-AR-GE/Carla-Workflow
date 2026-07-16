"""Front-axle Stanley path tracking controller."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class StanleyController:
    """Track the reference path with the standard Stanley steering law."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)
        self.wheelbase_m = 2.87
        self.maximum_wheel_angle_rad = math.radians(35.0)

        self.cross_track_gain = 0.9
        self.softening_speed_mps = 1.5
        self.maximum_cross_track_term_rad = math.radians(20.0)

        self.error_filter_time_s = 0.10
        self.filtered_cross_track_error_m = 0.0
        self.filtered_heading_error_rad = 0.0
        self.filter_initialized = False

        self.previous_steer = 0.0
        self.last_info = self._empty_info()

    def run_step(self, state):
        path = state.get("reference_path", [])
        speed = max(0.0, float(state.get("speed_mps", 0.0)))
        if len(path) < 2:
            steer = self._rate_limit(0.0, speed)
            self.last_info = self._empty_info()
            return steer

        location = state["location"]
        yaw = math.radians(float(state["yaw"]))
        front_x = location.x + self.wheelbase_m * math.cos(yaw)
        front_y = location.y + self.wheelbase_m * math.sin(yaw)

        projection = self._nearest_projection(front_x, front_y, path[:50])
        raw_cross_track_error = projection["cross_track_error_m"]
        raw_heading_error = normalize_angle(projection["path_heading_rad"] - yaw)
        self._filter_errors(raw_cross_track_error, raw_heading_error)

        # In CARLA coordinates positive cross-track error means the front axle
        # is on the right side of this forward path. Correction is to the left.
        cross_track_term = -math.atan2(
            self.cross_track_gain * self.filtered_cross_track_error_m,
            speed + self.softening_speed_mps,
        )
        cross_track_term = clamp(
            cross_track_term,
            -self.maximum_cross_track_term_rad,
            self.maximum_cross_track_term_rad,
        )

        curvature = self._path_curvature(path, projection["segment_index"])
        desired_wheel_angle = self.filtered_heading_error_rad + cross_track_term

        steer_limit = self._steering_limit(speed)
        desired_steer = clamp(
            desired_wheel_angle / self.maximum_wheel_angle_rad,
            -steer_limit,
            steer_limit,
        )
        steer = self._rate_limit(desired_steer, speed)

        self.last_info = {
            "cross_track_error_m": float(self.filtered_cross_track_error_m),
            "heading_error_rad": float(self.filtered_heading_error_rad),
            "raw_cross_track_error_m": float(raw_cross_track_error),
            "raw_heading_error_rad": float(raw_heading_error),
            "cross_track_term_rad": float(cross_track_term),
            "curvature_1pm": float(curvature),
            "target_index": int(projection["segment_index"]),
            "raw_steer": float(desired_steer),
            "steer_limit": float(steer_limit),
        }
        return steer

    def _filter_errors(self, cross_track_error, heading_error):
        if not self.filter_initialized:
            self.filtered_cross_track_error_m = cross_track_error
            self.filtered_heading_error_rad = heading_error
            self.filter_initialized = True
            return

        alpha = clamp(
            self.dt / (self.error_filter_time_s + self.dt),
            0.15,
            1.0,
        )
        self.filtered_cross_track_error_m += alpha * (
            cross_track_error - self.filtered_cross_track_error_m
        )
        heading_change = normalize_angle(
            heading_error - self.filtered_heading_error_rad
        )
        self.filtered_heading_error_rad = normalize_angle(
            self.filtered_heading_error_rad + alpha * heading_change
        )

    def _rate_limit(self, desired_steer, speed_mps):
        steer_rate_per_second = clamp(0.8 - 0.035 * speed_mps, 0.4, 0.8)
        maximum_change = steer_rate_per_second * self.dt
        change = clamp(
            desired_steer - self.previous_steer,
            -maximum_change,
            maximum_change,
        )
        self.previous_steer = clamp(self.previous_steer + change, -1.0, 1.0)
        return self.previous_steer

    @staticmethod
    def _steering_limit(speed_mps):
        return clamp(0.68 - 0.022 * speed_mps, 0.42, 0.68)

    @staticmethod
    def _nearest_projection(point_x, point_y, path):
        nearest = None
        for index, (start, end) in enumerate(zip(path, path[1:])):
            segment_x = end.x - start.x
            segment_y = end.y - start.y
            length_squared = segment_x**2 + segment_y**2
            if length_squared < 1e-8:
                continue

            offset_x = point_x - start.x
            offset_y = point_y - start.y
            fraction = clamp(
                (offset_x * segment_x + offset_y * segment_y) / length_squared,
                0.0,
                1.0,
            )
            projected_x = start.x + fraction * segment_x
            projected_y = start.y + fraction * segment_y
            error_x = point_x - projected_x
            error_y = point_y - projected_y
            distance_squared = error_x**2 + error_y**2
            if nearest is not None and distance_squared >= nearest["distance_squared"]:
                continue

            segment_length = math.sqrt(length_squared)
            signed_error = (-segment_y * error_x + segment_x * error_y) / segment_length
            nearest = {
                "distance_squared": distance_squared,
                "cross_track_error_m": signed_error,
                "path_heading_rad": math.atan2(segment_y, segment_x),
                "segment_index": index,
            }

        if nearest is not None:
            return nearest

        start, end = path[:2]
        return {
            "distance_squared": 0.0,
            "cross_track_error_m": 0.0,
            "path_heading_rad": math.atan2(end.y - start.y, end.x - start.x),
            "segment_index": 0,
        }

    @staticmethod
    def _path_curvature(path, segment_index):
        if len(path) < 5:
            return 0.0

        middle_index = clamp(segment_index + 3, 2, len(path) - 3)
        middle_index = int(middle_index)
        first = path[middle_index - 2]
        middle = path[middle_index]
        last = path[middle_index + 2]

        side_a = math.hypot(middle.x - first.x, middle.y - first.y)
        side_b = math.hypot(last.x - middle.x, last.y - middle.y)
        chord = math.hypot(last.x - first.x, last.y - first.y)
        denominator = side_a * side_b * chord
        if denominator < 1e-6:
            return 0.0

        twice_area = (middle.x - first.x) * (last.y - first.y) - (
            middle.y - first.y
        ) * (last.x - first.x)
        return 2.0 * twice_area / denominator

    @staticmethod
    def _empty_info():
        return {
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
            "raw_cross_track_error_m": 0.0,
            "raw_heading_error_rad": 0.0,
            "cross_track_term_rad": 0.0,
            "curvature_1pm": 0.0,
            "target_index": 0,
            "raw_steer": 0.0,
            "steer_limit": 0.0,
        }

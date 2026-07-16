"""Stable front-axle Stanley lateral controller."""

import math


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class StanleyController:
    """Lane controller with filtered feedback, feed-forward and rate limits."""

    def __init__(self, dt=0.05):
        self.dt = float(dt)
        self.wheelbase_m = 2.87
        self.max_wheel_angle_rad = math.radians(35.0)

        self.cross_track_gain = 0.9
        self.heading_gain = 1.0
        self.softening_speed_mps = 1.5
        self.maximum_cross_track_correction_rad = math.radians(20.0)
        self.curvature_feedforward_gain = 0.8

        self.previous_steer = 0.0
        self.filtered_cross_track_error = 0.0
        self.filtered_heading_error = 0.0
        self.filter_initialized = False

        self.last_info = self._empty_info()

    def run_step(self, state):
        reference_path = state.get("reference_path", [])
        speed = max(float(state.get("speed_mps", 0.0)), 0.0)

        if len(reference_path) < 2:
            self.previous_steer = self._rate_limit(0.0, speed)
            self.last_info = self._empty_info()
            return self.previous_steer

        yaw = math.radians(float(state["yaw"]))
        location = state["location"]

        # Stanley measures lateral error at the front axle.
        front_x = location.x + self.wheelbase_m * math.cos(yaw)
        front_y = location.y + self.wheelbase_m * math.sin(yaw)
        projection = self._project_to_path(
            front_x,
            front_y,
            reference_path[:45],
        )

        raw_cross_track_error = projection["cross_track_error_m"]
        raw_heading_error = normalize_angle(projection["heading_rad"] - yaw)
        self._filter_errors(raw_cross_track_error, raw_heading_error)

        cross_track_term = -math.atan2(
            self.cross_track_gain * self.filtered_cross_track_error,
            speed + self.softening_speed_mps,
        )
        cross_track_term = clamp(
            cross_track_term,
            -self.maximum_cross_track_correction_rad,
            self.maximum_cross_track_correction_rad,
        )

        target_index = projection["segment_index"]
        curvature = self._path_curvature(reference_path, target_index)
        feedforward = math.atan(self.wheelbase_m * curvature)
        desired_wheel_angle = (
            self.heading_gain * self.filtered_heading_error
            + cross_track_term
            + self.curvature_feedforward_gain * feedforward
        )

        dynamic_limit = self._steering_limit(speed)
        raw_steer = clamp(
            desired_wheel_angle / self.max_wheel_angle_rad,
            -dynamic_limit,
            dynamic_limit,
        )
        steer = self._rate_limit(raw_steer, speed)

        self.last_info = {
            "cross_track_error_m": float(self.filtered_cross_track_error),
            "heading_error_rad": float(self.filtered_heading_error),
            "raw_cross_track_error_m": float(raw_cross_track_error),
            "raw_heading_error_rad": float(raw_heading_error),
            "curvature_1pm": float(curvature),
            "target_index": int(target_index),
            "raw_steer": float(raw_steer),
            "steer_limit": float(dynamic_limit),
        }
        return steer

    def _filter_errors(self, cross_track_error, heading_error):
        if not self.filter_initialized:
            self.filtered_cross_track_error = cross_track_error
            self.filtered_heading_error = heading_error
            self.filter_initialized = True
            return

        # Roughly 0.1 s first-order filtering at the default 20 Hz tick rate.
        alpha = clamp(self.dt / (0.1 + self.dt), 0.2, 0.65)
        self.filtered_cross_track_error += alpha * (
            cross_track_error - self.filtered_cross_track_error
        )
        heading_delta = normalize_angle(heading_error - self.filtered_heading_error)
        self.filtered_heading_error = normalize_angle(
            self.filtered_heading_error + alpha * heading_delta
        )

    def _rate_limit(self, desired_steer, speed_mps):
        normalized_rate_per_second = clamp(
            0.8 - 0.035 * speed_mps,
            0.4,
            0.8,
        )
        maximum_change = normalized_rate_per_second * self.dt
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
    def _project_to_path(point_x, point_y, reference_path):
        best = None

        for index in range(len(reference_path) - 1):
            start = reference_path[index]
            end = reference_path[index + 1]
            segment_x = end.x - start.x
            segment_y = end.y - start.y
            length_squared = segment_x**2 + segment_y**2
            if length_squared < 1e-8:
                continue

            relative_x = point_x - start.x
            relative_y = point_y - start.y
            fraction = clamp(
                (relative_x * segment_x + relative_y * segment_y) / length_squared,
                0.0,
                1.0,
            )
            projection_x = start.x + fraction * segment_x
            projection_y = start.y + fraction * segment_y
            error_x = point_x - projection_x
            error_y = point_y - projection_y
            distance_squared = error_x**2 + error_y**2

            if best is None or distance_squared < best["distance_squared"]:
                segment_length = math.sqrt(length_squared)
                signed_error = (
                    -segment_y * error_x + segment_x * error_y
                ) / segment_length
                best = {
                    "distance_squared": distance_squared,
                    "cross_track_error_m": signed_error,
                    "heading_rad": math.atan2(segment_y, segment_x),
                    "segment_index": index,
                }

        if best is not None:
            return best

        start, end = reference_path[:2]
        return {
            "cross_track_error_m": 0.0,
            "heading_rad": math.atan2(end.y - start.y, end.x - start.x),
            "segment_index": 0,
        }

    @staticmethod
    def _path_curvature(reference_path, segment_index):
        if len(reference_path) < 5:
            return 0.0

        # Use points several metres apart instead of adjacent 1 m map samples;
        # this suppresses waypoint quantization spikes.
        middle_index = min(
            max(segment_index + 3, 2),
            len(reference_path) - 3,
        )
        first = reference_path[middle_index - 2]
        middle = reference_path[middle_index]
        last = reference_path[middle_index + 2]

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
            "curvature_1pm": 0.0,
            "target_index": 0,
        }

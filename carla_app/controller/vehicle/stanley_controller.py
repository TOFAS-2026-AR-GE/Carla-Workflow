import math


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


def normalize_angle(angle):
    return math.atan2(
        math.sin(angle),
        math.cos(angle),
    )


class StanleyController:
    """
    On aks tabanli Stanley yol takip kontrolcusu.

    Bilesenler:
    - heading error feedback
    - cross-track error feedback
    - curvature feed-forward
    - hiza bagli direksiyon siniri
    - direksiyon degisim hizi siniri
    """

    def __init__(self, dt=0.05):
        self.dt = float(dt)

        self.wheelbase_m = 2.87
        self.max_wheel_angle_rad = math.radians(
            35.0
        )

        # Stanley feedback parametreleri.
        self.cross_track_gain = 1.15
        self.heading_gain = 1.00
        self.softening_speed_mps = 1.50

        self.maximum_cross_track_correction_rad = (
            math.radians(22.0)
        )

        # Viraja girmeden once direksiyon beslemesi.
        self.curvature_feedforward_gain = 0.85

        self.previous_steer = 0.0

        self.last_info = {
            "cross_track_error_m": 0.0,
            "heading_error_rad": 0.0,
            "curvature_1pm": 0.0,
            "target_index": 0,
        }

    def run_step(self, state):
        reference_path = state["reference_path"]

        if len(reference_path) < 2:
            self.previous_steer = self._rate_limit(
                0.0,
                state["speed_mps"],
            )
            return self.previous_steer

        yaw = math.radians(
            float(state["yaw"])
        )
        speed = max(
            float(state["speed_mps"]),
            0.0,
        )
        location = state["location"]

        # Stanley yanal hatayi on aks noktasinda olcer.
        front_x = (
            location.x
            + self.wheelbase_m * math.cos(yaw)
        )
        front_y = (
            location.y
            + self.wheelbase_m * math.sin(yaw)
        )

        projection = self._project_to_path(
            front_x,
            front_y,
            reference_path,
        )

        path_heading = projection["heading_rad"]
        cross_track_error = projection[
            "cross_track_error_m"
        ]
        target_index = projection[
            "segment_index"
        ]

        heading_error = normalize_angle(
            path_heading - yaw
        )

        # CARLA'da pozitif y sag taraftir.
        # Arac yolun sagindaysa negatif steer ile sola donmelidir.
        cross_track_term = -math.atan2(
            self.cross_track_gain
            * cross_track_error,
            speed + self.softening_speed_mps,
        )

        cross_track_term = clamp(
            cross_track_term,
            -self.maximum_cross_track_correction_rad,
            self.maximum_cross_track_correction_rad,
        )

        curvature = self._path_curvature(
            reference_path,
            target_index,
        )

        feedforward_wheel_angle = math.atan(
            self.wheelbase_m * curvature
        )

        desired_wheel_angle = (
            self.heading_gain * heading_error
            + cross_track_term
            + self.curvature_feedforward_gain
            * feedforward_wheel_angle
        )

        raw_steer = (
            desired_wheel_angle
            / self.max_wheel_angle_rad
        )

        dynamic_limit = self._steering_limit(
            speed
        )

        raw_steer = clamp(
            raw_steer,
            -dynamic_limit,
            dynamic_limit,
        )

        steer = self._rate_limit(
            raw_steer,
            speed,
        )

        self.last_info = {
            "cross_track_error_m": float(
                cross_track_error
            ),
            "heading_error_rad": float(
                heading_error
            ),
            "curvature_1pm": float(curvature),
            "target_index": int(target_index),
            "raw_steer": float(raw_steer),
            "steer_limit": float(dynamic_limit),
        }

        return steer

    def _rate_limit(
        self,
        desired_steer,
        speed_mps,
    ):
        # Dusuk hizda daha hizli, yuksek hizda daha sakin
        # direksiyon degisimine izin ver.
        normalized_rate_per_second = clamp(
            0.85 - 0.035 * speed_mps,
            0.45,
            0.85,
        )

        maximum_change = (
            normalized_rate_per_second
            * self.dt
        )

        change = clamp(
            desired_steer
            - self.previous_steer,
            -maximum_change,
            maximum_change,
        )

        self.previous_steer = clamp(
            self.previous_steer + change,
            -1.0,
            1.0,
        )

        return self.previous_steer

    @staticmethod
    def _steering_limit(speed_mps):
        # Yuksek hizda ani ve buyuk direksiyon
        # komutlarini engeller.
        return clamp(
            0.72 - 0.022 * speed_mps,
            0.48,
            0.72,
        )

    @staticmethod
    def _project_to_path(
        point_x,
        point_y,
        reference_path,
    ):
        best = None

        for index in range(
            len(reference_path) - 1
        ):
            start = reference_path[index]
            end = reference_path[index + 1]

            segment_x = end.x - start.x
            segment_y = end.y - start.y

            length_squared = (
                segment_x**2
                + segment_y**2
            )

            if length_squared < 1e-8:
                continue

            relative_x = point_x - start.x
            relative_y = point_y - start.y

            fraction = clamp(
                (
                    relative_x * segment_x
                    + relative_y * segment_y
                )
                / length_squared,
                0.0,
                1.0,
            )

            projection_x = (
                start.x
                + fraction * segment_x
            )
            projection_y = (
                start.y
                + fraction * segment_y
            )

            error_x = point_x - projection_x
            error_y = point_y - projection_y

            distance_squared = (
                error_x**2 + error_y**2
            )

            if (
                best is None
                or distance_squared
                < best["distance_squared"]
            ):
                segment_length = math.sqrt(
                    length_squared
                )

                signed_error = (
                    -segment_y * error_x
                    + segment_x * error_y
                ) / segment_length

                best = {
                    "distance_squared": (
                        distance_squared
                    ),
                    "cross_track_error_m": (
                        signed_error
                    ),
                    "heading_rad": math.atan2(
                        segment_y,
                        segment_x,
                    ),
                    "segment_index": index,
                }

        if best is None:
            start = reference_path[0]
            end = reference_path[1]

            return {
                "cross_track_error_m": 0.0,
                "heading_rad": math.atan2(
                    end.y - start.y,
                    end.x - start.x,
                ),
                "segment_index": 0,
            }

        return best

    @staticmethod
    def _path_curvature(
        reference_path,
        segment_index,
    ):
        # Viraji biraz ileriden okuyarak
        # feedback gelmeden once feed-forward uygula.
        middle_index = min(
            max(segment_index + 3, 1),
            len(reference_path) - 2,
        )

        first = reference_path[
            middle_index - 1
        ]
        middle = reference_path[
            middle_index
        ]
        last = reference_path[
            middle_index + 1
        ]

        first_side = math.hypot(
            middle.x - first.x,
            middle.y - first.y,
        )
        second_side = math.hypot(
            last.x - middle.x,
            last.y - middle.y,
        )
        chord = math.hypot(
            last.x - first.x,
            last.y - first.y,
        )

        denominator = (
            first_side
            * second_side
            * chord
        )

        if denominator < 1e-6:
            return 0.0

        twice_area = (
            (middle.x - first.x)
            * (last.y - first.y)
            - (middle.y - first.y)
            * (last.x - first.x)
        )

        return (
            2.0
            * twice_area
            / denominator
        )
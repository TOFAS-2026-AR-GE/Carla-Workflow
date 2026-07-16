import math

from carla_app.controller.vehicle.tracking import (
    Tracker,
    polar_to_world,
)
from carla_app.perception.fusion import (
    fuse_detections_with_radar,
)


def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(value, maximum),
    )


def median(values):
    if not values:
        return None

    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2:
        return ordered[middle]

    return 0.5 * (
        ordered[middle - 1]
        + ordered[middle]
    )


def percentile(values, fraction):
    if not values:
        return None

    ordered = sorted(values)

    index = int(
        round(
            clamp(fraction, 0.0, 1.0)
            * (len(ordered) - 1)
        )
    )

    return ordered[index]


class LeadVehicleTracker:
    """
    Lead arac secimi.

    Iki bagimsiz kaynak kullanir:

    1. Kamera bbox + radar fusion + Kalman tracker
    2. Kamera calismasa bile dogrudan front radar clustering

    Kontrol sistemi kamera bbox'ina bagimli degildir.
    """

    def __init__(
        self,
        dt,
        image_width,
        camera_fov_deg,
    ):
        self.dt = float(dt)
        self.image_width = int(
            image_width
        )
        self.camera_fov_deg = float(
            camera_fov_deg
        )

        self.max_perception_age_frames = max(
            1,
            int(round(0.30 / self.dt)),
        )

        self.tracker = Tracker(
            gate_distance_m=5.0,
            max_misses=max(
                4,
                int(round(0.40 / self.dt)),
            ),
        )

        self.filtered_relative_speed = {}

        self.selected_track_id = None
        self.switch_advantage_m = 2.0

        self.previous_radar_distance = None
        self.previous_radar_speed = None
        self.radar_missing_ticks = 0

    def update(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        camera_measurements = (
            self._build_camera_radar_measurements(
                current_frame_id=current_frame_id,
                state=state,
                perception_result=perception_result,
                radar_frame_id=radar_frame_id,
                radar_points=radar_points,
            )
        )

        tracks = self.tracker.step(
            self.dt,
            camera_measurements,
        )

        tracked_lead = (
            self._select_tracked_lead(
                tracks,
                state,
            )
        )

        radar_lead = (
            self._select_direct_radar_lead(
                radar_points,
                state,
            )
        )

        return self._choose_safest_lead(
            tracked_lead,
            radar_lead,
        )

    def _build_camera_radar_measurements(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        # Kamera yoksa bos measurement doner,
        # fakat update() radar-only yolu calistirmaya devam eder.
        if perception_result is None:
            return []

        vehicles = perception_result.get(
            "vehicles",
            [],
        )

        if not vehicles:
            return []

        if (
            radar_frame_id is None
            or not radar_points
        ):
            return []

        detection_frame_id = (
            perception_result.get(
                "frame_id"
            )
        )

        if detection_frame_id is None:
            return []

        age_frames = (
            int(current_frame_id)
            - int(detection_frame_id)
        )

        if age_frames < 0:
            return []

        if (
            age_frames
            > self.max_perception_age_frames
        ):
            return []

        fused = fuse_detections_with_radar(
            detections=vehicles,
            detection_frame_id=(
                detection_frame_id
            ),
            radar_points=radar_points,
            radar_frame_id=radar_frame_id,
            image_width=self.image_width,
            camera_fov_deg=(
                self.camera_fov_deg
            ),
            fixed_delta_seconds=self.dt,
        )

        ego_location = state[
            "location"
        ]

        measurements = []

        for item in fused:
            if not item.get(
                "has_range",
                False,
            ):
                continue

            raw_range = item.get(
                "raw_range_m"
            )

            bearing = item.get(
                "bearing_deg"
            )

            if (
                raw_range is None
                or bearing is None
            ):
                continue

            raw_range = float(
                raw_range
            )
            bearing = float(
                bearing
            )

            if (
                raw_range <= 0.5
                or raw_range > 80.0
            ):
                continue

            if abs(bearing) > 16.0:
                continue

            world_x, world_y = (
                polar_to_world(
                    range_m=raw_range,
                    bearing_deg=bearing,
                    ego_x=ego_location.x,
                    ego_y=ego_location.y,
                    ego_yaw_deg=state["yaw"],
                )
            )

            measurements.append(
                {
                    "x": world_x,
                    "y": world_y,
                    "class_name": item.get(
                        "class_name",
                        "vehicle",
                    ),
                    "range_m": raw_range,
                    "bearing_deg": bearing,
                    "relative_velocity_mps": (
                        item.get(
                            "relative_velocity_mps"
                        )
                    ),
                }
            )

        return measurements

    def _select_tracked_lead(
        self,
        tracks,
        state,
    ):
        candidates = []

        for track in tracks:
            if not track.confirmed:
                continue

            if track.miss_count > 3:
                continue

            position = self._route_relative_position(
                world_x=track.x,
                world_y=track.y,
                state=state,
            )

            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]

            if (
                forward <= 0.5
                or forward > 80.0
            ):
                continue

            if not self._inside_ego_lane(
                lateral,
                state,
            ):
                continue

            relative_speed = (
                self._tracked_relative_speed(
                    track,
                    state,
                )
            )

            candidate = {
                "track_id": track.id,
                "class_name": (
                    track.class_name
                ),
                "distance_m": float(
                    forward
                ),
                "lateral_m": float(
                    lateral
                ),
                "lead_speed_mps": float(
                    max(
                        0.0,
                        state["speed_mps"]
                        + relative_speed,
                    )
                ),
                "relative_speed_mps": float(
                    relative_speed
                ),
                "source": "camera_radar_track",
                "radar_points": None,
            }

            candidates.append(candidate)

        return self._select_with_hysteresis(
            candidates
        )

    def _select_direct_radar_lead(
        self,
        radar_points,
        state,
    ):
        clusters = self._cluster_radar_points(
            radar_points
        )

        if not clusters:
            self.radar_missing_ticks += 1

            if self.radar_missing_ticks >= 3:
                self.previous_radar_distance = None
                self.previous_radar_speed = None

            return None

        ego_location = state[
            "location"
        ]

        candidates = []

        for cluster in clusters:
            depth_values = [
                point["depth_m"]
                for point in cluster
            ]

            bearing_values = [
                point["azimuth_deg"]
                for point in cluster
            ]

            velocity_values = [
                point[
                    "relative_velocity_mps"
                ]
                for point in cluster
            ]

            range_m = percentile(
                depth_values,
                0.20,
            )

            bearing_deg = median(
                bearing_values
            )

            radar_velocity = median(
                velocity_values
            )

            if (
                range_m is None
                or bearing_deg is None
                or radar_velocity is None
            ):
                continue

            # Tek radar noktasi normalde guvenilir sayilmaz.
            # Cok yakin bir cisimde ise guvenlik icin kabul edilir.
            if (
                len(cluster) < 2
                and range_m > 8.0
            ):
                continue

            world_x, world_y = (
                polar_to_world(
                    range_m=range_m,
                    bearing_deg=bearing_deg,
                    ego_x=ego_location.x,
                    ego_y=ego_location.y,
                    ego_yaw_deg=state["yaw"],
                )
            )

            position = self._route_relative_position(
                world_x=world_x,
                world_y=world_y,
                state=state,
            )

            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]

            if (
                forward <= 0.5
                or forward > 80.0
            ):
                continue

            if not self._inside_ego_lane(
                lateral,
                state,
            ):
                continue

            # CARLA radar velocity:
            # sensore dogru hiz pozitiftir.
            #
            # Controller convention:
            # lead_speed - ego_speed.
            # Ego araca yaklasiyorsa negatif olmalidir.
            relative_speed = clamp(
                -float(radar_velocity),
                -20.0,
                20.0,
            )

            candidates.append(
                {
                    "track_id": -1,
                    "class_name": (
                        "radar_obstacle"
                    ),
                    "distance_m": float(
                        range_m
                    ),
                    "lateral_m": float(
                        lateral
                    ),
                    "lead_speed_mps": float(
                        max(
                            0.0,
                            state["speed_mps"]
                            + relative_speed,
                        )
                    ),
                    "relative_speed_mps": float(
                        relative_speed
                    ),
                    "source": "radar_direct",
                    "radar_points": len(
                        cluster
                    ),
                    "bearing_deg": float(
                        bearing_deg
                    ),
                }
            )

        if not candidates:
            self.radar_missing_ticks += 1
            return None

        candidate = min(
            candidates,
            key=lambda item: item[
                "distance_m"
            ],
        )

        candidate = self._smooth_radar_lead(
            candidate
        )

        self.radar_missing_ticks = 0

        return candidate

    def _cluster_radar_points(
        self,
        radar_points,
    ):
        prepared = []

        for point in radar_points or []:
            depth = float(
                point.get(
                    "depth_m",
                    -1.0,
                )
            )

            azimuth = float(
                point.get(
                    "azimuth_deg",
                    999.0,
                )
            )

            altitude = float(
                point.get(
                    "altitude_deg",
                    999.0,
                )
            )

            velocity = float(
                point.get(
                    "relative_velocity_mps",
                    0.0,
                )
            )

            if not 0.5 < depth <= 80.0:
                continue

            if abs(azimuth) > 16.0:
                continue

            # Cok yuksek ve cok alcak yansimalari ele.
            if not -4.0 <= altitude <= 4.0:
                continue

            angle_rad = math.radians(
                azimuth
            )

            prepared.append(
                {
                    "depth_m": depth,
                    "azimuth_deg": azimuth,
                    "altitude_deg": altitude,
                    "relative_velocity_mps": (
                        velocity
                    ),
                    "x_m": (
                        depth
                        * math.cos(angle_rad)
                    ),
                    "y_m": (
                        depth
                        * math.sin(angle_rad)
                    ),
                }
            )

        if not prepared:
            return []

        clusters = []
        unvisited = set(
            range(len(prepared))
        )

        while unvisited:
            seed_index = unvisited.pop()

            cluster_indexes = {
                seed_index
            }

            stack = [
                seed_index
            ]

            while stack:
                current_index = (
                    stack.pop()
                )

                current = prepared[
                    current_index
                ]

                neighbours = []

                for other_index in unvisited:
                    other = prepared[
                        other_index
                    ]

                    minimum_depth = min(
                        current["depth_m"],
                        other["depth_m"],
                    )

                    spatial_gate = clamp(
                        0.75
                        + 0.02
                        * minimum_depth,
                        0.85,
                        1.75,
                    )

                    distance = math.hypot(
                        current["x_m"]
                        - other["x_m"],
                        current["y_m"]
                        - other["y_m"],
                    )

                    if distance <= spatial_gate:
                        neighbours.append(
                            other_index
                        )

                for neighbour in neighbours:
                    unvisited.remove(
                        neighbour
                    )
                    cluster_indexes.add(
                        neighbour
                    )
                    stack.append(
                        neighbour
                    )

            clusters.append(
                [
                    prepared[index]
                    for index
                    in cluster_indexes
                ]
            )

        return clusters

    def _route_relative_position(
        self,
        world_x,
        world_y,
        state,
    ):
        reference_path = state.get(
            "reference_path",
            [],
        )

        if len(reference_path) >= 2:
            ego_projection = (
                self._project_to_path(
                    state["location"].x,
                    state["location"].y,
                    reference_path,
                )
            )

            object_projection = (
                self._project_to_path(
                    world_x,
                    world_y,
                    reference_path,
                )
            )

            if (
                ego_projection is not None
                and object_projection is not None
            ):
                return {
                    "forward_m": (
                        object_projection["s_m"]
                        - ego_projection["s_m"]
                    ),
                    "lateral_m": (
                        object_projection[
                            "lateral_m"
                        ]
                    ),
                    "path_distance_m": (
                        object_projection[
                            "distance_to_path_m"
                        ]
                    ),
                }

        return self._world_to_ego(
            world_x=world_x,
            world_y=world_y,
            ego_x=state["location"].x,
            ego_y=state["location"].y,
            ego_yaw=math.radians(
                state["yaw"]
            ),
        )

    @staticmethod
    def _inside_ego_lane(
        lateral_m,
        state,
    ):
        lane_width = max(
            2.5,
            float(
                state.get(
                    "lane_width",
                    3.5,
                )
            ),
        )

        allowed_lateral = max(
            1.45,
            0.50 * lane_width + 0.20,
        )

        return (
            abs(lateral_m)
            <= allowed_lateral
        )

    def _tracked_relative_speed(
        self,
        track,
        state,
    ):
        radar_velocity = (
            track.last_relative_velocity_mps
        )

        if radar_velocity is not None:
            measured = -float(
                radar_velocity
            )
        else:
            yaw = math.radians(
                state["yaw"]
            )

            lead_speed = (
                track.vx * math.cos(yaw)
                + track.vy * math.sin(yaw)
            )

            measured = (
                lead_speed
                - state["speed_mps"]
            )

        measured = clamp(
            measured,
            -20.0,
            20.0,
        )

        previous = (
            self.filtered_relative_speed.get(
                track.id,
                measured,
            )
        )

        filtered = (
            0.65 * previous
            + 0.35 * measured
        )

        self.filtered_relative_speed[
            track.id
        ] = filtered

        return filtered

    def _smooth_radar_lead(
        self,
        candidate,
    ):
        distance = candidate[
            "distance_m"
        ]

        speed = candidate[
            "relative_speed_mps"
        ]

        if (
            self.previous_radar_distance
            is not None
            and abs(
                distance
                - self.previous_radar_distance
            )
            <= 8.0
        ):
            distance = (
                0.60 * distance
                + 0.40
                * self.previous_radar_distance
            )

        if (
            self.previous_radar_speed
            is not None
        ):
            speed = (
                0.40 * speed
                + 0.60
                * self.previous_radar_speed
            )

        candidate = dict(candidate)

        candidate["distance_m"] = float(
            distance
        )

        candidate[
            "relative_speed_mps"
        ] = float(speed)

        candidate["lead_speed_mps"] = max(
            0.0,
            candidate["lead_speed_mps"],
        )

        self.previous_radar_distance = (
            distance
        )

        self.previous_radar_speed = speed

        return candidate

    def _choose_safest_lead(
        self,
        tracked_lead,
        radar_lead,
    ):
        if tracked_lead is None:
            return radar_lead

        if radar_lead is None:
            return tracked_lead

        # Kamera-radar track ile direct radar birbirine
        # yakin mesafe veriyorsa semantik track'i kullan.
        if (
            abs(
                tracked_lead["distance_m"]
                - radar_lead["distance_m"]
            )
            <= 2.5
        ):
            return tracked_lead

        # Direct radar daha yakin bir engel goruyorsa
        # guvenlik acisindan onu kullan.
        if (
            radar_lead["distance_m"]
            < tracked_lead["distance_m"]
        ):
            return radar_lead

        return tracked_lead

    def _select_with_hysteresis(
        self,
        candidates,
    ):
        if not candidates:
            self.selected_track_id = None
            return None

        candidates.sort(
            key=lambda candidate: (
                candidate["distance_m"]
            )
        )

        nearest = candidates[0]

        current = next(
            (
                candidate
                for candidate in candidates
                if candidate["track_id"]
                == self.selected_track_id
            ),
            None,
        )

        if current is None:
            selected = nearest
        elif (
            nearest["track_id"]
            != current["track_id"]
            and nearest["distance_m"]
            < current["distance_m"]
            - self.switch_advantage_m
        ):
            selected = nearest
        else:
            selected = current

        self.selected_track_id = (
            selected["track_id"]
        )

        return selected

    @staticmethod
    def _project_to_path(
        point_x,
        point_y,
        reference_path,
    ):
        best = None
        cumulative_s = 0.0

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

            segment_length = math.sqrt(
                length_squared
            )

            relative_x = (
                point_x - start.x
            )
            relative_y = (
                point_y - start.y
            )

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

            error_x = (
                point_x - projection_x
            )
            error_y = (
                point_y - projection_y
            )

            distance_squared = (
                error_x**2
                + error_y**2
            )

            if (
                best is None
                or distance_squared
                < best["distance_squared"]
            ):
                lateral = (
                    -segment_y * error_x
                    + segment_x * error_y
                ) / segment_length

                best = {
                    "distance_squared": (
                        distance_squared
                    ),
                    "distance_to_path_m": (
                        math.sqrt(
                            distance_squared
                        )
                    ),
                    "s_m": (
                        cumulative_s
                        + fraction
                        * segment_length
                    ),
                    "lateral_m": lateral,
                }

            cumulative_s += segment_length

        return best

    @staticmethod
    def _world_to_ego(
        world_x,
        world_y,
        ego_x,
        ego_y,
        ego_yaw,
    ):
        dx = world_x - ego_x
        dy = world_y - ego_y

        forward = (
            dx * math.cos(ego_yaw)
            + dy * math.sin(ego_yaw)
        )

        lateral = (
            -dx * math.sin(ego_yaw)
            + dy * math.cos(ego_yaw)
        )

        return {
            "forward_m": forward,
            "lateral_m": lateral,
            "path_distance_m": abs(
                lateral
            ),
        }
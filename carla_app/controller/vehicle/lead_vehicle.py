"""Lead-vehicle selection from camera and front radar."""

import math

from carla_app.controller.vehicle.tracking import Tracker, polar_to_world
from carla_app.perception.fusion import fuse_detections_with_radar


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    index = int(round(clamp(fraction, 0.0, 1.0) * (len(ordered) - 1)))
    return ordered[index]


class LeadVehicleTracker:
    """Select the nearest stable object in the ego vehicle's route corridor.

    Camera/radar fusion is the preferred source. A temporally confirmed
    radar-only path remains available for collision safety when YOLO misses a
    frame or is temporarily unavailable.
    """

    def __init__(self, dt, image_width, camera_fov_deg):
        self.dt = float(dt)
        self.image_width = int(image_width)
        self.camera_fov_deg = float(camera_fov_deg)
        self.max_perception_age_frames = max(1, int(round(0.5 / self.dt)))

        self.tracker = Tracker(
            gate_distance_m=5.0,
            max_misses=max(4, int(round(0.4 / self.dt))),
        )
        self.last_fusion_key = None
        self.filtered_relative_speed = {}
        self.selected_track_id = None
        self.switch_advantage_m = 2.0

        self.radar_candidate = None
        self.radar_candidate_ticks = 0
        self.last_radar_lead = None
        self.radar_missing_ticks = 0
        self.emergency_obstacle = None

    def update(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        measurements = self._build_camera_radar_measurements(
            current_frame_id=current_frame_id,
            state=state,
            perception_result=perception_result,
            radar_frame_id=radar_frame_id,
            radar_points=radar_points,
        )
        tracks = self.tracker.step(self.dt, measurements)
        tracked_lead = self._select_tracked_lead(tracks, state)
        radar_lead = self._select_direct_radar_lead(radar_points, state)
        self.emergency_obstacle = self._select_emergency_radar_obstacle(
            radar_points,
            state,
        )
        return self._choose_safest_lead(tracked_lead, radar_lead)

    def get_emergency_obstacle(self):
        """Return the closest raw-radar hazard in the ego corridor.

        This last-resort AEB input deliberately does not depend on a camera
        bbox or a confirmed tracker. It is used only at short range or low TTC.
        """
        if self.emergency_obstacle is None:
            return None
        return dict(self.emergency_obstacle)

    def _build_camera_radar_measurements(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        if perception_result is None or radar_frame_id is None or not radar_points:
            return []

        vehicles = perception_result.get("vehicles", [])
        detection_frame_id = perception_result.get("frame_id")
        if not vehicles or detection_frame_id is None:
            return []

        age_frames = int(current_frame_id) - int(detection_frame_id)
        if age_frames < 0 or age_frames > self.max_perception_age_frames:
            return []

        fusion_key = (int(detection_frame_id), int(radar_frame_id))
        if fusion_key == self.last_fusion_key:
            return []
        self.last_fusion_key = fusion_key

        fused_detections = fuse_detections_with_radar(
            detections=vehicles,
            detection_frame_id=detection_frame_id,
            radar_points=radar_points,
            radar_frame_id=radar_frame_id,
            image_width=self.image_width,
            camera_fov_deg=self.camera_fov_deg,
            fixed_delta_seconds=self.dt,
        )

        ego_location = state["location"]
        measurements = []
        for detection in fused_detections:
            if not detection.get("has_range", False):
                continue

            range_m = detection.get("range_m")
            bearing_deg = detection.get("bearing_deg")
            if range_m is None or bearing_deg is None:
                continue

            range_m = float(range_m)
            bearing_deg = float(bearing_deg)
            if not 0.5 < range_m <= 80.0 or abs(bearing_deg) > 16.0:
                continue

            world_x, world_y = polar_to_world(
                range_m=range_m,
                bearing_deg=bearing_deg,
                ego_x=ego_location.x,
                ego_y=ego_location.y,
                ego_yaw_deg=state["yaw"],
            )
            measurements.append(
                {
                    "x": world_x,
                    "y": world_y,
                    "class_name": detection.get("class_name", "vehicle"),
                    "range_m": range_m,
                    "bearing_deg": bearing_deg,
                    "relative_velocity_mps": detection.get("relative_velocity_mps"),
                }
            )

        return measurements

    def _select_tracked_lead(self, tracks, state):
        candidates = []

        for track in tracks:
            if not track.confirmed or track.miss_count > 3:
                continue

            position = self._route_relative_position(track.x, track.y, state)
            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]
            if not 0.5 < forward <= 80.0:
                continue
            if not self._inside_ego_lane(lateral, state):
                continue

            relative_speed = self._tracked_relative_speed(track, state)
            candidates.append(
                {
                    "track_id": track.id,
                    "class_name": track.class_name,
                    "distance_m": float(forward),
                    "lateral_m": float(lateral),
                    "lead_speed_mps": max(
                        0.0, float(state["speed_mps"]) + relative_speed
                    ),
                    "relative_speed_mps": float(relative_speed),
                    "source": "camera_radar_track",
                    "radar_points": None,
                }
            )

        return self._select_with_hysteresis(candidates)

    def _select_direct_radar_lead(self, radar_points, state):
        candidates = self._radar_candidates(radar_points, state)
        raw_candidate = min(
            candidates,
            key=lambda item: item["distance_m"],
            default=None,
        )

        if raw_candidate is None:
            self.radar_candidate = None
            self.radar_candidate_ticks = 0
            return self._hold_last_radar_lead()

        if self._same_radar_target(self.radar_candidate, raw_candidate):
            self.radar_candidate_ticks += 1
        else:
            self.radar_candidate_ticks = 1
        self.radar_candidate = raw_candidate

        closing_speed = max(0.0, -raw_candidate["relative_speed_mps"])
        ttc = (
            raw_candidate["distance_m"] / closing_speed
            if closing_speed > 0.1
            else math.inf
        )
        immediate_hazard = raw_candidate["distance_m"] <= 5.0 or ttc <= 0.8

        if self.radar_candidate_ticks < 2 and not immediate_hazard:
            return self._hold_last_radar_lead()

        lead = self._smooth_radar_lead(raw_candidate)
        self.last_radar_lead = lead
        self.radar_missing_ticks = 0
        return lead

    def _radar_candidates(self, radar_points, state):
        candidates = []
        ego_location = state["location"]

        for cluster in self._cluster_radar_points(radar_points):
            depths = [point["depth_m"] for point in cluster]
            bearings = [point["azimuth_deg"] for point in cluster]
            velocities = [point["relative_velocity_mps"] for point in cluster]
            range_m = percentile(depths, 0.2)
            bearing_deg = median(bearings)
            radar_velocity = median(velocities)

            if range_m is None or bearing_deg is None or radar_velocity is None:
                continue
            if len(cluster) < 2 and range_m > 6.0:
                continue
            if len(cluster) < 3 and range_m > 35.0:
                continue

            world_x, world_y = polar_to_world(
                range_m=range_m,
                bearing_deg=bearing_deg,
                ego_x=ego_location.x,
                ego_y=ego_location.y,
                ego_yaw_deg=state["yaw"],
            )
            position = self._route_relative_position(world_x, world_y, state)
            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]
            if not 0.5 < forward <= 80.0:
                continue
            if not self._inside_ego_lane(lateral, state):
                continue

            # CARLA's signed radar value already matches the controller's
            # convention in this setup: negative means the range is closing,
            # positive means the target is pulling away. Do not invert it.
            relative_speed = clamp(float(radar_velocity), -20.0, 20.0)
            candidates.append(
                {
                    "track_id": -1,
                    "class_name": "radar_obstacle",
                    "distance_m": float(forward),
                    "lateral_m": float(lateral),
                    "lead_speed_mps": max(
                        0.0, float(state["speed_mps"]) + relative_speed
                    ),
                    "relative_speed_mps": relative_speed,
                    "source": "radar_direct",
                    "radar_points": len(cluster),
                    "bearing_deg": float(bearing_deg),
                }
            )

        return candidates

    def _select_emergency_radar_obstacle(self, radar_points, state):
        """Keep AEB alive when YOLO or normal radar clustering misses."""
        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        corridor_half_width = min(1.8, max(1.2, 0.5 * lane_width))
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        candidates = []

        for point in radar_points or []:
            try:
                depth = float(point["depth_m"])
                azimuth = float(point["azimuth_deg"])
                altitude = float(point.get("altitude_deg", 0.0))
                radar_velocity = float(point.get("relative_velocity_mps", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

            values = (depth, azimuth, altitude, radar_velocity)
            if not all(math.isfinite(value) for value in values):
                continue
            if not 0.5 < depth <= 20.0:
                continue
            if abs(azimuth) > 16.0 or abs(altitude) > 6.0:
                continue

            angle_rad = math.radians(azimuth)
            forward = depth * math.cos(angle_rad)
            lateral = depth * math.sin(angle_rad)
            if forward <= 0.5 or abs(lateral) > corridor_half_width:
                continue

            relative_speed = clamp(radar_velocity, -25.0, 20.0)
            closing_speed = max(0.0, -relative_speed)
            ttc = forward / closing_speed if closing_speed > 0.1 else math.inf

            # A single raw point is accepted only in the near-field or when
            # its measured closing rate is already safety-critical.
            if forward > 8.0 and ttc > 1.8:
                continue

            candidates.append(
                {
                    "track_id": -2,
                    "class_name": "radar_emergency_obstacle",
                    "distance_m": float(forward),
                    "lateral_m": float(lateral),
                    "lead_speed_mps": max(0.0, ego_speed + relative_speed),
                    "relative_speed_mps": float(relative_speed),
                    "source": "radar_emergency",
                    "radar_points": 1,
                    "bearing_deg": float(azimuth),
                    "ttc_s": float(ttc),
                }
            )

        return min(
            candidates,
            key=lambda candidate: (
                candidate["ttc_s"],
                candidate["distance_m"],
            ),
            default=None,
        )

    def _cluster_radar_points(self, radar_points):
        prepared = []
        for point in radar_points or []:
            depth = float(point.get("depth_m", -1.0))
            azimuth = float(point.get("azimuth_deg", 999.0))
            altitude = float(point.get("altitude_deg", 999.0))
            velocity = float(point.get("relative_velocity_mps", 0.0))

            if not 0.5 < depth <= 80.0:
                continue
            if abs(azimuth) > 16.0 or abs(altitude) > 6.0:
                continue

            angle_rad = math.radians(azimuth)
            prepared.append(
                {
                    "depth_m": depth,
                    "azimuth_deg": azimuth,
                    "altitude_deg": altitude,
                    "relative_velocity_mps": velocity,
                    "x_m": depth * math.cos(angle_rad),
                    "y_m": depth * math.sin(angle_rad),
                }
            )

        clusters = []
        unvisited = set(range(len(prepared)))
        while unvisited:
            seed = unvisited.pop()
            cluster_indexes = {seed}
            stack = [seed]

            while stack:
                current = prepared[stack.pop()]
                neighbours = []
                for other_index in unvisited:
                    other = prepared[other_index]
                    minimum_depth = min(current["depth_m"], other["depth_m"])
                    spatial_gate = clamp(0.75 + 0.02 * minimum_depth, 0.85, 1.75)
                    distance = math.hypot(
                        current["x_m"] - other["x_m"],
                        current["y_m"] - other["y_m"],
                    )
                    if distance <= spatial_gate:
                        neighbours.append(other_index)

                for neighbour in neighbours:
                    unvisited.remove(neighbour)
                    cluster_indexes.add(neighbour)
                    stack.append(neighbour)

            clusters.append([prepared[index] for index in cluster_indexes])

        return clusters

    def _same_radar_target(self, previous, current):
        if previous is None:
            return False
        expected_distance = (
            previous["distance_m"] + previous["relative_speed_mps"] * self.dt
        )
        distance_gate = max(2.0, 0.08 * current["distance_m"])
        return (
            abs(current["distance_m"] - expected_distance) <= distance_gate
            and abs(current["lateral_m"] - previous["lateral_m"]) <= 0.8
        )

    def _hold_last_radar_lead(self):
        if self.last_radar_lead is None:
            return None

        self.radar_missing_ticks += 1
        if self.radar_missing_ticks > 2:
            self.last_radar_lead = None
            return None

        lead = dict(self.last_radar_lead)
        lead["distance_m"] = max(
            0.0,
            lead["distance_m"] + lead["relative_speed_mps"] * self.dt,
        )
        lead["source"] = "radar_predicted"
        self.last_radar_lead = lead
        return lead

    def _smooth_radar_lead(self, candidate):
        lead = dict(candidate)
        previous = self.last_radar_lead

        if previous is not None and self._same_radar_target(previous, lead):
            lead["distance_m"] = (
                0.65 * lead["distance_m"] + 0.35 * previous["distance_m"]
            )
            lead["relative_speed_mps"] = (
                0.4 * lead["relative_speed_mps"] + 0.6 * previous["relative_speed_mps"]
            )

        lead["lead_speed_mps"] = max(
            0.0,
            lead["lead_speed_mps"],
        )
        return lead

    def _tracked_relative_speed(self, track, state):
        radar_velocity = track.last_relative_velocity_mps
        if radar_velocity is not None:
            measured = float(radar_velocity)
        else:
            yaw = math.radians(state["yaw"])
            lead_speed = track.vx * math.cos(yaw) + track.vy * math.sin(yaw)
            measured = lead_speed - float(state["speed_mps"])

        measured = clamp(measured, -20.0, 20.0)
        previous = self.filtered_relative_speed.get(track.id, measured)
        filtered = 0.65 * previous + 0.35 * measured
        self.filtered_relative_speed[track.id] = filtered
        return filtered

    def _choose_safest_lead(self, tracked_lead, radar_lead):
        if tracked_lead is None:
            return radar_lead
        if radar_lead is None:
            return tracked_lead

        # Camera identity is useful, but its Kalman range can lag when a bbox
        # disappears and returns. Keep that identity while using the closer
        # front-radar range and velocity. This prevents 4 m -> 7 m -> 5 m
        # source-switch jumps from reaching the controller.
        if abs(tracked_lead["distance_m"] - radar_lead["distance_m"]) <= 3.0:
            lead = dict(tracked_lead)
            if radar_lead["distance_m"] < tracked_lead["distance_m"]:
                lead["distance_m"] = radar_lead["distance_m"]
                lead["relative_speed_mps"] = radar_lead["relative_speed_mps"]
                lead["lead_speed_mps"] = radar_lead["lead_speed_mps"]
                lead["radar_points"] = radar_lead["radar_points"]
            return lead

        return min(
            (tracked_lead, radar_lead),
            key=lambda candidate: candidate["distance_m"],
        )

    def _select_with_hysteresis(self, candidates):
        if not candidates:
            self.selected_track_id = None
            return None

        candidates.sort(key=lambda candidate: candidate["distance_m"])
        nearest = candidates[0]
        current = next(
            (
                candidate
                for candidate in candidates
                if candidate["track_id"] == self.selected_track_id
            ),
            None,
        )

        if current is None:
            selected = nearest
        elif (
            nearest["track_id"] != current["track_id"]
            and nearest["distance_m"] < current["distance_m"] - self.switch_advantage_m
        ):
            selected = nearest
        else:
            selected = current

        self.selected_track_id = selected["track_id"]
        return selected

    @staticmethod
    def _inside_ego_lane(lateral_m, state):
        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        allowed_lateral = max(1.45, 0.5 * lane_width + 0.15)
        return abs(lateral_m) <= allowed_lateral

    def _route_relative_position(self, world_x, world_y, state):
        reference_path = state.get("reference_path", [])
        if len(reference_path) >= 2:
            ego = self._project_to_path(
                state["location"].x,
                state["location"].y,
                reference_path,
            )
            obstacle = self._project_to_path(world_x, world_y, reference_path)
            if ego is not None and obstacle is not None:
                return {
                    "forward_m": obstacle["s_m"] - ego["s_m"],
                    "lateral_m": obstacle["lateral_m"],
                    "path_distance_m": obstacle["distance_to_path_m"],
                }

        return self._world_to_ego(
            world_x=world_x,
            world_y=world_y,
            ego_x=state["location"].x,
            ego_y=state["location"].y,
            ego_yaw=math.radians(state["yaw"]),
        )

    @staticmethod
    def _project_to_path(point_x, point_y, reference_path):
        best = None
        cumulative_s = 0.0

        for index in range(len(reference_path) - 1):
            start = reference_path[index]
            end = reference_path[index + 1]
            segment_x = end.x - start.x
            segment_y = end.y - start.y
            length_squared = segment_x**2 + segment_y**2

            if length_squared < 1e-8:
                continue

            segment_length = math.sqrt(length_squared)
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
                lateral = (-segment_y * error_x + segment_x * error_y) / segment_length
                best = {
                    "distance_squared": distance_squared,
                    "distance_to_path_m": math.sqrt(distance_squared),
                    "s_m": cumulative_s + fraction * segment_length,
                    "lateral_m": lateral,
                }

            cumulative_s += segment_length

        return best

    @staticmethod
    def _world_to_ego(world_x, world_y, ego_x, ego_y, ego_yaw):
        dx = world_x - ego_x
        dy = world_y - ego_y
        forward = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
        lateral = -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
        return {
            "forward_m": forward,
            "lateral_m": lateral,
            "path_distance_m": abs(lateral),
        }

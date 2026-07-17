"""Ön kamera ve ön radardan takip edilecek aracı seçer.

İşlem sırası açıktır: eski ve zemin radar dönüşleri elenir, kamera ile radar
birleştirilir, hedefler zamana göre takip edilir ve sürüş koridorundaki en
güvenli ön araç seçilir.
"""

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
    """Aracın gideceği koridordaki en yakın ve kararlı hedefi seçer.

    Ana kaynak, ön kamera kutusu ile radar mesafesinin birleştirilmesidir.
    Kamera bir karede hedefi kaçırırsa iki kare boyunca doğrulanan radar
    kümesi geçici olarak yedek kaynak olur.
    """

    def __init__(
        self,
        dt,
        image_width,
        camera_fov_deg,
        radar_height_m=1.0,
        radar_pitch_deg=2.0,
    ):
        self.dt = float(dt)
        self.image_width = int(image_width)
        self.camera_fov_deg = float(camera_fov_deg)
        self.max_perception_age_frames = max(1, int(round(0.5 / self.dt)))

        self.radar_height_m = max(0.1, float(radar_height_m))
        self.radar_pitch_deg = float(radar_pitch_deg)
        self.minimum_obstacle_height_m = 0.25
        self.radar_diagnostics = {
            "raw_points": 0,
            "usable_points": 0,
            "ground_rejected": 0,
            "invalid_rejected": 0,
        }

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
        self.last_direct_radar_frame_id = None
        self.emergency_obstacle = None

    def update(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        """Yeni kamera-radar verisini işleyip bu çevrimin ön aracını seçer."""
        radar_age_frames = self.frame_age(current_frame_id, radar_frame_id)
        maximum_radar_age = max(2, int(round(0.20 / self.dt)))
        radar_is_fresh = (
            radar_age_frames is not None
            and 0 <= radar_age_frames <= maximum_radar_age
        )
        if not radar_is_fresh:
            radar_points = []

        radar_points = self.filter_ground_returns(radar_points)
        self.radar_diagnostics["frame_age"] = radar_age_frames
        self.radar_diagnostics["fresh"] = radar_is_fresh
        measurements = self.build_camera_radar_measurements(
            current_frame_id=current_frame_id,
            state=state,
            perception_result=perception_result,
            radar_frame_id=radar_frame_id,
            radar_points=radar_points,
        )
        tracks = self.tracker.step(self.dt, measurements)
        tracked_lead = self.select_tracked_lead(tracks, state)
        radar_lead = self.select_direct_radar_lead(
            radar_points,
            state,
            radar_frame_id,
        )
        self.emergency_obstacle = self.select_emergency_radar_obstacle(
            radar_points,
            state,
            radar_frame_id,
        )
        return self.choose_safest_lead(tracked_lead, radar_lead)

    def frame_age(self, current_frame_id, measurement_frame_id):
        if measurement_frame_id is None:
            return None
        return int(current_frame_id) - int(measurement_frame_id)

    def get_radar_diagnostics(self):
        return dict(self.radar_diagnostics)

    def get_emergency_obstacle(self):
        """Aracın koridorundaki en yakın ham radar tehlikesini verir.

        Bu son güvenlik yolu kamera kutusuna veya onaylanmış takibe bağlı
        değildir. Yalnızca çok yakın mesafede veya düşük TTC'de kullanılır.
        """
        if self.emergency_obstacle is None:
            return None
        return dict(self.emergency_obstacle)

    def filter_ground_returns(self, radar_points):
        """Fiziksel engel yüksekliğinin altındaki zemin dönüşlerini eler."""
        usable_points = []
        ground_rejected = 0
        invalid_rejected = 0

        for point in radar_points or []:
            try:
                depth = float(point["depth_m"])
                altitude = float(point.get("altitude_deg", 0.0))
            except (KeyError, TypeError, ValueError):
                invalid_rejected += 1
                continue

            if not math.isfinite(depth) or not math.isfinite(altitude) or depth <= 0.0:
                invalid_rejected += 1
                continue

            elevation = math.radians(self.radar_pitch_deg + altitude)
            hit_height = self.radar_height_m + depth * math.sin(elevation)
            if hit_height < self.minimum_obstacle_height_m:
                ground_rejected += 1
                continue

            usable_point = dict(point)
            usable_point["estimated_hit_height_m"] = float(hit_height)
            usable_points.append(usable_point)

        self.radar_diagnostics = {
            "raw_points": len(radar_points or []),
            "usable_points": len(usable_points),
            "ground_rejected": ground_rejected,
            "invalid_rejected": invalid_rejected,
        }
        return usable_points

    def build_camera_radar_measurements(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        """Kamera kutusunu radar mesafesiyle birleştirip dünyaya taşır."""
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
                    "measurement_frame_id": int(radar_frame_id),
                }
            )

        return measurements

    def select_tracked_lead(self, tracks, state):
        """Doğrulanmış takiplerden sürüş koridorundaki ön aracı seçer."""
        candidates = []

        for track in tracks:
            if not track.confirmed or track.miss_count > 3:
                continue

            position = self.route_relative_position(track.x, track.y, state)
            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]
            if not 0.5 < forward <= 80.0:
                continue
            if not self.inside_driving_corridor(lateral, state):
                continue

            relative_speed = self.tracked_relative_speed(track, state)
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
                    "measurement_frame_id": track.last_measurement_frame_id,
                }
            )

        return self.select_with_hysteresis(candidates)

    def select_direct_radar_lead(self, radar_points, state, radar_frame_id):
        """Kamera geçici kaybolursa doğrulanmış radar kümesini yedek seçer."""
        if radar_frame_id is None:
            return self.hold_last_radar_lead()

        radar_frame_id = int(radar_frame_id)
        if radar_frame_id == self.last_direct_radar_frame_id:
            # Aynı sensör karesi ikinci bir kanıt değildir. Kısa geri çağrı
            # gecikmesinde son hedefi fizik modeliyle bir çevrim ilerletiriz.
            return self.hold_last_radar_lead()

        self.last_direct_radar_frame_id = radar_frame_id
        candidates = self.radar_candidates(
            radar_points,
            state,
            radar_frame_id,
        )
        raw_candidate = None
        for candidate in candidates:
            if raw_candidate is None:
                raw_candidate = candidate
            elif candidate["distance_m"] < raw_candidate["distance_m"]:
                raw_candidate = candidate

        if raw_candidate is None:
            self.radar_candidate = None
            self.radar_candidate_ticks = 0
            return self.hold_last_radar_lead()

        if self.same_radar_target(self.radar_candidate, raw_candidate):
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
            return self.hold_last_radar_lead()

        lead = self.smooth_radar_lead(raw_candidate)
        self.last_radar_lead = lead
        self.radar_missing_ticks = 0
        return lead

    def radar_candidates(self, radar_points, state, radar_frame_id):
        """Radar kümelerini mesafe ve şerit koşullarından geçirir."""
        candidates = []
        ego_location = state["location"]

        for cluster in self.cluster_radar_points(radar_points):
            depths = []
            bearings = []
            velocities = []
            for point in cluster:
                depths.append(point["depth_m"])
                bearings.append(point["azimuth_deg"])
                velocities.append(point["relative_velocity_mps"])
            range_m = percentile(depths, 0.2)
            bearing_deg = median(bearings)
            radar_velocity = median(velocities)

            if range_m is None or bearing_deg is None or radar_velocity is None:
                continue
            if len(cluster) < 2:
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
            position = self.route_relative_position(world_x, world_y, state)
            if position is None:
                continue

            forward = position["forward_m"]
            lateral = position["lateral_m"]
            if not 0.5 < forward <= 80.0:
                continue
            if not self.inside_driving_corridor(lateral, state):
                continue

            # CARLA radar hızının işareti bu projedeki kuralla aynıdır:
            # negatif değer mesafenin kapandığını, pozitif değer ise
            # hedefin uzaklaştığını gösterir. İşareti ters çevirmeyin.
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
                    "measurement_frame_id": int(radar_frame_id),
                }
            )

        return candidates

    def select_emergency_radar_obstacle(
        self,
        radar_points,
        state,
        radar_frame_id,
    ):
        """Kamera veya radar kümesi kaçırsa bile acil fren yolunu açık tutar."""
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        ego_location = state["location"]
        candidates = []

        for point in radar_points or []:
            try:
                depth = float(point["depth_m"])
                azimuth = float(point["azimuth_deg"])
                altitude = float(point.get("altitude_deg", 0.0))
                radar_velocity = float(point.get("relative_velocity_mps", 0.0))
            except (KeyError, TypeError, ValueError):
                continue

            values_are_finite = (
                math.isfinite(depth)
                and math.isfinite(azimuth)
                and math.isfinite(altitude)
                and math.isfinite(radar_velocity)
            )
            if not values_are_finite:
                continue
            if not 0.5 < depth <= 20.0:
                continue
            if abs(azimuth) > 16.0 or abs(altitude) > 6.0:
                continue

            angle_rad = math.radians(azimuth)
            forward = depth * math.cos(angle_rad)
            lateral = depth * math.sin(angle_rad)
            if forward <= 0.5 or not self.inside_driving_corridor(lateral, state):
                continue

            world_x, world_y = polar_to_world(
                range_m=depth,
                bearing_deg=azimuth,
                ego_x=ego_location.x,
                ego_y=ego_location.y,
                ego_yaw_deg=state["yaw"],
            )
            route_position = self.route_relative_position(world_x, world_y, state)
            if route_position is None:
                continue
            if not self.inside_driving_corridor(route_position["lateral_m"], state):
                continue

            relative_speed = clamp(radar_velocity, -25.0, 20.0)
            closing_speed = max(0.0, -relative_speed)
            ttc = forward / closing_speed if closing_speed > 0.1 else math.inf

            # Tek bir ham nokta yalnızca çok yakınsa veya ölçülen yaklaşma
            # hızı güvenlik açısından kritikse acil fren adayı olabilir.
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
                    "measurement_frame_id": (
                        int(radar_frame_id) if radar_frame_id is not None else None
                    ),
                }
            )

        selected = None
        for candidate in candidates:
            if selected is None:
                selected = candidate
                continue
            candidate_priority = (
                candidate["ttc_s"],
                candidate["distance_m"],
            )
            selected_priority = (
                selected["ttc_s"],
                selected["distance_m"],
            )
            if candidate_priority < selected_priority:
                selected = candidate
        return selected

    def cluster_radar_points(self, radar_points):
        """Birbirine yakın radar noktalarını aynı fiziksel hedefte toplar."""
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

            cluster = []
            for index in cluster_indexes:
                cluster.append(prepared[index])
            clusters.append(cluster)

        return clusters

    def same_radar_target(self, previous, current):
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

    def hold_last_radar_lead(self):
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

    def smooth_radar_lead(self, candidate):
        lead = dict(candidate)
        previous = self.last_radar_lead

        if previous is not None and self.same_radar_target(previous, lead):
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

    def tracked_relative_speed(self, track, state):
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

    def choose_safest_lead(self, tracked_lead, radar_lead):
        """Kamera kimliğini koruyarak daha yakın ve güvenli ölçümü seçer."""
        if tracked_lead is None:
            return radar_lead
        if radar_lead is None:
            return tracked_lead

        # Kamera hedef kimliğini verir. Kamera kutusu kaybolup geri geldiğinde
        # Kalman mesafesi biraz geriden gelebilir. Bu durumda kamera kimliğini
        # korurken daha yakın radar mesafesi ve hızını kullanırız. Böylece
        # kaynak değişimindeki 4 m -> 7 m -> 5 m sıçrama kontrolcüye gitmez.
        if abs(tracked_lead["distance_m"] - radar_lead["distance_m"]) <= 3.0:
            lead = dict(tracked_lead)
            if radar_lead["distance_m"] < tracked_lead["distance_m"]:
                lead["distance_m"] = radar_lead["distance_m"]
                lead["relative_speed_mps"] = radar_lead["relative_speed_mps"]
                lead["lead_speed_mps"] = radar_lead["lead_speed_mps"]
                lead["radar_points"] = radar_lead["radar_points"]
            return lead

        if tracked_lead["distance_m"] <= radar_lead["distance_m"]:
            return tracked_lead
        return radar_lead

    def select_with_hysteresis(self, candidates):
        """Benzer mesafedeki iki hedef arasında sürekli geçişi önler."""
        if not candidates:
            self.selected_track_id = None
            return None

        nearest = candidates[0]
        current = None
        for candidate in candidates:
            if candidate["distance_m"] < nearest["distance_m"]:
                nearest = candidate
            if candidate["track_id"] == self.selected_track_id:
                if current is None:
                    current = candidate
                elif candidate["distance_m"] < current["distance_m"]:
                    current = candidate

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

    def inside_driving_corridor(self, lateral_m, state):
        """Noktanın ego aracının sürüş koridorunda olduğunu sınar."""
        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        vehicle_half_width = max(
            0.70,
            float(state.get("vehicle_half_width_m", 0.95)),
        )

        # Yalnızca ego aracının kapladığı koridoru ve küçük bir güvenlik
        # payını kabul et. Böylece komşu şerit, kaldırım ve yol kenarı
        # dönüşleri normal takip aracı olarak seçilmez.
        lane_boundary = 0.5 * lane_width - 0.15
        vehicle_corridor = vehicle_half_width + 0.40
        allowed_lateral = max(0.90, min(lane_boundary, vehicle_corridor))
        return abs(lateral_m) <= allowed_lateral

    def route_relative_position(self, world_x, world_y, state):
        reference_path = state.get("reference_path", [])
        if len(reference_path) >= 2:
            ego = self.project_to_path(
                state["location"].x,
                state["location"].y,
                reference_path,
            )
            obstacle = self.project_to_path(world_x, world_y, reference_path)
            if ego is not None and obstacle is not None:
                return {
                    "forward_m": obstacle["s_m"] - ego["s_m"],
                    "lateral_m": obstacle["lateral_m"],
                    "path_distance_m": obstacle["distance_to_path_m"],
                }

        return self.world_to_ego(
            world_x=world_x,
            world_y=world_y,
            ego_x=state["location"].x,
            ego_y=state["location"].y,
            ego_yaw=math.radians(state["yaw"]),
        )

    def project_to_path(self, point_x, point_y, reference_path):
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

    def world_to_ego(self, world_x, world_y, ego_x, ego_y, ego_yaw):
        dx = world_x - ego_x
        dy = world_y - ego_y
        forward = dx * math.cos(ego_yaw) + dy * math.sin(ego_yaw)
        lateral = -dx * math.sin(ego_yaw) + dy * math.cos(ego_yaw)
        return {
            "forward_m": forward,
            "lateral_m": lateral,
            "path_distance_m": abs(lateral),
        }

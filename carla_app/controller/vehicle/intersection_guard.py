"""Köşe radarlarıyla kavşak çatışma bölgesini izler."""

import math

from carla_app.controller.vehicle.tracking import Tracker


def should_hold_for_intersection(light, intersection, ego_speed):
    """Yeşilde düşük hızlı aracın kavşakta beklemesi gerekip gerekmediğini söyler."""
    if light is None or str(light.get("color", "unknown")) != "green":
        return False
    if max(0.0, float(ego_speed)) > 1.0:
        return False
    intersection = intersection or {}
    if not intersection.get("active", False):
        return False
    return not bool(intersection.get("clear", False))


class IntersectionGuard:
    """Yeşilde kalkmadan önce çapraz trafik yolunu kontrol eder."""

    def __init__(self, layout, dt, parameters):
        self.layout = layout
        self.dt = max(0.01, float(dt))
        self.parameters = parameters
        self.tracker = Tracker(
            gate_distance_m=7.0,
            max_misses=max(3, int(round(0.35 / self.dt))),
            mahalanobis_gate=9.21,
        )
        self.radar_specs = []
        for radar in layout.radars:
            if radar.name != "radar_front_long":
                self.radar_specs.append(radar)
        self.last_measurement_key = None
        self.clear_hits = 0

    def update(self, current_frame_id, state, sensor_snapshot, road_context=None):
        current_frame_id = int(current_frame_id)
        road_context = road_context or {}
        active = self.should_monitor(state, road_context)
        if not active:
            self.clear_hits = 0
            return {
                "active": False,
                "status": "INACTIVE",
                "clear": True,
                "conflict": None,
                "fresh_radar_count": 0,
            }

        measurements, fresh_radar_count, newest_frame = self.collect_measurements(
            current_frame_id,
            state,
            sensor_snapshot or {},
        )
        measurement_key = (newest_frame, len(measurements))
        new_measurement_evidence = measurement_key != self.last_measurement_key
        if new_measurement_evidence:
            tracks = self.tracker.step(self.dt, measurements)
            self.last_measurement_key = measurement_key
        else:
            tracks = self.tracker.step(self.dt, [])

        conflict = self.find_conflict(tracks, state)
        if fresh_radar_count == 0:
            status = "UNKNOWN"
            clear = False
            self.clear_hits = 0
        elif conflict is not None:
            status = "BLOCKED"
            clear = False
            self.clear_hits = 0
        else:
            # Aynı radar karesi yeni bir güvenlik kanıtı sayılmaz. Yalnız yeni
            # köşe radar paketi geldiğinde temiz-kavşak sayacı ilerler.
            if new_measurement_evidence:
                self.clear_hits += 1
            required_hits = int(
                getattr(
                    self.parameters,
                    "intersection_clear_confirmation_frames",
                    3,
                )
            )
            clear = self.clear_hits >= max(1, required_hits)
            status = "CLEAR" if clear else "VERIFYING_CLEAR"

        return {
            "active": True,
            "status": status,
            "clear": bool(clear),
            "conflict": conflict,
            "fresh_radar_count": int(fresh_radar_count),
            "measurement_count": len(measurements),
            "track_count": len(tracks),
            "clear_hits": int(self.clear_hits),
            "frame_id": current_frame_id,
        }

    def should_monitor(self, state, road_context):
        if bool(state.get("is_junction", False)):
            return True
        light = road_context.get("lead_traffic_light")
        if light is None:
            return False
        try:
            distance = float(light.get("estimated_distance_m", math.inf))
        except (TypeError, ValueError):
            return False
        maximum_distance = float(
            getattr(self.parameters, "intersection_monitor_distance_m", 35.0)
        )
        return distance <= maximum_distance

    def collect_measurements(self, current_frame_id, state, sensor_snapshot):
        measurements = []
        fresh_radar_count = 0
        newest_frame = None
        maximum_age_frames = max(
            1,
            int(
                round(
                    float(
                        getattr(
                            self.parameters,
                            "intersection_radar_maximum_age_s",
                            0.20,
                        )
                    )
                    / self.dt
                )
            ),
        )

        ego_location = state["location"]
        ego_yaw = math.radians(float(state.get("yaw", 0.0)))

        for spec in self.radar_specs:
            entry = sensor_snapshot.get(spec.name)
            if entry is None:
                continue
            radar_frame_id = entry.get("frame_id")
            if radar_frame_id is None:
                continue
            age_frames = current_frame_id - int(radar_frame_id)
            if age_frames < 0 or age_frames > maximum_age_frames:
                continue
            fresh_radar_count += 1
            if newest_frame is None or int(radar_frame_id) > newest_frame:
                newest_frame = int(radar_frame_id)

            sensor_x = float(spec.transform.location.x)
            sensor_y = float(spec.transform.location.y)
            sensor_yaw = math.radians(float(spec.transform.rotation.yaw))
            for cluster in self.cluster_points(entry.get("data") or []):
                measurement = self.cluster_to_measurement(
                    cluster,
                    sensor_x,
                    sensor_y,
                    sensor_yaw,
                    ego_location,
                    ego_yaw,
                    age_frames,
                    int(radar_frame_id),
                )
                if measurement is not None:
                    measurements.append(measurement)

        return measurements, fresh_radar_count, newest_frame

    def cluster_points(self, radar_points):
        prepared = []
        for point in radar_points:
            try:
                depth = float(point["depth_m"])
                azimuth = math.radians(float(point.get("azimuth_deg", 0.0)))
                altitude = float(point.get("altitude_deg", 0.0))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(depth) or not 0.5 < depth <= 45.0:
                continue
            if abs(altitude) > 8.0:
                continue
            prepared.append(
                {
                    "x": depth * math.cos(azimuth),
                    "y": depth * math.sin(azimuth),
                    "depth": depth,
                }
            )

        clusters = []
        used = set()
        for index, point in enumerate(prepared):
            if index in used:
                continue
            cluster = [point]
            used.add(index)
            for other_index in range(index + 1, len(prepared)):
                if other_index in used:
                    continue
                other = prepared[other_index]
                gate = 1.2 + 0.02 * min(point["depth"], other["depth"])
                if math.hypot(point["x"] - other["x"], point["y"] - other["y"]) <= gate:
                    cluster.append(other)
                    used.add(other_index)
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters

    def cluster_to_measurement(
        self,
        cluster,
        sensor_x,
        sensor_y,
        sensor_yaw,
        ego_location,
        ego_yaw,
        age_frames,
        radar_frame_id,
    ):
        local_x = sum(point["x"] for point in cluster) / len(cluster)
        local_y = sum(point["y"] for point in cluster) / len(cluster)
        ego_x = sensor_x + local_x * math.cos(sensor_yaw) - local_y * math.sin(
            sensor_yaw
        )
        ego_y = sensor_y + local_x * math.sin(sensor_yaw) + local_y * math.cos(
            sensor_yaw
        )

        # Ego aracın hemen kendi gövdesine düşen yansımaları ele.
        if -2.5 < ego_x < 2.5 and -1.8 < ego_y < 1.8:
            return None

        world_x = float(ego_location.x) + ego_x * math.cos(ego_yaw) - ego_y * math.sin(
            ego_yaw
        )
        world_y = float(ego_location.y) + ego_x * math.sin(ego_yaw) + ego_y * math.cos(
            ego_yaw
        )
        distance = math.hypot(ego_x, ego_y)
        position_std = 0.45 + 0.02 * distance + 0.30 * age_frames
        return {
            "x": world_x,
            "y": world_y,
            "class_name": "cross_traffic_candidate",
            "position_std_m": position_std,
            "measurement_frame_id": radar_frame_id,
            "range_m": distance,
        }

    def find_conflict(self, tracks, state):
        ego_location = state["location"]
        ego_yaw = math.radians(float(state.get("yaw", 0.0)))
        lane_width = max(2.5, float(state.get("lane_width", 3.5)))
        forward_min = float(
            getattr(self.parameters, "intersection_zone_start_m", 1.5)
        )
        forward_max = float(
            getattr(self.parameters, "intersection_zone_end_m", 18.0)
        )
        lateral_limit = max(
            lane_width,
            float(getattr(self.parameters, "intersection_zone_half_width_m", 7.0)),
        )
        horizon_s = float(
            getattr(self.parameters, "intersection_prediction_horizon_s", 4.0)
        )
        step_s = 0.25

        best = None
        for track in tracks:
            if track.hit_count < 2 or track.miss_count > 2:
                continue
            speed = math.hypot(track.vx, track.vy)
            if speed < 0.8:
                continue

            for step in range(int(horizon_s / step_s) + 1):
                time_s = step * step_s
                predicted_x = track.x + track.vx * time_s
                predicted_y = track.y + track.vy * time_s
                delta_x = predicted_x - float(ego_location.x)
                delta_y = predicted_y - float(ego_location.y)
                forward = delta_x * math.cos(ego_yaw) + delta_y * math.sin(ego_yaw)
                lateral = -delta_x * math.sin(ego_yaw) + delta_y * math.cos(ego_yaw)
                uncertainty = min(3.0, 2.0 * track.position_std_m)
                inside = (
                    forward_min - uncertainty <= forward <= forward_max + uncertainty
                    and abs(lateral) <= lateral_limit + uncertainty
                )
                if not inside:
                    continue
                candidate = {
                    "track_id": track.id,
                    "time_to_conflict_s": float(time_s),
                    "forward_m": float(forward),
                    "lateral_m": float(lateral),
                    "speed_mps": float(speed),
                    "position_std_m": float(track.position_std_m),
                }
                if best is None or candidate["time_to_conflict_s"] < best[
                    "time_to_conflict_s"
                ]:
                    best = candidate
                break
        return best

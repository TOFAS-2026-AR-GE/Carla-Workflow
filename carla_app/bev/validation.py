"""BEV kanıtını kontrol algısıyla güvenli biçimde çapraz doğrular."""

import math

import numpy as np


class BevValidationLayer:
    """BEV'yi karar kaynağı değil, güven dereceli ikinci görüş olarak kullanır.

    Negatif BEV sonucu mevcut bir tehlikeyi hiçbir zaman ``REJECTED`` yapmaz.
    Eksik veya eski kanıt ``UNKNOWN``/``UNAVAILABLE`` olarak kalır.
    """

    def __init__(
        self,
        maximum_scene_age_frames=10,
        maximum_sensor_age_frames=10,
        maximum_control_scene_age_frames=4,
        maximum_control_sensor_age_frames=4,
    ):
        self.maximum_scene_age_frames = max(0, int(maximum_scene_age_frames))
        self.maximum_sensor_age_frames = max(0, int(maximum_sensor_age_frames))
        self.maximum_control_scene_age_frames = max(
            0,
            int(maximum_control_scene_age_frames),
        )
        self.maximum_control_sensor_age_frames = max(
            0,
            int(maximum_control_sensor_age_frames),
        )

    def evaluate(
        self,
        scene,
        current_frame_id,
        lead_vehicle=None,
        emergency_obstacle=None,
    ):
        if not scene or scene.get("frame_id") is None:
            return self.unavailable_result("scene_missing", lead_vehicle)

        scene_age = int(current_frame_id) - int(scene["frame_id"])
        if scene_age < 0 or scene_age > self.maximum_scene_age_frames:
            return self.unavailable_result(
                "scene_stale",
                lead_vehicle,
                scene_age_frames=scene_age,
            )

        health = self.sensor_health(
            scene.get("sensor_status", {}),
            additional_age_frames=scene_age,
        )
        safe_to_use = health["available_modalities"] >= 2
        lead = self.validate_target(
            lead_vehicle,
            scene,
            safe_to_use,
            target_name="lead",
            current_frame_id=current_frame_id,
        )
        emergency = self.validate_target(
            emergency_obstacle,
            scene,
            safe_to_use,
            target_name="emergency",
            current_frame_id=current_frame_id,
        )
        unexpected = self.unexpected_forward_obstacle(scene)

        statuses = {lead["status"], emergency["status"]}
        if "CONFIRMED" in statuses:
            status = "CONFIRMED"
        elif "CONFLICT" in statuses or (
            lead_vehicle is None and unexpected["detected"]
        ):
            status = "CONFLICT"
        elif safe_to_use:
            status = "UNKNOWN"
        else:
            status = "UNAVAILABLE"

        return {
            "status": status,
            "safe_to_use": bool(safe_to_use),
            "scene_age_frames": scene_age,
            "health": health,
            "lead": lead,
            "emergency": emergency,
            "unexpected_forward_obstacle": unexpected,
            "policy": "confirm_or_warn_never_clear_hazard",
        }

    def contribute(
        self,
        scene,
        current_frame_id,
        state,
        lead_vehicle=None,
    ):
        """Eksik lead'i yalnız kontrol kalitesindeki BEV track'iyle tamamlar."""
        if lead_vehicle is not None:
            return {
                "applied": False,
                "reason": "primary_lead_preserved",
                "lead_vehicle": lead_vehicle,
                "track_id": None,
            }

        validation = self.evaluate(scene, current_frame_id)
        if not validation["safe_to_use"]:
            return {
                "applied": False,
                "reason": "bev_unavailable",
                "lead_vehicle": None,
                "track_id": None,
            }
        if (
            validation["scene_age_frames"]
            > self.maximum_control_scene_age_frames
        ):
            return {
                "applied": False,
                "reason": "bev_scene_not_control_fresh",
                "lead_vehicle": None,
                "track_id": None,
            }

        candidates = []
        for track in scene.get("tracks", []):
            candidate = self.control_candidate(
                track,
                scene,
                current_frame_id,
                state,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return {
                "applied": False,
                "reason": "no_control_grade_bev_track",
                "lead_vehicle": None,
                "track_id": None,
            }

        candidate = min(candidates, key=lambda item: item["distance_m"])
        return {
            "applied": True,
            "reason": "confirmed_bev_lead_recovery",
            "lead_vehicle": candidate,
            "track_id": candidate["track_id"],
        }

    def control_candidate(self, track, scene, current_frame_id, state):
        if not track.get("confirmed", False):
            return None
        if int(track.get("misses", 0)) > 2:
            return None
        if float(track.get("confidence", 0.0)) < 0.75:
            return None

        source_frames = track.get("source_frames", {})
        fresh_sources = []
        for source in set(track.get("sources", [])):
            frame_id = source_frames.get(source)
            if frame_id is None:
                continue
            age = int(current_frame_id) - int(frame_id)
            if 0 <= age <= self.maximum_control_sensor_age_frames:
                fresh_sources.append(source)
        if len(fresh_sources) < 2:
            return None
        if len(set(track.get("sensor_names", []))) < 2:
            return None

        forward = float(track.get("x_m", -1.0))
        lateral = float(track.get("y_m", math.inf))
        if not 1.0 < forward <= 60.0:
            return None
        if not self.inside_route_corridor(track, scene):
            return None

        occupancy = self.occupancy_support(
            scene.get("occupancy"),
            forward,
            lateral,
        )
        if not occupancy["supported"]:
            return None

        geometry = scene.get("vehicle_geometry", {})
        ego_half_length = max(0.5, float(geometry.get("half_length_m", 2.35)))
        object_half_length = max(0.5, float(track.get("length_m", 4.5)) / 2.0)
        distance = max(0.10, forward - ego_half_length - object_half_length)
        ego_speed = max(0.0, float((state or {}).get("speed_mps", 0.0)))
        lead_speed = max(0.0, float(track.get("velocity_x_mps", 0.0)))
        relative_speed = min(20.0, max(-20.0, lead_speed - ego_speed))
        return {
            "track_id": int(track.get("track_id", -1)),
            "class_name": track.get("class_name", "obstacle"),
            "distance_m": distance,
            "lateral_m": lateral,
            "lead_speed_mps": lead_speed,
            "relative_speed_mps": relative_speed,
            "source": "bev_multisensor_recovery",
            "measurement_frame_id": track.get("last_measurement_frame_id"),
            "bearing_deg": math.degrees(math.atan2(lateral, forward)),
            "confidence": float(track.get("confidence", 0.0)),
            "validation_sources": sorted(fresh_sources),
            "occupancy_probability": occupancy["probability"],
            "bev_scene_frame_id": int(scene["frame_id"]),
        }

    def inside_route_corridor(self, track, scene):
        forward = float(track["x_m"])
        lateral = float(track["y_m"])
        lane_width = max(2.5, float(scene.get("lane_width_m", 3.5)))
        object_width = max(0.5, float(track.get("width_m", 1.9)))
        gate = 0.5 * lane_width + 0.25 * object_width
        route_points = scene.get("route_points", [])
        if len(route_points) < 2:
            return abs(lateral) <= min(gate, 1.6)
        return self.distance_to_polyline(forward, lateral, route_points) <= gate

    def distance_to_polyline(self, x_m, y_m, points):
        best = math.inf
        target = np.array([float(x_m), float(y_m)], dtype=np.float64)
        for first, second in zip(points, points[1:]):
            start = np.asarray(first[:2], dtype=np.float64)
            end = np.asarray(second[:2], dtype=np.float64)
            segment = end - start
            length_squared = float(segment @ segment)
            if length_squared <= 1e-8:
                continue
            ratio = float((target - start) @ segment / length_squared)
            ratio = min(1.0, max(0.0, ratio))
            projection = start + ratio * segment
            best = min(best, float(np.linalg.norm(target - projection)))
        return best

    def sensor_health(self, sensor_status, additional_age_frames=0):
        fresh_by_kind = {"camera": 0, "radar": 0, "lidar": 0}
        stale = []
        missing = []
        for sensor_name, item in sensor_status.items():
            kind = str(item.get("kind", ""))
            if kind not in fresh_by_kind:
                continue
            age = item.get("age_frames")
            if age is None:
                missing.append(sensor_name)
                continue
            effective_age = int(age) + max(0, int(additional_age_frames))
            if 0 <= effective_age <= self.maximum_sensor_age_frames:
                fresh_by_kind[kind] += 1
            else:
                stale.append(sensor_name)

        modalities = sum(count > 0 for count in fresh_by_kind.values())
        if modalities >= 3:
            status = "HEALTHY"
        elif modalities >= 2:
            status = "DEGRADED"
        else:
            status = "UNAVAILABLE"
        return {
            "status": status,
            "fresh_by_kind": fresh_by_kind,
            "available_modalities": modalities,
            "stale_sensors": sorted(stale),
            "missing_sensors": sorted(missing),
        }

    def validate_target(
        self,
        target,
        scene,
        can_compare,
        target_name,
        current_frame_id,
    ):
        if target is None:
            return {"status": "NOT_REQUESTED", "target": target_name}

        try:
            forward = float(target["distance_m"])
            lateral = float(target.get("lateral_m", 0.0))
        except (KeyError, TypeError, ValueError):
            return {
                "status": "UNKNOWN",
                "target": target_name,
                "reason": "invalid_target",
            }

        match = self.best_track_match(scene.get("tracks", []), forward, lateral)
        occupancy = self.occupancy_support(
            scene.get("occupancy"),
            forward,
            lateral,
        )
        if match is not None:
            sources = sorted(set(match.get("sources", [])))
            source_frames = match.get("source_frames", {})
            if source_frames:
                fresh_sources = []
                for source in sources:
                    frame_id = source_frames.get(source)
                    if frame_id is None:
                        continue
                    age = int(current_frame_id) - int(frame_id)
                    if 0 <= age <= self.maximum_sensor_age_frames:
                        fresh_sources.append(source)
            else:
                fresh_sources = list(sources)
            independent_sensors = self.independent_sensor_names(target, match)
            independent = len(fresh_sources)
            result = {
                "status": "SUPPORTED",
                "target": target_name,
                "track_id": int(match.get("track_id", -1)),
                "sources": sources,
                "fresh_sources": fresh_sources,
                "independent_sensor_names": independent_sensors,
                "independent_modalities": independent,
                "position_error_m": float(match["position_error_m"]),
                "occupancy_probability": occupancy["probability"],
            }
            if (
                bool(match.get("confirmed", False))
                and int(match.get("misses", 0)) == 0
                and float(match.get("confidence", 0.0)) >= 0.60
                and independent >= 2
                and self.has_independent_support(target, independent_sensors)
            ):
                result["status"] = "CONFIRMED"
            return result

        if occupancy["supported"]:
            return {
                "status": "SUPPORTED",
                "target": target_name,
                "reason": "occupancy_only",
                "occupancy_probability": occupancy["probability"],
            }

        return {
            "status": "CONFLICT" if can_compare else "UNKNOWN",
            "target": target_name,
            "reason": "no_independent_bev_support",
            "occupancy_probability": occupancy["probability"],
        }

    def independent_sensor_names(self, target, track):
        sensor_names = set(track.get("sensor_names", []))
        source = str(target.get("source", ""))
        control_sensors = set()
        if source.startswith("radar"):
            control_sensors.add("radar_front_long")
        elif source:
            control_sensors.update(
                {"camera_front_wide", "radar_front_long", "lidar_roof"}
            )
        return sorted(sensor_names - control_sensors)

    def has_independent_support(self, target, independent_sensor_names):
        # Kaynak belirtilmeyen sentetik/harici hedeflerde modalite çeşitliliği
        # yeterlidir. Uygulamanın kendi hedeflerinde en az bir ayrı fiziksel
        # sensör istenir.
        if not target.get("source"):
            return True
        return bool(independent_sensor_names)

    def best_track_match(self, tracks, forward, lateral):
        best = None
        best_score = None
        for track in tracks:
            if not track.get("confirmed", False):
                continue
            uncertainty = max(0.20, float(track.get("uncertainty_m", 1.0)))
            delta_forward = float(track.get("x_m", math.inf)) - forward
            delta_lateral = float(track.get("y_m", math.inf)) - lateral
            forward_gate = max(2.0, 2.5 * uncertainty)
            lateral_gate = max(1.0, 1.5 * uncertainty + 0.5)
            if abs(delta_forward) > forward_gate:
                continue
            if abs(delta_lateral) > lateral_gate:
                continue
            score = (delta_forward / forward_gate) ** 2
            score += (delta_lateral / lateral_gate) ** 2
            if best_score is None or score < best_score:
                best_score = score
                best = dict(track)
                best["position_error_m"] = math.hypot(
                    delta_forward,
                    delta_lateral,
                )
        return best

    def occupancy_support(self, occupancy, forward, lateral):
        if not occupancy or occupancy.get("probability") is None:
            return {"supported": False, "probability": None}
        probability = np.asarray(
            occupancy.get("sensor_probability", occupancy["probability"])
        )
        pixel = self.metric_to_cell(occupancy, forward, lateral)
        if pixel is None:
            return {"supported": False, "probability": None}
        pixel_x, pixel_y = pixel
        radius = max(1, int(round(1.0 / float(occupancy["cell_size_m"]))))
        y1 = max(0, pixel_y - radius)
        y2 = min(probability.shape[0], pixel_y + radius + 1)
        x1 = max(0, pixel_x - radius)
        x2 = min(probability.shape[1], pixel_x + radius + 1)
        local_probability = float(np.max(probability[y1:y2, x1:x2]))
        return {
            "supported": local_probability >= 0.62,
            "probability": local_probability,
        }

    def unexpected_forward_obstacle(self, scene):
        occupancy = scene.get("occupancy")
        if not occupancy or occupancy.get("occupied") is None:
            return {"detected": False, "nearest_m": None}
        sensor_probability = np.asarray(
            occupancy.get("sensor_probability", occupancy["probability"])
        )
        occupied = sensor_probability >= 0.62
        nearest = None
        for forward in np.arange(2.0, 30.1, 0.5):
            occupied_cells = 0
            for lateral in np.arange(-1.25, 1.26, 0.5):
                pixel = self.metric_to_cell(occupancy, forward, lateral)
                if pixel is None:
                    continue
                if occupied[pixel[1], pixel[0]]:
                    occupied_cells += 1
            if occupied_cells >= 3:
                nearest = float(forward)
                break
        return {"detected": nearest is not None, "nearest_m": nearest}

    def metric_to_cell(self, occupancy, forward, lateral):
        probability = np.asarray(occupancy["probability"])
        height, width = probability.shape[:2]
        forward_range = float(occupancy["forward_range_m"])
        rear_range = float(occupancy["rear_range_m"])
        side_range = float(occupancy["side_range_m"])
        if not -rear_range <= forward <= forward_range:
            return None
        if not -side_range <= lateral <= side_range:
            return None
        pixel_x = int(round((lateral + side_range) * (width - 1) / (2 * side_range)))
        pixel_y = int(
            round(
                (forward_range - forward)
                * (height - 1)
                / (forward_range + rear_range)
            )
        )
        return pixel_x, pixel_y

    def unavailable_result(self, reason, lead_vehicle, scene_age_frames=None):
        lead_status = "NOT_REQUESTED" if lead_vehicle is None else "UNAVAILABLE"
        return {
            "status": "UNAVAILABLE",
            "safe_to_use": False,
            "scene_age_frames": scene_age_frames,
            "health": {
                "status": "UNAVAILABLE",
                "fresh_by_kind": {"camera": 0, "radar": 0, "lidar": 0},
                "available_modalities": 0,
                "stale_sensors": [],
                "missing_sensors": [],
            },
            "lead": {"status": lead_status, "reason": reason},
            "emergency": {"status": "UNAVAILABLE", "reason": reason},
            "unexpected_forward_obstacle": {
                "detected": False,
                "nearest_m": None,
            },
            "policy": "confirm_or_warn_never_clear_hazard",
        }

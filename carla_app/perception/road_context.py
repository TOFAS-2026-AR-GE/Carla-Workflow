"""Kamera algılamalarını izler, mesafe ekler ve yol bağlamına dönüştürür."""

import math
import re

import numpy as np

from carla_app.bev.calibration import CalibrationSet
from carla_app.config import DrivingParameters


class RoadContextTracker:
    """Ham model kutularını zamansal olarak doğrulanmış yol bilgisine çevirir."""

    def __init__(self, layout=None, parameters=None):
        self.parameters = parameters or DrivingParameters()
        self.layout = layout
        self.calibrations = CalibrationSet(layout) if layout is not None else None
        self.primary_camera_name = (
            layout.primary_camera_name if layout is not None else "camera_front_wide"
        )
        self.tracks = {}
        self.next_track_id = 1
        self.last_perception_frame_id = None
        self.first_update_frame_id = None
        self.current_speed_limit_kmh = None
        self.latest_detections = []
        self.sensor_fault_start_frame_id = None

    def update(
        self,
        current_frame_id,
        perception_result,
        lidar_entry=None,
        state=None,
    ):
        """Yeni algılama karesini işler, eski kareyi tekrar doğrulama saymaz."""
        current_frame_id = int(current_frame_id)
        if self.first_update_frame_id is None:
            self.first_update_frame_id = current_frame_id

        perception_frame_id = None
        if perception_result is not None:
            perception_frame_id = perception_result.get("frame_id")
        is_new_frame = (
            perception_frame_id is not None
            and perception_frame_id != self.last_perception_frame_id
        )

        if is_new_frame:
            raw_objects = self.collect_raw_objects(perception_result)
            standardized = []
            for raw in raw_objects:
                detection = self.standardize_detection(
                    raw,
                    int(perception_frame_id),
                    perception_result.get("image"),
                    lidar_entry,
                )
                if detection is not None:
                    standardized.append(detection)
            self.latest_detections = self.update_tracks(
                standardized,
                int(perception_frame_id),
            )
            self.last_perception_frame_id = int(perception_frame_id)
        else:
            self.latest_detections = self.active_track_outputs(current_frame_id)

        route_center_x, image_width = self.route_center_pixel(
            state,
            perception_result,
        )
        lead_light = self.select_lead_traffic_light(
            self.latest_detections,
            route_center_x,
            image_width,
        )
        self.update_speed_limit(
            self.latest_detections,
            route_center_x,
            image_width,
        )
        pedestrian, pedestrian_risk = self.evaluate_pedestrian_risk(
            self.latest_detections,
            route_center_x,
            image_width,
        )

        age_frames = self.perception_age(current_frame_id)
        fresh = age_frames <= self.parameters.sensor_timeout_frames
        errors = perception_result.get("errors", {}) if perception_result else {}
        detector_failed = bool(errors.get("vehicle"))
        if detector_failed and self.sensor_fault_start_frame_id is None:
            self.sensor_fault_start_frame_id = current_frame_id
        elif not detector_failed and perception_result is not None:
            self.sensor_fault_start_frame_id = None
        fault_age_frames = 0
        if self.sensor_fault_start_frame_id is not None:
            fault_age_frames = current_frame_id - self.sensor_fault_start_frame_id
        sensor_fault = detector_failed or not fresh
        lidar_status = self.lidar_status(current_frame_id, lidar_entry)

        return {
            "perception_frame_id": self.last_perception_frame_id,
            "perception_age_frames": age_frames,
            "fresh": fresh,
            "sensor_fault": sensor_fault,
            "sensor_fault_age_frames": max(age_frames, fault_age_frames),
            "errors": dict(errors),
            "detections": list(self.latest_detections),
            "lead_traffic_light": lead_light,
            "speed_limit_kmh": self.current_speed_limit_kmh,
            "pedestrian": pedestrian,
            "pedestrian_risk": pedestrian_risk,
            "route_center_x": route_center_x,
            "lidar": lidar_status,
        }

    def collect_raw_objects(self, perception_result):
        objects = list(perception_result.get("objects", []))
        if not objects:
            objects.extend(perception_result.get("vehicles", []))
        known_boxes = set()
        for item in objects:
            known_boxes.add(tuple(item.get("bbox", ())))
        for sign in perception_result.get("signs", []):
            box = tuple(sign.get("bbox", ()))
            if box not in known_boxes:
                objects.append(sign)
                known_boxes.add(box)
        return objects

    def standardize_detection(
        self,
        raw,
        frame_id,
        image,
        lidar_entry,
    ):
        bbox = raw.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        image_height = int(image.shape[0]) if image is not None else int(max(y2 + 1, 1))
        image_width = int(image.shape[1]) if image is not None else int(max(x2 + 1, 1))
        class_name = str(raw.get("class_name", "unknown")).strip().lower()
        category = self.category_for_class(class_name)

        camera_distance = self.valid_distance(
            raw.get("estimated_distance_m", raw.get("range_m"))
        )
        if camera_distance is None:
            camera_distance = self.camera_distance_m(
                category,
                (x1, y1, x2, y2),
                image_width,
                image_height,
            )
        lidar_distance = self.lidar_distance_m(
            (x1, y1, x2, y2),
            image_width,
            image_height,
            frame_id,
            lidar_entry,
        )
        distance, source, conflict = self.combine_distances(
            camera_distance,
            lidar_distance,
        )
        if conflict:
            confidence *= 0.60
        valid = confidence >= self.confidence_threshold(category)

        return {
            "class_name": class_name,
            "category": category,
            "confidence": confidence,
            "raw_confidence": float(raw.get("confidence", 0.0)),
            "bbox": (x1, y1, x2, y2),
            "bbox_center": (0.5 * (x1 + x2), 0.5 * (y1 + y2)),
            "bbox_width": x2 - x1,
            "bbox_height": y2 - y1,
            "image_width": image_width,
            "image_height": image_height,
            "estimated_distance_m": distance,
            "camera_distance_m": camera_distance,
            "lidar_distance_m": lidar_distance,
            "distance_source": source,
            "sensor_conflict": conflict,
            "frame_id": int(frame_id),
            "valid": bool(valid),
        }

    def category_for_class(self, class_name):
        if class_name.startswith("traffic_light_"):
            return "traffic_light"
        if class_name.startswith("traffic_sign_") or class_name.startswith(
            "speed_limit_"
        ):
            return "speed_sign"
        if class_name in {"person", "pedestrian", "walker"}:
            return "pedestrian"
        if class_name in {"bike", "bicycle", "motobike", "motorbike"}:
            return "two_wheeler"
        if class_name in {"vehicle", "car", "truck", "bus", "van"}:
            return "vehicle"
        return "other"

    def confidence_threshold(self, category):
        if category == "traffic_light":
            return self.parameters.traffic_light_confidence
        if category == "speed_sign":
            return self.parameters.speed_sign_confidence
        if category == "pedestrian":
            return self.parameters.pedestrian_confidence
        return self.parameters.minimum_detection_confidence

    def valid_distance(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    def camera_distance_m(self, category, bbox, image_width, image_height):
        x1, _, x2, y2 = bbox
        if self.calibrations is not None and category != "traffic_light":
            camera = self.calibrations.get_camera(self.primary_camera_name)
            scale_x = camera.width / max(1.0, float(image_width))
            scale_y = camera.height / max(1.0, float(image_height))
            point = camera.pixel_to_ground_point(
                0.5 * (x1 + x2) * scale_x,
                y2 * scale_y,
            )
            if point is not None:
                return math.hypot(point[0], point[1])

        physical_heights = {
            "traffic_light": 0.90,
            "speed_sign": 0.60,
            "pedestrian": 1.70,
            "two_wheeler": 1.40,
            "vehicle": 1.50,
        }
        physical_height = physical_heights.get(category)
        pixel_height = max(1.0, bbox[3] - bbox[1])
        if physical_height is None:
            return None
        focal_length = float(image_width) / (
            2.0 * math.tan(math.radians(90.0) / 2.0)
        )
        if self.calibrations is not None:
            camera = self.calibrations.get_camera(self.primary_camera_name)
            focal_length = camera.K[1, 1] * image_height / camera.height
        distance = focal_length * physical_height / pixel_height
        return max(0.5, min(distance, 150.0))

    def lidar_distance_m(
        self,
        bbox,
        image_width,
        image_height,
        camera_frame_id,
        lidar_entry,
    ):
        if self.calibrations is None or lidar_entry is None:
            return None
        lidar_frame_id = lidar_entry.get("frame_id")
        if lidar_frame_id is None:
            return None
        if abs(int(camera_frame_id) - int(lidar_frame_id)) > (
            self.parameters.lidar_maximum_age_frames
        ):
            return None

        points = np.asarray(lidar_entry.get("data", []), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return None
        lidar = self.calibrations.get_sensor(self.layout.lidar.name)
        camera = self.calibrations.get_camera(self.primary_camera_name)
        ego_points = lidar.sensor_points_to_ego(points[:, :3])
        homogeneous = np.column_stack((ego_points, np.ones(len(ego_points))))
        camera_points = homogeneous @ camera.ego_to_sensor.T
        forward = camera_points[:, 0]
        valid = np.isfinite(camera_points).all(axis=1) & (forward > 0.20)
        if not np.any(valid):
            return None

        camera_points = camera_points[valid]
        ego_points = ego_points[valid]
        forward = camera_points[:, 0]
        pixel_x = camera.K[0, 0] * camera_points[:, 1] / forward + camera.K[0, 2]
        pixel_y = camera.K[1, 1] * (-camera_points[:, 2]) / forward + camera.K[1, 2]
        pixel_x *= image_width / camera.width
        pixel_y *= image_height / camera.height
        x1, y1, x2, y2 = bbox
        inside = (pixel_x >= x1) & (pixel_x <= x2) & (pixel_y >= y1) & (pixel_y <= y2)
        matched = ego_points[inside]
        if len(matched) < self.parameters.lidar_minimum_points:
            return None
        distances = np.hypot(matched[:, 0], matched[:, 1])
        return float(np.percentile(distances, 25.0))

    def combine_distances(self, camera_distance, lidar_distance):
        if lidar_distance is None:
            return camera_distance, "camera", False
        if camera_distance is None:
            return lidar_distance, "lidar", False
        difference = abs(camera_distance - lidar_distance)
        tolerance = max(
            self.parameters.camera_lidar_conflict_m,
            self.parameters.camera_lidar_conflict_ratio * camera_distance,
        )
        if difference > tolerance:
            return min(camera_distance, lidar_distance), "camera_lidar_conflict", True
        return 0.35 * camera_distance + 0.65 * lidar_distance, "camera_lidar", False

    def update_tracks(self, detections, frame_id):
        # Yeni kutuyu eşleştirmeden önce süresi dolmuş takipleri silmek,
        # eski bir kırmızı ışığın tek kareyle yeniden doğrulanmasını önler.
        self.remove_expired_tracks(frame_id)
        used_track_ids = set()
        for detection in detections:
            track = self.find_matching_track(detection, used_track_ids)
            if track is None:
                track = self.create_track(detection, frame_id)
            else:
                self.update_track(track, detection, frame_id)
            used_track_ids.add(track["track_id"])

        self.remove_expired_tracks(frame_id)
        return self.active_track_outputs(frame_id)

    def remove_expired_tracks(self, frame_id):
        """Takip türüne uygun kayıp süresi aşılmış kayıtları siler."""
        old_track_ids = []
        for track_id, track in self.tracks.items():
            track["missed_frames"] = max(0, frame_id - track["last_frame_id"])
            if track["missed_frames"] > self.track_lost_tolerance(track):
                old_track_ids.append(track_id)
        for track_id in old_track_ids:
            del self.tracks[track_id]

    def track_lost_tolerance(self, track):
        """Trafik ışıklarını normal kutulardan daha uzun süre hatırlar."""
        if track.get("family") == "traffic_light":
            return self.parameters.traffic_light_dropout_tolerance_frames
        return self.parameters.tracker_lost_tolerance_frames

    def find_matching_track(self, detection, used_track_ids):
        best_track = None
        best_score = None
        family = self.track_family(detection["category"])
        for track in self.tracks.values():
            if track["track_id"] in used_track_ids or track["family"] != family:
                continue
            overlap = self.iou(detection["bbox"], track["bbox"])
            distance = self.center_distance(detection["bbox_center"], track["center"])
            image_diagonal = math.hypot(
                detection["image_width"],
                detection["image_height"],
            )
            gate = max(60.0, 0.50 * image_diagonal)
            if overlap < 0.05 and distance > gate:
                continue
            score = distance - 200.0 * overlap
            if best_score is None or score < best_score:
                best_track = track
                best_score = score
        return best_track

    def create_track(self, detection, frame_id):
        track = {
            "track_id": self.next_track_id,
            "family": self.track_family(detection["category"]),
            "bbox": detection["bbox"],
            "center": detection["bbox_center"],
            "previous_center": detection["bbox_center"],
            "velocity_x_px_per_frame": 0.0,
            "smoothed_confidence": detection["confidence"],
            "candidate_class": detection["class_name"],
            "candidate_hits": 1 if detection["valid"] else 0,
            "stable_class": None,
            "hits": 1,
            "last_frame_id": frame_id,
            "missed_frames": 0,
            "detection": dict(detection),
        }
        if (
            detection["valid"]
            and self.required_confirmation_frames(detection["category"]) <= 1
        ):
            track["stable_class"] = detection["class_name"]
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        return track

    def update_track(self, track, detection, frame_id):
        frame_delta = max(1, frame_id - track["last_frame_id"])
        previous_center = track["center"]
        track["previous_center"] = previous_center
        track["velocity_x_px_per_frame"] = (
            detection["bbox_center"][0] - previous_center[0]
        ) / frame_delta
        track["bbox"] = detection["bbox"]
        track["center"] = detection["bbox_center"]
        track["smoothed_confidence"] = (
            0.65 * track["smoothed_confidence"] + 0.35 * detection["confidence"]
        )
        if detection["valid"]:
            if detection["class_name"] == track["candidate_class"]:
                track["candidate_hits"] += 1
            else:
                track["candidate_class"] = detection["class_name"]
                track["candidate_hits"] = 1
            required = self.required_confirmation_frames(detection["category"])
            if track["candidate_hits"] >= required:
                track["stable_class"] = track["candidate_class"]
        track["hits"] += 1
        track["last_frame_id"] = frame_id
        track["missed_frames"] = 0
        track["detection"] = dict(detection)

    def active_track_outputs(self, frame_id):
        outputs = []
        for track in self.tracks.values():
            missed = max(0, int(frame_id) - int(track["last_frame_id"]))
            tolerance = self.track_lost_tolerance(track)
            if missed > tolerance:
                continue
            item = dict(track["detection"])
            item["track_id"] = track["track_id"]
            item["smoothed_confidence"] = track["smoothed_confidence"]
            item["stable_class_name"] = track["stable_class"]
            item["confirmed"] = track["stable_class"] is not None
            item["missed_frames"] = missed
            item["velocity_x_px_per_frame"] = track["velocity_x_px_per_frame"]
            item["valid"] = bool(
                item["valid"]
                and missed <= tolerance
            )
            outputs.append(item)
        return outputs

    def required_confirmation_frames(self, category):
        if category == "traffic_light":
            return self.parameters.traffic_light_confirmation_frames
        if category == "speed_sign":
            return self.parameters.speed_sign_confirmation_frames
        return 1

    def track_family(self, category):
        return category

    def route_center_pixel(self, state, perception_result):
        image = perception_result.get("image") if perception_result else None
        width = int(image.shape[1]) if image is not None else 800
        center_x = 0.5 * width
        if (
            not state
            or not state.get("reference_path")
            or state.get("location") is None
        ):
            return center_x, width

        location = state["location"]
        yaw = math.radians(float(state.get("yaw", 0.0)))
        selected = None
        for point in state["reference_path"]:
            dx = float(point.x) - float(location.x)
            dy = float(point.y) - float(location.y)
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            right = -dx * math.sin(yaw) + dy * math.cos(yaw)
            if forward >= 20.0:
                selected = (forward, right)
                break
        if selected is None:
            return center_x, width
        bearing = math.atan2(selected[1], selected[0])
        focal = width / (2.0 * math.tan(math.radians(90.0) / 2.0))
        center_x += focal * math.tan(bearing)
        return max(0.15 * width, min(center_x, 0.85 * width)), width

    def select_lead_traffic_light(self, detections, route_center_x, image_width):
        selected = None
        selected_score = None
        for detection in detections:
            if detection["category"] != "traffic_light":
                continue
            if not detection["valid"] or not detection["confirmed"]:
                continue
            stable_class = detection.get("stable_class_name")
            if not stable_class or not stable_class.startswith("traffic_light_"):
                continue
            alignment = abs(detection["bbox_center"][0] - route_center_x) / image_width
            if alignment > 0.28:
                continue
            distance = detection.get("estimated_distance_m")
            if (
                distance is not None
                and float(distance)
                > self.parameters.traffic_light_maximum_distance_m
            ):
                continue
            distance_score = float(distance) if distance is not None else 120.0
            area_ratio = (
                detection["bbox_width"] * detection["bbox_height"]
                / max(1.0, detection["image_width"] * detection["image_height"])
            )
            score = distance_score + 35.0 * alignment - 20.0 * area_ratio
            if selected_score is None or score < selected_score:
                selected = dict(detection)
                selected["color"] = stable_class.rsplit("_", 1)[-1]
                selected["lane_alignment"] = 1.0 - min(1.0, alignment / 0.28)
                selected_score = score
        return selected

    def update_speed_limit(self, detections, route_center_x, image_width):
        candidates = []
        for detection in detections:
            if detection["category"] != "speed_sign":
                continue
            if not detection["valid"] or not detection["confirmed"]:
                continue
            horizontal_error = abs(
                detection["bbox_center"][0] - route_center_x
            )
            if horizontal_error > 0.45 * image_width:
                continue
            match = re.search(r"(\d{2,3})", detection["stable_class_name"] or "")
            if match is None:
                continue
            speed = int(match.group(1))
            if 10 <= speed <= 130:
                distance = detection.get("estimated_distance_m")
                candidates.append((distance if distance is not None else 999.0, speed))
        if candidates:
            candidates.sort()
            self.current_speed_limit_kmh = candidates[0][1]

    def evaluate_pedestrian_risk(self, detections, route_center_x, image_width):
        selected = None
        selected_level = "NONE"
        levels = {
            "NONE": 0,
            "MONITOR": 1,
            "SLOW": 2,
            "PREPARE_STOP": 3,
            "EMERGENCY": 4,
        }
        for detection in detections:
            if detection["category"] != "pedestrian" or not detection["valid"]:
                continue
            distance = detection.get("estimated_distance_m")
            if distance is None:
                continue
            offset = detection["bbox_center"][0] - route_center_x
            in_lane = abs(offset) <= 0.22 * image_width
            horizontal_velocity = detection.get("velocity_x_px_per_frame", 0.0)
            moving_toward_lane = offset * horizontal_velocity < -2.0
            level = "MONITOR"
            if in_lane and distance <= self.parameters.pedestrian_emergency_distance_m:
                level = "EMERGENCY"
            elif in_lane and distance <= self.parameters.pedestrian_stop_distance_m:
                level = "PREPARE_STOP"
            elif (
                (in_lane or moving_toward_lane)
                and distance <= self.parameters.pedestrian_slow_distance_m
            ):
                level = "SLOW"
            if levels[level] > levels[selected_level]:
                selected = dict(detection)
                selected["in_lane"] = in_lane
                selected["moving_toward_lane"] = moving_toward_lane
                selected_level = level
        return selected, selected_level

    def perception_age(self, current_frame_id):
        if self.last_perception_frame_id is None:
            return max(0, current_frame_id - self.first_update_frame_id)
        return max(0, current_frame_id - self.last_perception_frame_id)

    def lidar_status(self, current_frame_id, lidar_entry):
        if lidar_entry is None or lidar_entry.get("frame_id") is None:
            return {"available": False, "fresh": False, "age_frames": None}
        age = max(0, current_frame_id - int(lidar_entry["frame_id"]))
        return {
            "available": True,
            "fresh": age <= self.parameters.lidar_maximum_age_frames,
            "age_frames": age,
        }

    def iou(self, first, second):
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union > 0.0 else 0.0

    def center_distance(self, first, second):
        return math.hypot(first[0] - second[0], first[1] - second[1])

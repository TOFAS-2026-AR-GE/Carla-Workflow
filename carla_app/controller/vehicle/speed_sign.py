"""Kamera tespitlerinden kalıcı hız sınırı seçer."""

from __future__ import annotations

from collections import deque
import math
import re


class SpeedSignObserver:
    """Hız levhasını doğrular ve yeni referans hızı üretir.

    Hız sınırı, yeni bir hız levhası veya sınır sonu levhası görülene kadar
    geçerli kalır. Tek karelik yanlış sınıflandırmalar kontrolü değiştirmez.
    """

    def __init__(
        self,
        dt: float,
        image_width: int,
        image_height: int,
        maximum_speed_kmh: float,
    ) -> None:
        self.dt = max(1e-3, float(dt))
        self.image_width = max(1, int(image_width))
        self.image_height = max(1, int(image_height))
        self.maximum_speed_kmh = max(10.0, float(maximum_speed_kmh))

        self.maximum_perception_age_frames = max(1, int(round(0.60 / self.dt)))
        self.required_votes = 2
        self.recent_candidates = deque(maxlen=max(3, int(round(0.45 / self.dt))))

        self.active_speed_kmh: float | None = None
        self.last_processed_frame_id: int | None = None
        self.last_class_name = None
        self.last_confidence = 0.0
        self.last_bbox = None

    def update(self, current_frame_id: int, perception_result: dict | None) -> dict:
        changed = False
        detection = self.select_relevant_detection(perception_result)

        if detection is not None and perception_result is not None:
            detection_frame_id = int(perception_result.get("frame_id", current_frame_id))
            age = int(current_frame_id) - detection_frame_id
            fresh = 0 <= age <= self.maximum_perception_age_frames

            if fresh and detection_frame_id != self.last_processed_frame_id:
                self.last_processed_frame_id = detection_frame_id
                self.last_class_name = str(detection.get("class_name", "unknown"))
                self.last_confidence = float(detection.get("confidence", 0.0))
                self.last_bbox = tuple(detection.get("bbox", ()))

                action, speed_kmh = self.parse_class_name(self.last_class_name)
                if action == "clear":
                    self.recent_candidates.clear()
                    if self.active_speed_kmh is not None:
                        self.active_speed_kmh = None
                        changed = True
                elif action == "set" and speed_kmh is not None:
                    speed_kmh = min(float(speed_kmh), self.maximum_speed_kmh)
                    self.recent_candidates.append(speed_kmh)
                    votes = sum(
                        1
                        for candidate in self.recent_candidates
                        if abs(candidate - speed_kmh) < 0.1
                    )
                    if votes >= self.required_votes:
                        if (
                            self.active_speed_kmh is None
                            or abs(self.active_speed_kmh - speed_kmh) >= 0.1
                        ):
                            self.active_speed_kmh = speed_kmh
                            changed = True

        return {
            "active": self.active_speed_kmh is not None,
            "speed_limit_kmh": self.active_speed_kmh,
            "speed_limit_mps": (
                self.active_speed_kmh / 3.6
                if self.active_speed_kmh is not None
                else None
            ),
            "changed": changed,
            "class_name": self.last_class_name,
            "confidence": self.last_confidence,
            "bbox": self.last_bbox,
            "measurement_frame_id": self.last_processed_frame_id,
        }

    def select_relevant_detection(self, perception_result: dict | None) -> dict | None:
        if not perception_result:
            return None

        best = None
        best_score = -math.inf
        for detection in perception_result.get("signs", []) or []:
            action, speed_kmh = self.parse_class_name(detection.get("class_name"))
            if action == "ignore":
                continue

            try:
                x1, y1, x2, y2 = [float(value) for value in detection["bbox"]]
                confidence = float(detection.get("confidence", 0.0))
                detector_confidence = float(
                    detection.get("detection_confidence", confidence)
                )
            except (KeyError, TypeError, ValueError):
                continue

            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width < 6.0 or height < 8.0:
                continue

            center_x = 0.5 * (x1 + x2) / self.image_width
            center_y = 0.5 * (y1 + y2) / self.image_height
            if not 0.02 <= center_x <= 0.98:
                continue
            if not 0.08 <= center_y <= 0.92:
                continue

            area_ratio = (width * height) / (self.image_width * self.image_height)
            horizontal_relevance = 1.0 - min(1.0, abs(center_x - 0.5) / 0.5)
            score = (
                2.0 * confidence
                + detector_confidence
                + 18.0 * area_ratio
                + 0.15 * horizontal_relevance
            )
            if score > best_score:
                best = detection
                best_score = score

        return best

    @staticmethod
    def parse_class_name(class_name) -> tuple[str, float | None]:
        normalized = str(class_name or "").strip().lower().replace("-", "_")
        if not normalized or normalized == "unknown":
            return "ignore", None

        if normalized.startswith("end_speed_limit"):
            return "clear", None
        if "end_all_speed" in normalized or "end_all_limit" in normalized:
            return "clear", None

        if "speed" not in normalized and "limit" not in normalized:
            return "ignore", None

        match = re.search(r"(?<!\d)(10|20|30|40|50|60|70|80|90|100|110|120|130)(?!\d)", normalized)
        if match is None:
            return "ignore", None
        return "set", float(match.group(1))

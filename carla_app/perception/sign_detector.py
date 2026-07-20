"""İki aşamalı trafik levhası algılayıcısı."""

import json

import numpy as np
from ultralytics import YOLO

from carla_app.perception.device import (
    move_model_to_cpu,
    resolve_device,
)


class TrafficSignDetector:
    def __init__(
        self,
        detector_path,
        classifier_path,
        class_names_path,
        detector_confidence,
        detector_iou,
        classifier_confidence,
        detector_image_size,
        classifier_image_size,
        device,
        max_candidates,
    ):
        self.detector = YOLO(str(detector_path), task="detect")
        self.classifier = YOLO(
            str(classifier_path),
            task="classify",
        )
        self.class_names = self._load_names(class_names_path)
        self.detector_confidence = float(detector_confidence)
        self.detector_iou = float(detector_iou)
        self.classifier_confidence = float(
            classifier_confidence
        )
        self.detector_image_size = int(detector_image_size)
        self.classifier_image_size = int(
            classifier_image_size
        )
        self.device = resolve_device(device)
        self.max_candidates = max(1, int(max_candidates))

        print(
            f"[OK] Trafik levhasi modelleri: {self.device}, "
            f"max_candidates={self.max_candidates}"
        )

    def detect(self, rgb_image):
        bgr_image = np.ascontiguousarray(
            rgb_image[:, :, :3][:, :, ::-1]
        )
        height, width = bgr_image.shape[:2]

        result = self._predict(
            self.detector,
            source=bgr_image,
            conf=self.detector_confidence,
            iou=self.detector_iou,
            imgsz=self.detector_image_size,
            max_det=self.max_candidates * 2,
        )[0]

        if result.boxes is None:
            return []

        candidates = self._collect_candidates(
            result.boxes,
            bgr_image,
            width,
            height,
        )
        if not candidates:
            return []

        crops = [candidate["crop"] for candidate in candidates]
        classification_results = self._predict(
            self.classifier,
            source=crops,
            imgsz=self.classifier_image_size,
        )

        detections = []
        for candidate, classification in zip(
            candidates,
            classification_results,
        ):
            class_id, class_name, class_conf = (
                self._read_classification(classification)
            )

            detections.append(
                {
                    "type": "traffic_sign",
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": class_conf,
                    "detection_confidence": (
                        candidate["detection_confidence"]
                    ),
                    "bbox": candidate["bbox"],
                }
            )

        return detections

    def _collect_candidates(
        self,
        boxes,
        bgr_image,
        image_width,
        image_height,
    ):
        candidates = []

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            detection_confidence = float(
                box.conf[0].item()
            )

            crop_box = self._crop_box(
                x1,
                y1,
                x2,
                y2,
                image_width,
                image_height,
            )
            cx1, cy1, cx2, cy2 = crop_box
            crop = bgr_image[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                continue

            box_area = max(0.0, x2 - x1) * max(
                0.0,
                y2 - y1,
            )
            candidates.append(
                {
                    "crop": crop,
                    "detection_confidence": (
                        detection_confidence
                    ),
                    "box_area": box_area,
                    "bbox": (
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                item["detection_confidence"],
                item["box_area"],
            ),
            reverse=True,
        )
        return candidates[: self.max_candidates]

    def _read_classification(self, result):
        if result.probs is None:
            return -1, "unknown", 0.0

        class_id = int(result.probs.top1)
        confidence = float(
            result.probs.top1conf.item()
        )
        name = self.class_names.get(
            class_id,
            str(class_id),
        )

        if confidence < self.classifier_confidence:
            return -1, "unknown", confidence

        return class_id, name, confidence

    @staticmethod
    def _crop_box(
        x1,
        y1,
        x2,
        y2,
        width,
        height,
        padding=0.25,
    ):
        box_width = x2 - x1
        box_height = y2 - y1
        side = max(box_width, box_height)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        side *= 1 + 2 * padding

        return (
            max(0, int(center_x - side / 2)),
            max(0, int(center_y - side / 2)),
            min(width, int(center_x + side / 2)),
            min(height, int(center_y + side / 2)),
        )

    @staticmethod
    def _load_names(path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return {
            int(key): str(value)
            for key, value in data.items()
        }

    def _predict(self, model, **arguments):
        arguments.update(
            device=self.device,
            verbose=False,
        )

        try:
            return model.predict(**arguments)
        except (RuntimeError, ValueError) as error:
            if self.device == "cpu":
                raise

            print(
                f"[WARN] Sign GPU inference basarisiz ({error}); "
                "CPU ile yeniden deneniyor."
            )
            self.device = "cpu"
            move_model_to_cpu(self.detector)
            move_model_to_cpu(self.classifier)
            arguments["device"] = "cpu"
            return model.predict(**arguments)

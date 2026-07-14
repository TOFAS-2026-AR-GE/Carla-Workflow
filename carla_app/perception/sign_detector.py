import json

import numpy as np
from ultralytics import YOLO


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
    ):
        self.detector = YOLO(str(detector_path), task="detect")
        self.classifier = YOLO(str(classifier_path), task="classify")
        self.class_names = self._load_names(class_names_path)
        self.detector_confidence = detector_confidence
        self.detector_iou = detector_iou
        self.classifier_confidence = classifier_confidence
        self.detector_image_size = detector_image_size
        self.classifier_image_size = classifier_image_size
        self.device = device
        print(f"[OK] Trafik levhasi modelleri: {device}")

    def detect(self, rgb_image):
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
        height, width = bgr_image.shape[:2]
        result = self.detector.predict(
            source=bgr_image,
            conf=self.detector_confidence,
            iou=self.detector_iou,
            imgsz=self.detector_image_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            detect_conf = float(box.conf[0].item())
            crop_box = self._crop_box(x1, y1, x2, y2, width, height)
            cx1, cy1, cx2, cy2 = crop_box
            crop = bgr_image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            class_id, class_name, class_conf = self._classify(crop)
            detections.append(
                {
                    "type": "traffic_sign",
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": class_conf,
                    "detection_confidence": detect_conf,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                }
            )
        return detections

    def _classify(self, crop):
        result = self.classifier.predict(
            source=crop,
            imgsz=self.classifier_image_size,
            device=self.device,
            verbose=False,
        )[0]
        if result.probs is None:
            return -1, "unknown", 0.0

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf.item())
        name = self.class_names.get(class_id, str(class_id))
        if confidence < self.classifier_confidence:
            return -1, "unknown", confidence
        return class_id, name, confidence

    @staticmethod
    def _crop_box(x1, y1, x2, y2, width, height, padding=0.25):
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
        return {int(key): str(value) for key, value in data.items()}

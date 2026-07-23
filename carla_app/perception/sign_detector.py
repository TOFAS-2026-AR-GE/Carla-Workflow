"""Trafik levhasını önce bulur, sonra bulunan alanın sınıfını belirler."""

import json

import numpy as np
from ultralytics import YOLO

from carla_app.perception.device import (
    is_cuda_device,
    move_model_to_cpu,
    recover_cuda_memory,
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
        use_half=True,
    ):
        self.detector = YOLO(str(detector_path), task="detect")
        self.classifier = YOLO(str(classifier_path), task="classify")
        self.class_names = self._load_names(class_names_path)
        self.detector_confidence = detector_confidence
        self.detector_iou = detector_iou
        self.classifier_confidence = classifier_confidence
        self.detector_image_size = detector_image_size
        self.classifier_image_size = classifier_image_size
        self.device = resolve_device(device)
        self.use_half = bool(use_half and is_cuda_device(self.device))
        print(
            f"[OK] Trafik levhasi modelleri: {self.device}, "
            f"fp16={self.use_half}"
        )
        self._warmup_cuda()

    def detect(self, rgb_image):
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
        height, width = bgr_image.shape[:2]
        result = self._predict_detector(bgr_image)[0]

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
        result = self._predict_classifier(crop)[0]
        if result.probs is None:
            return -1, "unknown", 0.0

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf.item())
        name = self.class_names.get(class_id, str(class_id))
        if confidence < self.classifier_confidence:
            return -1, "unknown", confidence
        return class_id, name, confidence

    def _crop_box(self, x1, y1, x2, y2, width, height, padding=0.25):
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

    def _load_names(self, path):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        names = {}
        for key, value in data.items():
            names[int(key)] = str(value)
        return names

    def _predict_detector(self, image):
        """Levha bulma modelini çalıştırır; gerekirse bir kez CPU'ya geçer."""
        try:
            return self.detector.predict(
                source=image,
                conf=self.detector_confidence,
                iou=self.detector_iou,
                imgsz=self.detector_image_size,
                device=self.device,
                verbose=False,
                half=bool(getattr(self, "use_half", False)),
            )
        except (RuntimeError, ValueError) as error:
            if is_cuda_device(self.device):
                recover_cuda_memory()
                try:
                    return self.detector.predict(
                        source=image,
                        conf=self.detector_confidence,
                        iou=self.detector_iou,
                        imgsz=self.detector_image_size,
                        device=self.device,
                        verbose=False,
                        half=bool(getattr(self, "use_half", False)),
                    )
                except (RuntimeError, ValueError) as retry_error:
                    error = retry_error
            self._switch_to_cpu(error)
            return self.detector.predict(
                source=image,
                conf=self.detector_confidence,
                iou=self.detector_iou,
                imgsz=self.detector_image_size,
                device="cpu",
                verbose=False,
                half=False,
            )

    def _predict_classifier(self, image):
        """Levha sınıflandırma modelini çalıştırır; gerekirse CPU'ya geçer."""
        try:
            return self.classifier.predict(
                source=image,
                imgsz=self.classifier_image_size,
                device=self.device,
                verbose=False,
                half=bool(getattr(self, "use_half", False)),
            )
        except (RuntimeError, ValueError) as error:
            if is_cuda_device(self.device):
                recover_cuda_memory()
                try:
                    return self.classifier.predict(
                        source=image,
                        imgsz=self.classifier_image_size,
                        device=self.device,
                        verbose=False,
                        half=bool(getattr(self, "use_half", False)),
                    )
                except (RuntimeError, ValueError) as retry_error:
                    error = retry_error
            self._switch_to_cpu(error)
            return self.classifier.predict(
                source=image,
                imgsz=self.classifier_image_size,
                device="cpu",
                verbose=False,
                half=False,
            )

    def _switch_to_cpu(self, error):
        """İki levha modelini CPU'ya taşır."""
        if self.device == "cpu":
            raise error

        print(
            f"[WARN] Sign GPU inference basarisiz ({error}); "
            "CPU ile yeniden deneniyor."
        )
        self.device = "cpu"
        self.use_half = False
        move_model_to_cpu(self.detector)
        move_model_to_cpu(self.classifier)

    def _warmup_cuda(self):
        """ONNX/TensorRT sağlayıcılarını canlı kare gelmeden hazırlar."""
        if not is_cuda_device(self.device):
            return
        detector_image = np.zeros(
            (self.detector_image_size, self.detector_image_size, 3),
            dtype=np.uint8,
        )
        classifier_image = np.zeros(
            (self.classifier_image_size, self.classifier_image_size, 3),
            dtype=np.uint8,
        )
        self._predict_detector(detector_image)
        self._predict_classifier(classifier_image)

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    TARGET_CLASS_IDS = [0, 1, 9]

    def __init__(self, model_path, confidence, image_size, device):
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = confidence
        self.image_size = image_size
        self.device = device
        print(f"[OK] Vehicle modeli: {device}")

    def detect(self, rgb_image):
        bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
        result = self.model.predict(
            source=bgr_image,
            conf=self.confidence,
            classes=self.TARGET_CLASS_IDS,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is None:
            return detections

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].cpu().tolist()]
            names = result.names if isinstance(result.names, dict) else {}
            detections.append(
                {
                    "type": "vehicle",
                    "class_id": class_id,
                    "class_name": str(names.get(class_id, class_id)),
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return detections

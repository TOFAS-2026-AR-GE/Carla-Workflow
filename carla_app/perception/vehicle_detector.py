import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    YOLO arac dedektoru.

    Model sinif ID'lerini sabit kabul etmez.
    Modelin class name bilgisinden arac siniflarini otomatik bulur.
    """

    VEHICLE_KEYWORDS = (
        "vehicle",
        "car",
        "automobile",
        "sedan",
        "suv",
        "jeep",
        "van",
        "minivan",
        "pickup",
        "truck",
        "lorry",
        "bus",
        "coach",
        "motorcycle",
        "motorbike",
        "bicycle",
        "bike",
        "tram",
        "ambulance",
        "police",
        "taxi",
    )

    def __init__(
        self,
        model_path,
        confidence,
        image_size,
        device,
    ):
        self.model = YOLO(
            str(model_path),
            task="detect",
        )

        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = device

        self.model_names = self._normalise_names(
            self.model.names
        )

        self.vehicle_class_ids = (
            self._find_vehicle_class_ids(
                self.model_names
            )
        )

        self.inference_count = 0

        print(
            "[OK] Vehicle modeli yuklendi: "
            f"{model_path}"
        )

        print(
            "[OK] Vehicle device: "
            f"{device}"
        )

        print(
            "[INFO] YOLO model siniflari: "
            f"{self.model_names}"
        )

        if self.vehicle_class_ids:
            selected_names = {
                class_id: self.model_names[class_id]
                for class_id in self.vehicle_class_ids
            }

            print(
                "[INFO] Arac olarak kullanilan "
                f"siniflar: {selected_names}"
            )
        else:
            print(
                "[WARN] Model sinif isimlerinde bilinen "
                "arac sinifi bulunamadi. "
                "Modelin tum siniflari bbox icin kullanilacak."
            )

    def detect(self, rgb_image):
        if rgb_image is None:
            return []

        if not isinstance(rgb_image, np.ndarray):
            raise TypeError(
                "VehicleDetector numpy.ndarray bekliyor."
            )

        if (
            rgb_image.ndim != 3
            or rgb_image.shape[2] < 3
        ):
            raise ValueError(
                "Kamera goruntusu HxWx3 formatinda olmali. "
                f"Gelen shape: {rgb_image.shape}"
            )

        self.inference_count += 1

        # Sensor processor RGB donduruyor.
        # Ultralytics numpy kaynaklarinda BGR bekliyor.
        bgr_image = np.ascontiguousarray(
            rgb_image[:, :, :3][:, :, ::-1]
        )

        predict_arguments = {
            "source": bgr_image,
            "conf": self.confidence,
            "iou": 0.45,
            "imgsz": self.image_size,
            "device": self.device,
            "verbose": False,
            "max_det": 100,
        }

        # Sadece gercekten tespit edebildigimiz class ID'leri
        # filtrele. Sabit [0, 1, 9] kullanma.
        if self.vehicle_class_ids:
            predict_arguments["classes"] = (
                self.vehicle_class_ids
            )

        results = self.model.predict(
            **predict_arguments
        )

        if not results:
            self._print_debug(0)
            return []

        result = results[0]

        if result.boxes is None:
            self._print_debug(0)
            return []

        result_names = self._normalise_names(
            result.names
        )

        image_height, image_width = (
            bgr_image.shape[:2]
        )

        detections = []

        for box in result.boxes:
            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            coordinates = (
                box.xyxy[0]
                .detach()
                .cpu()
                .tolist()
            )

            x1, y1, x2, y2 = [
                int(round(value))
                for value in coordinates
            ]

            x1 = max(
                0,
                min(x1, image_width - 1),
            )
            x2 = max(
                0,
                min(x2, image_width - 1),
            )
            y1 = max(
                0,
                min(y1, image_height - 1),
            )
            y2 = max(
                0,
                min(y2, image_height - 1),
            )

            # Bozuk veya cok kucuk bbox'lari alma.
            if x2 <= x1 or y2 <= y1:
                continue

            if (
                x2 - x1 < 3
                or y2 - y1 < 3
            ):
                continue

            class_name = str(
                result_names.get(
                    class_id,
                    class_id,
                )
            )

            detections.append(
                {
                    "type": "vehicle",
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (
                        x1,
                        y1,
                        x2,
                        y2,
                    ),
                }
            )

        self._print_debug(
            len(detections)
        )

        return detections

    def _print_debug(
        self,
        detection_count,
    ):
        if (
            self.inference_count <= 3
            or self.inference_count % 20 == 0
        ):
            print(
                "[VISION] "
                f"inference={self.inference_count} "
                f"vehicle_bbox={detection_count} "
                f"conf={self.confidence:.2f}"
            )

    def _find_vehicle_class_ids(
        self,
        names,
    ):
        selected = []

        for class_id, class_name in names.items():
            normalised_name = (
                str(class_name)
                .strip()
                .lower()
                .replace("-", " ")
                .replace("_", " ")
            )

            if any(
                keyword in normalised_name
                for keyword
                in self.VEHICLE_KEYWORDS
            ):
                selected.append(
                    int(class_id)
                )

        return sorted(
            set(selected)
        )

    @staticmethod
    def _normalise_names(names):
        if isinstance(names, dict):
            return {
                int(class_id): str(class_name)
                for class_id, class_name
                in names.items()
            }

        if isinstance(
            names,
            (list, tuple),
        ):
            return {
                index: str(class_name)
                for index, class_name
                in enumerate(names)
            }

        return {}
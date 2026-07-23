"""Ön kamera görüntüsünde YOLO ile yol araçlarını bulur."""

import numpy as np
from ultralytics import YOLO

from carla_app.perception.device import (
    is_cuda_device,
    move_model_to_cpu,
    recover_cuda_memory,
    resolve_device,
)


class VehicleDetector:
    """Model sonucunu kontrol katmanının kullanacağı sade sözlüğe çevirir."""

    VEHICLE_NAMES = {
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
        "motobike",  # Repoda bulunan CARLA modelinin kullandığı sınıf adı.
        "bicycle",
        "bike",
        "tram",
        "ambulance",
        "police",
        "taxi",
    }

    RELEVANT_NAMES = {
        "bike",
        "motobike",
        "person",
        "traffic_light_green",
        "traffic_light_orange",
        "traffic_light_red",
        "traffic_sign_30",
        "traffic_sign_60",
        "traffic_sign_90",
        "vehicle",
    }

    def __init__(
        self,
        model_path,
        confidence,
        image_size,
        device,
        use_half=True,
    ):
        self.model = YOLO(str(model_path), task="detect")
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.device = resolve_device(device)
        self.use_half = bool(use_half and is_cuda_device(self.device))

        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("VEHICLE_CONFIDENCE 0 ile 1 arasinda olmali.")
        if self.image_size <= 0:
            raise ValueError("VEHICLE_IMAGE_SIZE sifirdan buyuk olmali.")

        self.model_names = self._normalize_names(self.model.names)
        self.vehicle_class_ids = self._find_vehicle_class_ids(self.model_names)
        self.relevant_class_ids = self._find_relevant_class_ids(self.model_names)

        if not self.vehicle_class_ids:
            raise ValueError(
                "Vehicle modelinde arac sinifi bulunamadi. "
                f"Model siniflari: {self.model_names}"
            )

        selected_names = {}
        for class_id in self.vehicle_class_ids:
            selected_names[class_id] = self.model_names[class_id]
        relevant_names = {}
        for class_id in self.relevant_class_ids:
            relevant_names[class_id] = self.model_names[class_id]
        print(
            f"[OK] Vehicle modeli: device={self.device}, "
            f"vehicle_classes={selected_names}, "
            f"traffic_classes={relevant_names}, conf={self.confidence:.2f}, "
            f"fp16={self.use_half}"
        )
        self._warmup_cuda()

    def detect(self, rgb_image):
        """Tek RGB kamera karesindeki araç kutularını döndürür."""
        bgr_image = self.prepare_image(rgb_image)
        results = self._predict(bgr_image)

        if not results or results[0].boxes is None:
            return []

        return self.parse_result(results[0], bgr_image)

    def detect_objects(self, rgb_image):
        """Modelin trafik için yararlı bütün sınıflarını tek çağrıda döndürür."""
        bgr_image = self.prepare_image(rgb_image)
        results = self._predict(bgr_image, self.relevant_class_ids)
        if not results or results[0].boxes is None:
            return []
        return self.parse_objects_result(results[0], bgr_image)

    def detect_many(self, images_by_name):
        """Birden fazla kamerayı tek YOLO çağrısında işler.

        Tek model örneği kullanıldığı için yedi ayrı modelin ekran kartı
        belleğini doldurması önlenir. Sonuçlar kamera adlarıyla döndürülür.
        """
        camera_names = []
        bgr_images = []
        for camera_name, rgb_image in images_by_name.items():
            camera_names.append(camera_name)
            bgr_images.append(self.prepare_image(rgb_image))

        detections_by_name = {}
        if not bgr_images:
            return detections_by_name

        source = bgr_images[0] if len(bgr_images) == 1 else bgr_images
        results = self._predict(source)

        for index, camera_name in enumerate(camera_names):
            detections = []
            if results and index < len(results):
                result = results[index]
                if result.boxes is not None:
                    detections = self.parse_result(result, bgr_images[index])
            detections_by_name[camera_name] = detections

        return detections_by_name

    def detect_many_objects(self, images_by_name):
        """Birden fazla kameradaki bütün ilgili sınıfları toplu işler."""
        return self._detect_many_selected(
            images_by_name,
            self.relevant_class_ids,
            False,
        )

    def prepare_image(self, rgb_image):
        """CARLA RGB görüntüsünü YOLO'nun beklediği BGR biçimine çevirir."""
        self._validate_image(rgb_image)
        return np.ascontiguousarray(rgb_image[:, :, :3][:, :, ::-1])

    def parse_result(self, result, bgr_image):
        """Tek bir YOLO sonucunu sade bbox sözlüklerine çevirir."""

        return self._parse_selected_result(
            result,
            bgr_image,
            self.vehicle_class_ids,
            vehicles_only=True,
        )

    def parse_objects_result(self, result, bgr_image):
        """Ham YOLO sonucunu trafik sınıflarını koruyan sade sözlüklere çevirir."""
        return self._parse_selected_result(
            result,
            bgr_image,
            self.relevant_class_ids,
            vehicles_only=False,
        )

    def _parse_selected_result(
        self,
        result,
        bgr_image,
        selected_class_ids,
        vehicles_only,
    ):

        result_names = self._normalize_names(result.names)
        image_height, image_width = bgr_image.shape[:2]
        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id not in selected_class_ids:
                continue

            coordinates = box.xyxy[0].detach().cpu().tolist()
            x1 = int(round(coordinates[0]))
            y1 = int(round(coordinates[1]))
            x2 = int(round(coordinates[2]))
            y2 = int(round(coordinates[3]))
            x1 = max(0, min(x1, image_width - 1))
            x2 = max(0, min(x2, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            y2 = max(0, min(y2, image_height - 1))

            if x2 - x1 < 3 or y2 - y1 < 3:
                continue

            class_name = result_names.get(class_id, self.model_names[class_id])
            if class_id in self.vehicle_class_ids:
                object_type = "vehicle"
            else:
                object_type = "road_object"
            if vehicles_only:
                object_type = "vehicle"
            detections.append(
                {
                    "type": object_type,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(box.conf[0].item()),
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections

    def _predict(self, bgr_image, class_ids=None):
        if class_ids is None:
            class_ids = self.vehicle_class_ids
        try:
            return self._predict_on_device(bgr_image, self.device, class_ids)
        except (RuntimeError, ValueError) as error:
            if self.device == "cpu":
                raise

            recover_cuda_memory()
            try:
                return self._predict_on_device(
                    bgr_image,
                    self.device,
                    class_ids,
                )
            except (RuntimeError, ValueError):
                pass

            print(
                f"[WARN] Vehicle GPU inference basarisiz ({error}); "
                "CPU ile yeniden deneniyor."
            )
            self.device = "cpu"
            self.use_half = False
            move_model_to_cpu(self.model)
            return self._predict_on_device(bgr_image, "cpu", class_ids)

    def _predict_on_device(self, bgr_image, device, class_ids=None):
        """YOLO çağrısını bütün seçenekleri görünür biçimde yapar."""
        if class_ids is None:
            class_ids = self.vehicle_class_ids
        return self.model.predict(
            source=bgr_image,
            conf=self.confidence,
            iou=0.45,
            imgsz=self.image_size,
            device=device,
            classes=class_ids,
            verbose=False,
            max_det=100,
            half=bool(
                getattr(self, "use_half", False)
                and is_cuda_device(device)
            ),
        )

    def _validate_image(self, rgb_image):
        if rgb_image is None:
            raise ValueError("VehicleDetector bos kamera goruntusu aldi.")
        if not isinstance(rgb_image, np.ndarray):
            raise TypeError("VehicleDetector numpy.ndarray bekliyor.")
        if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
            raise ValueError(
                "Kamera goruntusu HxWx3 formatinda olmali; "
                f"gelen shape: {rgb_image.shape}"
            )

    def _warmup_cuda(self):
        """İlk canlı karedeki model hazırlama maliyetini uygulama açılışında öder."""
        if not is_cuda_device(self.device):
            return
        dummy = np.zeros(
            (self.image_size, self.image_size, 3),
            dtype=np.uint8,
        )
        self._predict(dummy, self.relevant_class_ids)

    def _find_vehicle_class_ids(self, names):
        selected = []

        for class_id, class_name in names.items():
            normalized = (
                str(class_name).strip().lower().replace("-", " ").replace("_", " ")
            )
            words = normalized.split()

            vehicle_word_found = False
            for word in words:
                if word in self.VEHICLE_NAMES:
                    vehicle_word_found = True
                    break

            if normalized in self.VEHICLE_NAMES or vehicle_word_found:
                number = int(class_id)
                if number not in selected:
                    selected.append(number)

        selected.sort()
        return selected

    def _find_relevant_class_ids(self, names):
        selected = []
        for class_id, class_name in names.items():
            normalized = str(class_name).strip().lower().replace("-", "_")
            if normalized in self.RELEVANT_NAMES:
                selected.append(int(class_id))
        selected.sort()
        return selected or list(self.vehicle_class_ids)

    def _detect_many_selected(self, images_by_name, class_ids, vehicles_only):
        camera_names = []
        bgr_images = []
        for camera_name, rgb_image in images_by_name.items():
            camera_names.append(camera_name)
            bgr_images.append(self.prepare_image(rgb_image))

        detections_by_name = {}
        if not bgr_images:
            return detections_by_name

        source = bgr_images[0] if len(bgr_images) == 1 else bgr_images
        results = self._predict(source, class_ids)
        for index, camera_name in enumerate(camera_names):
            detections = []
            if results and index < len(results) and results[index].boxes is not None:
                detections = self._parse_selected_result(
                    results[index],
                    bgr_images[index],
                    class_ids,
                    vehicles_only,
                )
            detections_by_name[camera_name] = detections
        return detections_by_name

    def _normalize_names(self, names):
        normalized = {}
        if isinstance(names, dict):
            for class_id, class_name in names.items():
                normalized[int(class_id)] = str(class_name)
            return normalized
        if isinstance(names, (list, tuple)):
            for index, class_name in enumerate(names):
                normalized[index] = str(class_name)
        return normalized

import os

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    """
    RGB kamera görüntüsünde şu nesneleri algılar:

    0 -> bike
    1 -> motobike
    9 -> vehicle
    """

    # Kullanacağımız YOLO sınıfları
    TARGET_CLASS_IDS = [0, 1, 9]

    CLASS_NAMES = {
        0: "bike",
        1: "motobike",
        9: "vehicle",
    }

    def __init__(
        self,
        model_path=None,
        confidence=0.35,
        device=None,
    ):
        """
        VehicleDetector nesnesini oluşturur ve YOLO modelini yükler.

        model_path:
            Model dosyasının yolu.

        confidence:
            Minimum güven değeri.
            Örneğin 0.35, yüzde 35 güven anlamına gelir.

        device:
            None     -> Ultralytics otomatik seçer
            "cpu"    -> İşlemci kullanır
            "cuda:0" -> Ekran kartı kullanır
        """

        # Model yolu verilmediyse varsayılan modeli kullan.
        if model_path is None:
            detector_folder = os.path.dirname(__file__)
            project_folder = os.path.dirname(detector_folder)

            model_path = os.path.join(
                project_folder,
                "models",
                "carla_yolov8n_best.pt",
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO modeli bulunamadı: {model_path}"
            )

        self.confidence = confidence
        self.device = device

        # Model program başlarken yalnızca bir kez yüklenir.
        self.model = YOLO(model_path)

        print("[VehicleDetector] YOLO modeli yüklendi.")
        print("[VehicleDetector] Algılanacak sınıflar:")
        print("  0 -> bike")
        print("  1 -> motobike")
        print("  9 -> vehicle")

    def detect(self, rgb_image):
        """
        Verilen RGB görüntüsünde araç, bisiklet ve motosiklet algılar.

        Parametre:
            rgb_image:
                Kameradan gelen NumPy RGB görüntüsü.

        Dönen değer:
            Algılanan nesnelerin bulunduğu liste.

        Örnek:

            [
                {
                    "class_id": 9,
                    "class_name": "vehicle",
                    "confidence": 0.92,
                    "bbox": {
                        "x1": 100,
                        "y1": 150,
                        "x2": 300,
                        "y2": 400
                    },
                    "center_x": 200,
                    "center_y": 275,
                    "width": 200,
                    "height": 250
                }
            ]
        """

        if rgb_image is None:
            return []

        if not isinstance(rgb_image, np.ndarray):
            raise TypeError(
                "rgb_image bir NumPy dizisi olmalıdır."
            )

        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(
                "Görüntü 3 kanallı olmalıdır: (height, width, 3)"
            )

        # Kameradan gelen görüntü RGB.
        # Ultralytics NumPy görüntüsünü BGR formatında bekler.
        bgr_image = rgb_image[:, :, ::-1]

        # YOLO tahmin ayarları
        predict_options = {
            "source": bgr_image,
            "conf": self.confidence,
            "classes": self.TARGET_CLASS_IDS,
            "verbose": False,
        }

        # Device verilmişse ayarlara ekle.
        if self.device is not None:
            predict_options["device"] = self.device

        results = self.model.predict(**predict_options)

        detections = []

        # İlk görüntünün sonucunu al.
        result = results[0]

        if result.boxes is None:
            return detections

        # Bulunan her nesneyi işle.
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Bounding box koordinatlarını al.
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            width = x2 - x1
            height = y2 - y1

            center_x = x1 + width // 2
            center_y = y1 + height // 2

            detection = {
                "class_id": class_id,
                "class_name": self.CLASS_NAMES[class_id],
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "center_x": center_x,
                "center_y": center_y,
                "width": width,
                "height": height,
            }

            detections.append(detection)

        return detections
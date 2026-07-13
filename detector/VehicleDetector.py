from pathlib import Path

import numpy as np
from ultralytics import YOLO


class VehicleDetector:
    def __init__(self):
        # VehicleDetector.py dosyası:
        # Carla-Workflow-main/detector/VehicleDetector.py
        self.project_root = Path(__file__).resolve().parent.parent

        self.model_path = (
            self.project_root
            / "models"
            / "carla_yolov8n_best.pt"
        )

        self.data_folder = (
            self.project_root
            / "data"
            / "runs"
        )

        self.output_image = (
            self.project_root
            /"detector"
            / "images"
            / "detection_result.jpg"
        )

        self.min_confidence = 0.80

        self.vehicle_class_names = {
            "vehicle",
            "bike",
            "motobike"
        }

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model bulunamadı: {self.model_path}"
            )

        self.output_image.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model = YOLO(
            str(self.model_path)
        )

    def find_latest_rgb_frame(self):
        rgb_files = list(
            self.data_folder.glob(
                "run_*/sensors/rgb_camera/*.npy"
            )
        )

        if not rgb_files:
            raise FileNotFoundError(
                "Kaydedilmiş RGB kamera görüntüsü bulunamadı.\n"
                "Önce CARLA serverını açıp "
                "'python main.py' çalıştır."
            )

        latest_file = max(
            rgb_files,
            key=lambda file_path: file_path.stat().st_mtime,
        )

        return latest_file

    def load_rgb_image(self, image_path):
        """
        Kaydedilmiş NumPy kamera görüntüsünü yükler.
        """

        image = np.load(
            image_path,
            allow_pickle=False,
        )

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Görüntü formatı hatalı: {image.shape}\n"
                "Beklenen format: yükseklik x genişlik x 3"
            )

        return image

    def detect_vehicles(self, image):
        """
        YOLO modelini çalıştırır ve yalnızca araçları döndürür.
        """

        results = self.model.predict(
            source=image,
            conf=self.min_confidence,
            verbose=False,
        )

        result = results[0]
        detected_vehicles = []

        if result.boxes is None:
            return detected_vehicles, result

        for box in result.boxes:
            class_id = int(
                box.cls[0].item()
            )

            class_name = str(
                result.names[class_id]
            ).lower()

            if class_name not in self.vehicle_class_names:
                continue

            confidence = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = [
                int(value)
                for value in box.xyxy[0].tolist()
            ]

            width = x2 - x1
            height = y2 - y1
            area = width * height

            vehicle = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                },
                "area_pixels": area,
            }

            detected_vehicles.append(vehicle)

        detected_vehicles.sort(
            key=lambda vehicle: vehicle["area_pixels"],
            reverse=True,
        )

        return detected_vehicles, result
    
def main():
    try:
        # VehicleDetector sınıfından bir nesne oluştur.
        detector = VehicleDetector()

        print("[INFO] Son RGB kamera görüntüsü aranıyor...")

        # En son kaydedilen RGB görüntüsünü bul.
        image_path = detector.find_latest_rgb_frame()

        print(f"[OK] Görüntü bulundu: {image_path}")

        # NumPy görüntüsünü yükle.
        image = detector.load_rgb_image(image_path)

        print(f"[INFO] Görüntü boyutu: {image.shape}")

        # Görüntüdeki araçları tespit et.
        detected_vehicles, result = detector.detect_vehicles(
            image=image
        )

        if not detected_vehicles:
            print("[INFO] Görüntüde araç tespit edilmedi.")

        else:
            print(
                f"[OK] Toplam {len(detected_vehicles)} araç tespit edildi."
            )

            for index, vehicle in enumerate(
                detected_vehicles,
                start=1,
            ):
                bbox = vehicle["bbox"]

                print()
                print(f"Araç {index}")
                print(
                    f"  Sınıf ID: {vehicle['class_id']}"
                )
                print(
                    f"  Sınıf adı: {vehicle['class_name']}"
                )
                print(
                    f"  Güven: %{vehicle['confidence'] * 100:.2f}"
                )
                print(
                    "  Kutu: "
                    f"({bbox['x1']}, {bbox['y1']}) - "
                    f"({bbox['x2']}, {bbox['y2']})"
                )
                print(
                    f"  Alan: {vehicle['area_pixels']} piksel"
                )

        # Modelin çizdiği kutulu görüntüyü kaydet.
        result.save(
            filename=str(detector.output_image)
        )

        print()
        print(
            f"[OK] Sonuç görüntüsü kaydedildi: "
            f"{detector.output_image}"
        )

    except Exception as error:
        print(f"[ERROR] {error}")


if __name__ == "__main__":
    main()
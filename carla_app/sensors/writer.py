import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class DatasetWriter:
    def __init__(self, output_folder: Path):
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.run_folder = output_folder / run_name
        self.rgb_folder = self.run_folder / "rgb"
        self.lidar_folder = self.run_folder / "lidar"
        self.meta_folder = self.run_folder / "metadata"

        for folder in (self.rgb_folder, self.lidar_folder, self.meta_folder):
            folder.mkdir(parents=True, exist_ok=True)

        print(f"[OK] Veri klasoru: {self.run_folder}")

    def save(self, frame_id, sensor_data, vehicle_data):
        name = f"{int(frame_id):08d}"
        bgr_image = cv2.cvtColor(sensor_data["rgb"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.rgb_folder / f"{name}.png"), bgr_image)
        np.save(self.lidar_folder / f"{name}.npy", sensor_data["lidar"])

        metadata = {
            "frame_id": int(frame_id),
            "vehicle": vehicle_data,
            "gnss": sensor_data["gnss"],
            "imu": sensor_data["imu"],
            "radars": sensor_data["radars"],
        }
        with (self.meta_folder / f"{name}.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)

"""İşlenmiş sensör verilerini diske kaydeder."""

import json
import os
from datetime import datetime

import numpy as np


class DatasetWriter:
    """Sensör verilerini düzenli klasörlere kaydeder."""

    def __init__(self, output_folder="data/runs", run_name=None):
        if run_name is None:
            current_time = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            run_name = f"run_{current_time}"

        self.run_folder = os.path.join(
            output_folder,
            run_name,
        )

        self.frames_folder = os.path.join(
            self.run_folder,
            "frames",
        )

        self.sensors_folder = os.path.join(
            self.run_folder,
            "sensors",
        )

        self.manifest_path = os.path.join(
            self.run_folder,
            "manifest.jsonl",
        )

        os.makedirs(
            self.frames_folder,
            exist_ok=True,
        )

        os.makedirs(
            self.sensors_folder,
            exist_ok=True,
        )

        print(
            f"[OK] Dataset klasörü oluşturuldu: {self.run_folder}"
        )

    def save_frame(
        self,
        packet,
        processed_data,
        vehicle_data=None,
    ):
        """Bir frame'e ait bütün sensör verilerini kaydeder."""

        if vehicle_data is None:
            vehicle_data = {}

        frame_name = str(packet.frame_id).zfill(6)

        saved_sensor_files = {}

        for sensor_name, sensor_data in processed_data.items():
            saved_path = self._save_sensor(
                sensor_name=sensor_name,
                frame_name=frame_name,
                sensor_data=sensor_data,
            )

            saved_sensor_files[sensor_name] = saved_path

        frame_info = packet.to_dict()

        frame_info["vehicle_data"] = vehicle_data
        frame_info["sensor_files"] = saved_sensor_files

        frame_json_path = os.path.join(
            self.frames_folder,
            f"{frame_name}.json",
        )

        with open(
            frame_json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                frame_info,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self._append_manifest(frame_info)

        print(
            f"[OK] Frame {packet.frame_id} kaydedildi."
        )

        return frame_info

    def _save_sensor(
        self,
        sensor_name,
        frame_name,
        sensor_data,
    ):
        """Tek bir sensör verisini kaydeder."""

        sensor_folder = os.path.join(
            self.sensors_folder,
            sensor_name,
        )

        os.makedirs(
            sensor_folder,
            exist_ok=True,
        )

        # Kamera, semantic camera ve LiDAR NumPy dizisidir.
        if isinstance(sensor_data, np.ndarray):
            file_name = f"{frame_name}.npy"

            full_path = os.path.join(
                sensor_folder,
                file_name,
            )

            np.save(
                full_path,
                sensor_data,
            )

        # GNSS ve IMU sözlüktür.
        elif isinstance(sensor_data, dict):
            file_name = f"{frame_name}.json"

            full_path = os.path.join(
                sensor_folder,
                file_name,
            )

            with open(
                full_path,
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    sensor_data,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        else:
            raise TypeError(
                f"{sensor_name} verisi kaydedilemiyor. "
                f"Veri tipi: {type(sensor_data)}"
            )

        # Manifest içine uzun absolute path yerine göreceli path yazıyoruz.
        relative_path = os.path.relpath(
            full_path,
            self.run_folder,
        )

        return relative_path

    def _append_manifest(self, frame_info):
        """Frame özetini manifest.jsonl dosyasına ekler."""

        short_info = {
            "frame_id": frame_info["frame_id"],
            "complete": frame_info["complete"],
            "missing_sensors": frame_info["missing_sensors"],
            "sensor_files": frame_info["sensor_files"],
            "vehicle_data": frame_info["vehicle_data"],
        }

        with open(
            self.manifest_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    short_info,
                    ensure_ascii=False,
                )
            )

            file.write("\n")
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class DatasetWriter:
    def __init__(
        self,
        output_folder: Path,
    ):
        run_name = datetime.now().strftime(
            "run_%Y%m%d_%H%M%S"
        )

        self.run_folder = (
            output_folder / run_name
        )

        # Ana algılama kamerası geriye uyumluluk için
        # mevcut rgb/ klasöründe tutulur.
        self.rgb_folder = (
            self.run_folder / "rgb"
        )

        # Diğer altı kamera burada tutulur.
        self.cameras_folder = (
            self.run_folder / "cameras"
        )

        self.lidar_folder = (
            self.run_folder / "lidar"
        )

        self.meta_folder = (
            self.run_folder / "metadata"
        )

        self.calibration_folder = (
            self.run_folder / "calibration"
        )

        self.primary_camera = None

        for folder in (
            self.rgb_folder,
            self.cameras_folder,
            self.lidar_folder,
            self.meta_folder,
            self.calibration_folder,
        ):
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        print(
            f"[OK] Veri klasoru: "
            f"{self.run_folder}"
        )

    def write_manifest(
        self,
        manifest,
    ):
        self.primary_camera = (
            manifest["primary_camera"]
        )

        manifest_path = (
            self.calibration_folder
            / "sensor_manifest.json"
        )

        with manifest_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
                ensure_ascii=False,
            )

        print(
            f"[OK] Sensor manifesti: "
            f"{manifest_path}"
        )

    @staticmethod
    def _write_rgb(
        path: Path,
        rgb_image,
    ) -> None:
        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        written = cv2.imwrite(
            str(path),
            bgr_image,
        )

        if not written:
            raise IOError(
                f"Goruntu yazilamadi: {path}"
            )

    def save(
        self,
        frame_id,
        sensor_data,
        vehicle_data,
    ):
        name = f"{int(frame_id):08d}"

        primary_camera = (
            sensor_data["primary_camera"]
        )

        camera_files = {}

        for camera_name, rgb_image in (
            sensor_data["cameras"].items()
        ):
            if camera_name == primary_camera:
                path = (
                    self.rgb_folder
                    / f"{name}.png"
                )
            else:
                camera_folder = (
                    self.cameras_folder
                    / camera_name
                )

                camera_folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path = (
                    camera_folder
                    / f"{name}.png"
                )

            self._write_rgb(
                path,
                rgb_image,
            )

            camera_files[camera_name] = str(
                path.relative_to(
                    self.run_folder
                )
            )

        lidar_path = (
            self.lidar_folder
            / f"{name}.npy"
        )

        np.save(
            lidar_path,
            sensor_data["lidar"],
        )

        metadata = {
            "frame_id": int(frame_id),
            "primary_camera": primary_camera,
            "files": {
                "cameras": camera_files,
                "lidar": str(
                    lidar_path.relative_to(
                        self.run_folder
                    )
                ),
            },
            "vehicle": vehicle_data,
            "gnss": sensor_data["gnss"],
            "imu": sensor_data["imu"],
            "radars": sensor_data["radars"],
            "ultrasonics": (
                sensor_data["ultrasonics"]
            ),
        }

        metadata_path = (
            self.meta_folder
            / f"{name}.json"
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )
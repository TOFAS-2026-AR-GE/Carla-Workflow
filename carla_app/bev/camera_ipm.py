"""Yedi RGB kamerayı homografi ile ortak kuş bakışı düzleme taşır."""

import cv2
import numpy as np


class CameraIpm:
    """Kamera görüntülerini geometrik örtüşme ağırlıklarıyla birleştirir."""

    def __init__(self, layout, calibrations, grid):
        self.layout = layout
        self.calibrations = calibrations
        self.grid = grid
        self.camera_maps = {}
        self.build_camera_maps()

    def camera_range(self, camera_name):
        if camera_name == "camera_front_narrow":
            return 80.0
        if camera_name == "camera_front_wide":
            return 60.0
        return 45.0

    def build_camera_maps(self):
        """Her BEV pikselinin kaynak kameradaki karşılığını bir kez hesaplar."""
        forward, right = self.grid.metric_mesh()
        ground_points = np.column_stack((forward.ravel(), right.ravel()))

        geometry = self.layout.vehicle_geometry
        half_length = float(geometry["half_length_m"]) + 0.20
        half_width = float(geometry["half_width_m"]) + 0.20
        outside_ego = (np.abs(forward) > half_length) | (
            np.abs(right) > half_width
        )

        for camera in self.layout.cameras:
            calibration = self.calibrations.get_camera(camera.name)
            pixels, depth = calibration.project_ground_points(ground_points)
            map_x = pixels[:, 0].reshape(self.grid.height, self.grid.width)
            map_y = pixels[:, 1].reshape(self.grid.height, self.grid.width)
            camera_depth = depth.reshape(self.grid.height, self.grid.width)

            maximum_range = self.camera_range(camera.name)
            distance = np.hypot(forward, right)
            valid = camera_depth > 0.30
            valid &= distance <= maximum_range
            valid &= map_x >= 0.0
            valid &= map_x <= calibration.width - 1
            valid &= map_y >= 0.38 * calibration.height
            valid &= map_y <= calibration.height - 1
            valid &= outside_ego

            horizontal_distance = np.abs(
                (map_x - 0.5 * calibration.width)
                / (0.5 * calibration.width)
            )
            edge_weight = np.clip(1.0 - horizontal_distance, 0.0, 1.0)
            edge_weight = 0.08 + 0.92 * edge_weight**2

            vertical_weight = (
                map_y / max(1.0, float(calibration.height)) - 0.35
            ) / 0.65
            vertical_weight = np.clip(vertical_weight, 0.12, 1.0)
            fov_weight = np.clip(90.0 / calibration.fov_deg, 0.75, 1.80)
            weight = edge_weight * vertical_weight * fov_weight
            weight[~valid] = 0.0

            self.camera_maps[camera.name] = {
                "map_x": map_x.astype(np.float32),
                "map_y": map_y.astype(np.float32),
                "weight": weight.astype(np.float32),
            }

    def build_mosaic(
        self,
        camera_results,
        motion_compensator=None,
        current_frame_id=None,
    ):
        """Mevcut kameraları tek kuş bakışı görüntüde birleştirir."""
        shape = (self.grid.height, self.grid.width, 3)
        color_sum = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros((self.grid.height, self.grid.width), dtype=np.float32)
        used_cameras = []

        for camera in self.layout.cameras:
            result = camera_results.get(camera.name)
            if result is None or result.get("image") is None:
                continue

            calibration = self.calibrations.get_camera(camera.name)
            rgb_image = np.asarray(result["image"])
            if rgb_image.ndim != 3 or rgb_image.shape[2] < 3:
                continue

            bgr_image = np.ascontiguousarray(rgb_image[:, :, :3][:, :, ::-1])
            if (
                bgr_image.shape[1] != calibration.width
                or bgr_image.shape[0] != calibration.height
            ):
                bgr_image = cv2.resize(
                    bgr_image,
                    (calibration.width, calibration.height),
                    interpolation=cv2.INTER_LINEAR,
                )

            camera_map = self.camera_maps[camera.name]
            warped = cv2.remap(
                bgr_image,
                camera_map["map_x"],
                camera_map["map_y"],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            weight = camera_map["weight"]

            measurement_frame = result.get("frame_id")
            if motion_compensator is not None and current_frame_id is not None:
                warp_matrix = motion_compensator.image_warp_matrix(
                    self.grid,
                    measurement_frame,
                    current_frame_id,
                )
                warped = cv2.warpPerspective(
                    warped,
                    warp_matrix,
                    (self.grid.width, self.grid.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                weight = cv2.warpPerspective(
                    weight,
                    warp_matrix,
                    (self.grid.width, self.grid.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                )

            color_sum += warped.astype(np.float32) * weight[:, :, None]
            weight_sum += weight
            used_cameras.append(camera.name)

        mosaic = np.full(shape, (24, 27, 31), dtype=np.uint8)
        covered = weight_sum > 1e-4
        if np.any(covered):
            blended = color_sum[covered] / weight_sum[covered, None]
            mosaic[covered] = np.clip(blended, 0, 255).astype(np.uint8)

        coverage = np.clip(weight_sum, 0.0, 1.0)
        return mosaic, coverage, used_cameras

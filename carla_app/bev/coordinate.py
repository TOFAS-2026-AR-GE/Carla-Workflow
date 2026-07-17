"""Metre tabanlı BEV ızgarasını ve ego hareket telafisini yönetir."""

import math
import threading

import numpy as np


class MetricGrid:
    """Ego koordinatı ile görüntü pikseli arasındaki dönüşümü tutar."""

    def __init__(
        self,
        width=800,
        height=600,
        forward_range_m=60.0,
        rear_range_m=20.0,
        side_range_m=30.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.forward_range_m = float(forward_range_m)
        self.rear_range_m = float(rear_range_m)
        self.side_range_m = float(side_range_m)

        if self.width <= 1 or self.height <= 1:
            raise ValueError("BEV görüntü boyutu en az 2x2 olmalı.")

        horizontal_scale = (self.width - 1) / (2.0 * self.side_range_m)
        vertical_scale = (self.height - 1) / (
            self.forward_range_m + self.rear_range_m
        )
        self.metric_to_image = np.array(
            [
                [0.0, horizontal_scale, self.side_range_m * horizontal_scale],
                [-vertical_scale, 0.0, self.forward_range_m * vertical_scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.image_to_metric = np.linalg.inv(self.metric_to_image)

    def to_pixel(self, forward_m, right_m):
        point = self.metric_to_image @ np.array(
            [float(forward_m), float(right_m), 1.0],
            dtype=np.float64,
        )
        return int(round(point[0])), int(round(point[1]))

    def to_metric(self, pixel_x, pixel_y):
        point = self.image_to_metric @ np.array(
            [float(pixel_x), float(pixel_y), 1.0],
            dtype=np.float64,
        )
        return float(point[0]), float(point[1])

    def metric_mesh(self):
        pixel_y, pixel_x = np.indices((self.height, self.width))
        pixels = np.stack(
            (
                pixel_x.ravel(),
                pixel_y.ravel(),
                np.ones(self.width * self.height),
            ),
            axis=0,
        )
        metric = self.image_to_metric @ pixels
        forward = metric[0].reshape(self.height, self.width)
        right = metric[1].reshape(self.height, self.width)
        return forward, right


class EgoMotionCompensator:
    """Eski sensör ölçümlerini ölçüm anından güncel ego karesine taşır."""

    def __init__(self, maximum_history=160):
        self.maximum_history = int(maximum_history)
        self.poses = {}
        self.lock = threading.Lock()

    def remember(self, frame_id, vehicle_state):
        if vehicle_state is None or vehicle_state.get("location") is None:
            return

        location = vehicle_state["location"]
        with self.lock:
            self.poses[int(frame_id)] = {
                "x": float(location.x),
                "y": float(location.y),
                "yaw_deg": float(vehicle_state.get("yaw", 0.0)),
            }

            while len(self.poses) > self.maximum_history:
                del self.poses[min(self.poses)]

    def get_pose(self, frame_id):
        if frame_id is None:
            return None
        with self.lock:
            pose = self.poses.get(int(frame_id))
            return None if pose is None else dict(pose)

    def old_ego_to_current_matrix(self, old_frame_id, current_frame_id):
        """Eski ego XY koordinatını güncel ego XY koordinatına taşıyan matris."""
        old_pose = self.get_pose(old_frame_id)
        current_pose = self.get_pose(current_frame_id)
        if old_pose is None or current_pose is None:
            return np.eye(3, dtype=np.float64)

        old_yaw = math.radians(old_pose["yaw_deg"])
        current_yaw = math.radians(current_pose["yaw_deg"])

        old_to_world = np.array(
            [
                [math.cos(old_yaw), -math.sin(old_yaw), old_pose["x"]],
                [math.sin(old_yaw), math.cos(old_yaw), old_pose["y"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        current_to_world = np.array(
            [
                [
                    math.cos(current_yaw),
                    -math.sin(current_yaw),
                    current_pose["x"],
                ],
                [
                    math.sin(current_yaw),
                    math.cos(current_yaw),
                    current_pose["y"],
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return np.linalg.inv(current_to_world) @ old_to_world

    def compensate_points(self, points, old_frame_id, current_frame_id):
        """Nx2 veya Nx3 noktaları eski ego karesinden güncel kareye taşır."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
            return points.copy()

        matrix = self.old_ego_to_current_matrix(old_frame_id, current_frame_id)
        homogeneous = np.column_stack(
            (points[:, 0], points[:, 1], np.ones(len(points)))
        )
        compensated = homogeneous @ matrix.T
        result = points.copy()
        result[:, 0] = compensated[:, 0]
        result[:, 1] = compensated[:, 1]
        return result

    def image_warp_matrix(self, grid, old_frame_id, current_frame_id):
        """Eski BEV görüntüsünü güncel ego karesine taşıyan piksel matrisi."""
        motion = self.old_ego_to_current_matrix(old_frame_id, current_frame_id)
        return grid.metric_to_image @ motion @ grid.image_to_metric

"""LiDAR, radar ve izlenen nesnelerden kalıcı olasılıksal occupancy grid üretir."""

import math

import cv2
import numpy as np

from carla_app.bev.coordinate import MetricGrid


class OccupancyGrid:
    """Boş, dolu ve bilinmeyen hücreleri log-odds değeriyle biriktirir."""

    def __init__(
        self,
        forward_range_m=60.0,
        rear_range_m=20.0,
        side_range_m=30.0,
        cell_size_m=0.25,
        fixed_delta_seconds=0.05,
        decay_half_life_seconds=2.0,
    ):
        self.cell_size_m = float(cell_size_m)
        self.fixed_delta_seconds = max(1e-3, float(fixed_delta_seconds))
        self.decay_half_life_seconds = max(
            self.fixed_delta_seconds,
            float(decay_half_life_seconds),
        )
        width = int(math.ceil(2.0 * side_range_m / self.cell_size_m))
        height = int(
            math.ceil((forward_range_m + rear_range_m) / self.cell_size_m)
        )
        self.grid = MetricGrid(
            width=width,
            height=height,
            forward_range_m=forward_range_m,
            rear_range_m=rear_range_m,
            side_range_m=side_range_m,
        )
        self.log_odds = np.zeros((height, width), dtype=np.float32)
        self.previous_frame_id = None
        self.last_measurement_frames = {}

    def update(
        self,
        lidar_points,
        lidar_origin,
        radar_points,
        tracked_objects,
        ground_z_m,
        motion_compensator,
        current_frame_id,
        measurement_frames=None,
    ):
        self.compensate_old_grid(motion_compensator, current_frame_id)
        frame_delta = 1
        if self.previous_frame_id is not None:
            frame_delta = max(1, int(current_frame_id) - self.previous_frame_id)
        elapsed_seconds = frame_delta * self.fixed_delta_seconds
        self.log_odds *= 0.5 ** (elapsed_seconds / self.decay_half_life_seconds)

        measurement_frames = dict(measurement_frames or {})
        lidar_frame = measurement_frames.get("lidar_roof")
        if self.is_new_measurement("lidar_roof", lidar_frame):
            self.add_lidar(lidar_points, lidar_origin, ground_z_m)
            self.remember_measurement("lidar_roof", lidar_frame)

        fresh_radar_points = self.fresh_radar_points(radar_points)
        self.add_radar(fresh_radar_points)
        self.remember_radar_frames(fresh_radar_points)
        np.clip(self.log_odds, -4.0, 4.0, out=self.log_odds)

        # Kalıcı grid yalnız ham sensör kanıtını tutar. Track kutuları geçici
        # görsel overlay'dir; doğrulamanın kendi sonucunu tekrar kanıt olarak
        # kullanmasını önlemek için sensor_probability'ye karışmaz.
        composite_log_odds = self.log_odds.copy()
        self.add_tracked_objects(tracked_objects, composite_log_odds)
        np.clip(composite_log_odds, -4.0, 4.0, out=composite_log_odds)
        self.previous_frame_id = int(current_frame_id)
        return self.build_state(
            composite_log_odds,
            sensor_log_odds=self.log_odds,
        )

    def compensate_old_grid(self, motion_compensator, current_frame_id):
        if self.previous_frame_id is None:
            return
        matrix = motion_compensator.image_warp_matrix(
            self.grid,
            self.previous_frame_id,
            current_frame_id,
        )
        self.log_odds = cv2.warpPerspective(
            self.log_odds,
            matrix,
            (self.grid.width, self.grid.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

    def add_lidar(self, lidar_points, lidar_origin, ground_z_m):
        points = np.asarray(lidar_points)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return

        maximum_rays = 1800
        step = max(1, int(math.ceil(len(points) / maximum_rays)))
        origin_x = float(lidar_origin[0])
        origin_y = float(lidar_origin[1])
        for point in points[::step]:
            endpoint_is_occupied = float(point[2]) > float(ground_z_m) + 0.18
            self.add_ray(
                origin_x,
                origin_y,
                float(point[0]),
                float(point[1]),
                endpoint_is_occupied,
                occupied_delta=0.85,
            )

    def add_radar(self, radar_points):
        maximum_rays = 500
        step = max(1, int(math.ceil(len(radar_points) / maximum_rays)))
        for point in radar_points[::step]:
            if point.get("ground_return", False):
                continue
            self.add_ray(
                float(point.get("origin_x_m", 0.0)),
                float(point.get("origin_y_m", 0.0)),
                float(point["x_m"]),
                float(point["y_m"]),
                True,
                occupied_delta=0.55,
            )

    def add_tracked_objects(self, tracked_objects, log_odds=None):
        target = self.log_odds if log_odds is None else log_odds
        for tracked in tracked_objects:
            if not tracked.get("confirmed", False):
                continue
            if int(tracked.get("misses", 0)) > 0:
                continue
            center_x, center_y = self.grid.to_pixel(
                tracked["x_m"],
                tracked["y_m"],
            )
            half_length = max(1, int(tracked["length_m"] / self.cell_size_m / 2.0))
            half_width = max(1, int(tracked["width_m"] / self.cell_size_m / 2.0))
            x1 = max(0, center_x - half_width)
            x2 = min(self.grid.width - 1, center_x + half_width)
            y1 = max(0, center_y - half_length)
            y2 = min(self.grid.height - 1, center_y + half_length)
            if x1 <= x2 and y1 <= y2:
                target[y1 : y2 + 1, x1 : x2 + 1] += 0.75

    def is_new_measurement(self, sensor_name, frame_id):
        if frame_id is None:
            return True
        previous = self.last_measurement_frames.get(sensor_name)
        return previous is None or int(frame_id) > previous

    def remember_measurement(self, sensor_name, frame_id):
        if frame_id is not None:
            self.last_measurement_frames[sensor_name] = int(frame_id)

    def fresh_radar_points(self, radar_points):
        fresh = []
        for point in radar_points:
            sensor_name = str(point.get("sensor_name", "radar"))
            frame_id = point.get("frame_id")
            if self.is_new_measurement(sensor_name, frame_id):
                fresh.append(point)
        return fresh

    def remember_radar_frames(self, radar_points):
        newest = {}
        for point in radar_points:
            sensor_name = str(point.get("sensor_name", "radar"))
            frame_id = point.get("frame_id")
            if frame_id is None:
                continue
            newest[sensor_name] = max(
                int(frame_id),
                newest.get(sensor_name, int(frame_id)),
            )
        for sensor_name, frame_id in newest.items():
            self.remember_measurement(sensor_name, frame_id)

    def add_ray(
        self,
        origin_forward,
        origin_right,
        endpoint_forward,
        endpoint_right,
        endpoint_is_occupied,
        occupied_delta,
    ):
        start_x, start_y = self.grid.to_pixel(origin_forward, origin_right)
        end_x, end_y = self.grid.to_pixel(endpoint_forward, endpoint_right)
        cells = self.bresenham(start_x, start_y, end_x, end_y)
        if not cells:
            return

        free_cells = cells[:-1] if endpoint_is_occupied else cells
        for pixel_x, pixel_y in free_cells:
            if 0 <= pixel_x < self.grid.width and 0 <= pixel_y < self.grid.height:
                self.log_odds[pixel_y, pixel_x] -= 0.40

        if endpoint_is_occupied:
            pixel_x, pixel_y = cells[-1]
            if 0 <= pixel_x < self.grid.width and 0 <= pixel_y < self.grid.height:
                self.log_odds[pixel_y, pixel_x] += float(occupied_delta)

    def bresenham(self, start_x, start_y, end_x, end_y):
        cells = []
        delta_x = abs(end_x - start_x)
        delta_y = -abs(end_y - start_y)
        step_x = 1 if start_x < end_x else -1
        step_y = 1 if start_y < end_y else -1
        error = delta_x + delta_y
        pixel_x = start_x
        pixel_y = start_y

        maximum_steps = delta_x + abs(delta_y) + 2
        for _step in range(maximum_steps):
            cells.append((pixel_x, pixel_y))
            if pixel_x == end_x and pixel_y == end_y:
                break
            twice_error = 2 * error
            if twice_error >= delta_y:
                error += delta_y
                pixel_x += step_x
            if twice_error <= delta_x:
                error += delta_x
                pixel_y += step_y
        return cells

    def build_state(self, log_odds=None, sensor_log_odds=None):
        combined = self.log_odds if log_odds is None else log_odds
        sensor = self.log_odds if sensor_log_odds is None else sensor_log_odds
        probability = 1.0 / (1.0 + np.exp(-combined))
        sensor_probability = 1.0 / (1.0 + np.exp(-sensor))
        known = np.abs(combined) >= 0.20
        occupied = probability >= 0.62
        free = probability <= 0.42
        return {
            "probability": probability.astype(np.float32),
            "sensor_probability": sensor_probability.astype(np.float32),
            "known": known,
            "occupied": occupied,
            "free": free,
            "cell_size_m": self.cell_size_m,
            "forward_range_m": self.grid.forward_range_m,
            "rear_range_m": self.grid.rear_range_m,
            "side_range_m": self.grid.side_range_m,
            "measurement_frames": dict(self.last_measurement_frames),
        }

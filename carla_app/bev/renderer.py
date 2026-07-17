"""Metre cinsindeki BEV sahnesini OpenCV görüntüsüne çizer."""

import cv2
import numpy as np


class BevRenderer:
    """Ego aracı altta, ileri yön yukarı olacak biçimde BEV çizer."""

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

    def render(self, scene, current_frame_id=None):
        canvas_shape = (self.height, self.width, 3)
        canvas = np.full(canvas_shape, (24, 27, 31), dtype=np.uint8)
        self.draw_grid(canvas)
        self.draw_route(canvas, scene.get("route_points", []))
        self.draw_lidar(canvas, scene.get("lidar_points"))
        self.draw_radars(canvas, scene.get("radar_points", []))
        self.draw_detections(canvas, scene.get("detections", []))
        self.draw_ego(canvas, scene.get("vehicle_geometry", {}))
        self.draw_header(canvas, scene, current_frame_id)
        return canvas

    def to_pixel(self, forward_m, right_m):
        horizontal_fraction = (
            float(right_m) + self.side_range_m
        ) / (2.0 * self.side_range_m)
        vertical_fraction = (
            self.forward_range_m - float(forward_m)
        ) / (self.forward_range_m + self.rear_range_m)
        pixel_x = int(round(horizontal_fraction * (self.width - 1)))
        pixel_y = int(round(vertical_fraction * (self.height - 1)))
        return pixel_x, pixel_y

    def inside_image(self, pixel_x, pixel_y):
        return 0 <= pixel_x < self.width and 0 <= pixel_y < self.height

    def draw_grid(self, canvas):
        for forward_m in range(
            -int(self.rear_range_m),
            int(self.forward_range_m) + 1,
            10,
        ):
            start = self.to_pixel(forward_m, -self.side_range_m)
            end = self.to_pixel(forward_m, self.side_range_m)
            cv2.line(canvas, start, end, (52, 57, 63), 1, cv2.LINE_AA)

        for right_m in range(
            -int(self.side_range_m),
            int(self.side_range_m) + 1,
            10,
        ):
            start = self.to_pixel(-self.rear_range_m, right_m)
            end = self.to_pixel(self.forward_range_m, right_m)
            cv2.line(canvas, start, end, (52, 57, 63), 1, cv2.LINE_AA)

        origin = self.to_pixel(0.0, 0.0)
        forward_end = self.to_pixel(12.0, 0.0)
        right_end = self.to_pixel(0.0, 8.0)
        cv2.arrowedLine(canvas, origin, forward_end, (80, 210, 80), 2)
        cv2.arrowedLine(canvas, origin, right_end, (255, 170, 70), 2)
        cv2.putText(
            canvas,
            "X ileri",
            forward_end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 210, 80),
            1,
        )
        cv2.putText(
            canvas,
            "Y sag",
            right_end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 170, 70),
            1,
        )

    def draw_route(self, canvas, route_points):
        previous = None
        for forward_m, right_m in route_points:
            pixel = self.to_pixel(forward_m, right_m)
            if not self.inside_image(pixel[0], pixel[1]):
                previous = None
                continue
            if previous is not None:
                cv2.line(canvas, previous, pixel, (60, 180, 255), 2, cv2.LINE_AA)
            previous = pixel

    def draw_lidar(self, canvas, points):
        if points is None:
            return
        for point in points:
            pixel_x, pixel_y = self.to_pixel(point[0], point[1])
            if self.inside_image(pixel_x, pixel_y):
                canvas[pixel_y, pixel_x] = (105, 110, 115)

    def draw_radars(self, canvas, radar_points):
        for point in radar_points:
            pixel_x, pixel_y = self.to_pixel(point["x_m"], point["y_m"])
            if self.inside_image(pixel_x, pixel_y):
                cv2.circle(canvas, (pixel_x, pixel_y), 2, (255, 80, 190), -1)

    def draw_detections(self, canvas, detections):
        for detection in detections:
            pixel_x, pixel_y = self.to_pixel(
                detection["x_m"],
                detection["y_m"],
            )
            if not self.inside_image(pixel_x, pixel_y):
                continue
            cv2.circle(canvas, (pixel_x, pixel_y), 7, (40, 230, 80), 2)
            short_name = detection["camera_name"].replace("camera_", "")
            cv2.putText(
                canvas,
                short_name,
                (pixel_x + 8, pixel_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (90, 245, 120),
                1,
                cv2.LINE_AA,
            )

    def draw_ego(self, canvas, geometry):
        half_length = float(geometry.get("half_length_m", 2.35))
        half_width = float(geometry.get("half_width_m", 0.95))
        front_left = self.to_pixel(half_length, -half_width)
        rear_right = self.to_pixel(-half_length, half_width)
        cv2.rectangle(canvas, front_left, rear_right, (230, 230, 235), 2)

    def draw_header(self, canvas, scene, current_frame_id):
        active = int(scene.get("active_sensor_count", 0))
        total = int(scene.get("total_sensor_count", 0))
        detections = len(scene.get("detections", []))
        frame_text = "-" if current_frame_id is None else str(current_frame_id)
        text = (
            f"BEV | Frame {frame_text} | Sensors {active}/{total} | "
            f"Objects {detections}"
        )
        cv2.rectangle(canvas, (0, 0), (self.width, 32), (15, 17, 20), -1)
        cv2.putText(
            canvas,
            text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

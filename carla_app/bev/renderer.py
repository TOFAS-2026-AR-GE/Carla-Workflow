"""IPM, occupancy ve izlenen nesneleri tek kuş bakışı görüntüde çizer."""

import cv2
import numpy as np

from carla_app.bev.coordinate import MetricGrid


class BevRenderer:
    """Ego aracı altta, ileri yön yukarı olacak şekilde gerçek BEV çizer."""

    def __init__(
        self,
        width=800,
        height=600,
        forward_range_m=60.0,
        rear_range_m=20.0,
        side_range_m=30.0,
        grid=None,
    ):
        self.grid = grid or MetricGrid(
            width=width,
            height=height,
            forward_range_m=forward_range_m,
            rear_range_m=rear_range_m,
            side_range_m=side_range_m,
        )
        self.width = self.grid.width
        self.height = self.grid.height
        self.forward_range_m = self.grid.forward_range_m
        self.rear_range_m = self.grid.rear_range_m
        self.side_range_m = self.grid.side_range_m

    def render(self, scene, current_frame_id=None):
        canvas = self.build_background(scene.get("ipm_image"))
        self.draw_occupancy(canvas, scene.get("occupancy"))
        self.draw_grid(canvas)
        self.draw_route(canvas, scene.get("route_points", []))
        self.draw_lidar(canvas, scene.get("lidar_points"), scene.get("ground_z_m"))
        self.draw_radars(canvas, scene.get("radar_points", []))
        self.draw_tracks(canvas, scene.get("tracks", []))
        self.draw_ego(canvas, scene.get("vehicle_geometry", {}))
        self.draw_legend(canvas)
        self.draw_header(canvas, scene, current_frame_id)
        return canvas

    def build_background(self, ipm_image):
        if ipm_image is None:
            shape = (self.height, self.width, 3)
            return np.full(shape, (24, 27, 31), dtype=np.uint8)

        image = np.asarray(ipm_image)
        if image.shape[:2] != (self.height, self.width):
            image = cv2.resize(
                image,
                (self.width, self.height),
                interpolation=cv2.INTER_AREA,
            )
        # Yol dokusu görünür kalırken ölçüm katmanlarının okunmasını sağlar.
        return cv2.addWeighted(image, 0.58, np.zeros_like(image), 0.42, 0.0)

    def to_pixel(self, forward_m, right_m):
        return self.grid.to_pixel(forward_m, right_m)

    def inside_image(self, pixel_x, pixel_y):
        return 0 <= pixel_x < self.width and 0 <= pixel_y < self.height

    def draw_occupancy(self, canvas, occupancy):
        if not occupancy:
            return

        occupied = occupancy["occupied"].astype(np.uint8)
        free = occupancy["free"].astype(np.uint8)
        occupied = cv2.resize(
            occupied,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        free = cv2.resize(
            free,
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        overlay = canvas.copy()
        overlay[free] = (45, 105, 65)
        overlay[occupied] = (40, 55, 230)
        cv2.addWeighted(overlay, 0.34, canvas, 0.66, 0.0, dst=canvas)

    def draw_grid(self, canvas):
        for forward_m in range(
            -int(self.rear_range_m),
            int(self.forward_range_m) + 1,
            10,
        ):
            start = self.to_pixel(forward_m, -self.side_range_m)
            end = self.to_pixel(forward_m, self.side_range_m)
            cv2.line(canvas, start, end, (105, 110, 115), 1, cv2.LINE_AA)

        for right_m in range(
            -int(self.side_range_m),
            int(self.side_range_m) + 1,
            10,
        ):
            start = self.to_pixel(-self.rear_range_m, right_m)
            end = self.to_pixel(self.forward_range_m, right_m)
            cv2.line(canvas, start, end, (105, 110, 115), 1, cv2.LINE_AA)

        origin = self.to_pixel(0.0, 0.0)
        forward_end = self.to_pixel(10.0, 0.0)
        right_end = self.to_pixel(0.0, 7.0)
        cv2.arrowedLine(canvas, origin, forward_end, (80, 230, 90), 2)
        cv2.arrowedLine(canvas, origin, right_end, (255, 180, 70), 2)
        cv2.putText(
            canvas,
            "X ileri",
            forward_end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 230, 90),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Y sag",
            right_end,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 180, 70),
            1,
            cv2.LINE_AA,
        )

    def draw_route(self, canvas, route_points):
        previous = None
        for forward_m, right_m in route_points:
            pixel = self.to_pixel(forward_m, right_m)
            if not self.inside_image(pixel[0], pixel[1]):
                previous = None
                continue
            if previous is not None:
                cv2.line(canvas, previous, pixel, (40, 180, 255), 2, cv2.LINE_AA)
            previous = pixel

    def draw_lidar(self, canvas, points, ground_z_m):
        if points is None:
            return
        ground_z_m = 0.0 if ground_z_m is None else float(ground_z_m)
        for point in points:
            if len(point) >= 3 and float(point[2]) <= ground_z_m + 0.18:
                continue
            pixel_x, pixel_y = self.to_pixel(point[0], point[1])
            if self.inside_image(pixel_x, pixel_y):
                canvas[pixel_y, pixel_x] = (215, 215, 215)

    def draw_radars(self, canvas, radar_points):
        for point in radar_points:
            if point.get("ground_return", False):
                continue
            pixel_x, pixel_y = self.to_pixel(point["x_m"], point["y_m"])
            if self.inside_image(pixel_x, pixel_y):
                cv2.circle(canvas, (pixel_x, pixel_y), 2, (255, 70, 210), -1)

    def draw_tracks(self, canvas, tracks):
        for track in tracks:
            pixel_x, pixel_y = self.to_pixel(track["x_m"], track["y_m"])
            if not self.inside_image(pixel_x, pixel_y):
                continue

            confirmed = bool(track.get("confirmed", False))
            color = (60, 245, 100) if confirmed else (40, 210, 255)
            half_length = max(1.0, float(track.get("length_m", 4.5)) / 2.0)
            half_width = max(0.5, float(track.get("width_m", 1.9)) / 2.0)
            front_left = self.to_pixel(
                float(track["x_m"]) + half_length,
                float(track["y_m"]) - half_width,
            )
            rear_right = self.to_pixel(
                float(track["x_m"]) - half_length,
                float(track["y_m"]) + half_width,
            )
            cv2.rectangle(canvas, front_left, rear_right, color, 2)

            uncertainty_m = min(6.0, float(track.get("uncertainty_m", 1.0)))
            uncertainty_edge = self.to_pixel(
                float(track["x_m"]),
                float(track["y_m"]) + uncertainty_m,
            )
            uncertainty_pixels = max(2, abs(uncertainty_edge[0] - pixel_x))
            cv2.circle(canvas, (pixel_x, pixel_y), uncertainty_pixels, color, 1)

            sources = "+".join(track.get("sources", []))
            label = (
                f"#{track['track_id']} {track.get('class_name', 'object')} "
                f"{track.get('confidence', 0.0):.2f} {sources}"
            )
            cv2.putText(
                canvas,
                label,
                (pixel_x + 7, pixel_y - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    def draw_ego(self, canvas, geometry):
        half_length = float(geometry.get("half_length_m", 2.35))
        half_width = float(geometry.get("half_width_m", 0.95))
        front_left = self.to_pixel(half_length, -half_width)
        rear_right = self.to_pixel(-half_length, half_width)
        cv2.rectangle(canvas, front_left, rear_right, (245, 245, 245), 2)

    def draw_legend(self, canvas):
        cv2.rectangle(
            canvas,
            (8, self.height - 50),
            (285, self.height - 8),
            (18, 20, 23),
            -1,
        )
        cv2.putText(
            canvas,
            "Gri: LiDAR  Pembe: Radar  Kirmizi: Dolu",
            (15, self.height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Yesil: Onayli takip  Sari: Yeni olcum",
            (15, self.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    def draw_header(self, canvas, scene, current_frame_id):
        active = int(scene.get("active_sensor_count", 0))
        total = int(scene.get("total_sensor_count", 0))
        tracks = len(scene.get("tracks", []))
        cameras = len(scene.get("ipm_cameras", []))
        frame_text = "-" if current_frame_id is None else str(current_frame_id)
        text = (
            f"BEV IPM | Frame {frame_text} | Sensors {active}/{total} | "
            f"Cameras {cameras}/7 | Tracks {tracks}"
        )
        cv2.rectangle(canvas, (0, 0), (self.width, 32), (15, 17, 20), -1)
        cv2.putText(
            canvas,
            text,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

"""BEV verisini debug veya sade sürüş ekranı olarak çizer."""

import math

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

    def render(self, scene, current_frame_id=None, display_mode="driving"):
        """Seçilen görsel modu aynı BEV sahne verisinden üretir."""
        if str(display_mode).lower() == "debug":
            return self.render_debug(scene, current_frame_id)
        return self.render_driving(scene, current_frame_id)

    def render_debug(self, scene, current_frame_id=None):
        """Ham IPM ve sensör ayrıntılarını gösteren mühendislik ekranı."""
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

    def render_driving(self, scene, current_frame_id=None):
        """Sürücünün kolay okuyacağı temiz ve özgün BEV ekranı."""
        canvas = self.build_driving_background(scene.get("ipm_image"))
        route_points = scene.get("route_points", [])

        # Yol ve occupancy altta, hareketli nesneler ile ego araç üstte kalır.
        self.draw_driving_corridor(
            canvas,
            route_points,
            scene.get("lane_width_m", 3.5),
        )
        self.draw_driving_occupancy(canvas, scene.get("occupancy"))
        self.draw_distance_guides(canvas)
        self.draw_planned_path(canvas, route_points)
        self.draw_driving_sensor_points(
            canvas,
            scene.get("lidar_points"),
            scene.get("ground_z_m"),
            scene.get("radar_points", []),
        )
        self.draw_driving_tracks(canvas, scene.get("tracks", []))
        self.draw_driving_ego(canvas, scene.get("vehicle_geometry", {}))
        self.draw_driving_header(canvas, scene, current_frame_id)
        self.draw_driving_footer(canvas, scene)
        return canvas

    def build_driving_background(self, ipm_image):
        """Koyu gradyan ve çok hafif IPM dokusundan sakin arka plan kurar."""
        top_color = np.array((16, 18, 21), dtype=np.float32)
        bottom_color = np.array((31, 33, 36), dtype=np.float32)
        ratios = np.linspace(0.0, 1.0, self.height, dtype=np.float32)
        rows = top_color + (bottom_color - top_color) * ratios[:, None]
        canvas = np.repeat(rows[:, None, :], self.width, axis=1)

        if ipm_image is not None:
            image = np.asarray(ipm_image)
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(
                    image,
                    (self.width, self.height),
                    interpolation=cv2.INTER_AREA,
                )
            # Renkli kamera mozaiği dikkati dağıtmasın; yalnızca yol
            # dokusunu belli eden düşük kontrastlı gri bilgi korunur.
            gray = np.mean(image[:, :, :3], axis=2, keepdims=True)
            texture = np.repeat(gray, 3, axis=2)
            canvas = 0.88 * canvas + 0.12 * texture

        return np.clip(canvas, 0, 255).astype(np.uint8)

    def draw_driving_corridor(self, canvas, route_points, lane_width_m):
        """Referans rota etrafında tek şeritlik yumuşak yol koridoru çizer."""
        points = self.valid_route_points(route_points)
        if len(points) < 2:
            return

        lane_half_width = max(1.5, min(3.0, float(lane_width_m) / 2.0))
        left_edge = []
        right_edge = []
        for index, point in enumerate(points):
            before = points[max(0, index - 1)]
            after = points[min(len(points) - 1, index + 1)]
            tangent_forward = after[0] - before[0]
            tangent_right = after[1] - before[1]
            length = math.hypot(tangent_forward, tangent_right)
            if length <= 1e-6:
                normal_forward, normal_right = 0.0, 1.0
            else:
                normal_forward = -tangent_right / length
                normal_right = tangent_forward / length
            left_edge.append(
                (
                    point[0] - normal_forward * lane_half_width,
                    point[1] - normal_right * lane_half_width,
                )
            )
            right_edge.append(
                (
                    point[0] + normal_forward * lane_half_width,
                    point[1] + normal_right * lane_half_width,
                )
            )

        polygon_metric = left_edge + list(reversed(right_edge))
        polygon = np.array(
            [self.to_pixel(forward, right) for forward, right in polygon_metric],
            dtype=np.int32,
        )
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [polygon], (47, 49, 52), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0.0, dst=canvas)

        left_pixels = np.array(
            [self.to_pixel(forward, right) for forward, right in left_edge],
            dtype=np.int32,
        )
        right_pixels = np.array(
            [self.to_pixel(forward, right) for forward, right in right_edge],
            dtype=np.int32,
        )
        edge_color = (100, 103, 108)
        cv2.polylines(canvas, [left_pixels], False, edge_color, 1, cv2.LINE_AA)
        cv2.polylines(canvas, [right_pixels], False, edge_color, 1, cv2.LINE_AA)

    def draw_driving_occupancy(self, canvas, occupancy):
        """Boş alanı sakin, dolu alanı belirgin bir renk katmanıyla gösterir."""
        if not occupancy:
            return
        occupied = cv2.resize(
            occupancy["occupied"].astype(np.uint8),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        free = cv2.resize(
            occupancy["free"].astype(np.uint8),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        overlay = canvas.copy()
        overlay[free] = (48, 64, 59)
        overlay[occupied] = (54, 92, 224)
        cv2.addWeighted(overlay, 0.30, canvas, 0.70, 0.0, dst=canvas)

    def draw_distance_guides(self, canvas):
        """Her on metrede sade mesafe çizgileri ve merkez kılavuzu çizer."""
        guide_color = (61, 64, 68)
        label_color = (126, 130, 135)
        for forward_m in range(10, int(self.forward_range_m) + 1, 10):
            start = self.to_pixel(forward_m, -self.side_range_m)
            end = self.to_pixel(forward_m, self.side_range_m)
            cv2.line(canvas, start, end, guide_color, 1, cv2.LINE_AA)
            label = self.to_pixel(forward_m, -self.side_range_m + 1.0)
            cv2.putText(
                canvas,
                f"{forward_m} m",
                (label[0] + 5, label[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                label_color,
                1,
                cv2.LINE_AA,
            )

        # Kesikli merkez çizgisi aracın ileri eksenini sakin biçimde anlatır.
        for forward_m in range(0, int(self.forward_range_m), 4):
            start = self.to_pixel(forward_m, 0.0)
            end = self.to_pixel(min(self.forward_range_m, forward_m + 2.0), 0.0)
            cv2.line(canvas, start, end, (76, 79, 84), 1, cv2.LINE_AA)

    def draw_planned_path(self, canvas, route_points):
        """Planlanan rotayı parlak mavi ve okunabilir bir şerit olarak çizer."""
        points = self.valid_route_points(route_points)
        if len(points) < 2:
            return
        pixels = np.array(
            [self.to_pixel(forward, right) for forward, right in points],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [pixels], False, (96, 63, 20), 12, cv2.LINE_AA)
        cv2.polylines(canvas, [pixels], False, (255, 154, 48), 5, cv2.LINE_AA)
        cv2.polylines(canvas, [pixels], False, (255, 220, 160), 1, cv2.LINE_AA)

    def valid_route_points(self, route_points):
        """Ekran çevresindeki sonlu rota noktalarını seçer."""
        valid = []
        for point in route_points:
            try:
                forward_m = float(point[0])
                right_m = float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not math.isfinite(forward_m) or not math.isfinite(right_m):
                continue
            if not -self.rear_range_m - 4.0 <= forward_m <= self.forward_range_m + 4.0:
                continue
            if abs(right_m) > self.side_range_m + 5.0:
                continue
            valid.append((forward_m, right_m))
        return valid

    def draw_driving_sensor_points(
        self,
        canvas,
        lidar_points,
        ground_z_m,
        radar_points,
    ):
        """Ham noktaları sürüş ekranında geri planda ve düşük kontrastta tutar."""
        if lidar_points is not None:
            points = np.asarray(lidar_points)
            if points.ndim == 2 and points.shape[1] >= 3 and len(points):
                ground = 0.0 if ground_z_m is None else float(ground_z_m)
                points = points[points[:, 2] > ground + 0.18]
                if len(points) > 2500:
                    step = int(math.ceil(len(points) / 2500.0))
                    points = points[::step]
                for point in points:
                    pixel_x, pixel_y = self.to_pixel(point[0], point[1])
                    if self.inside_image(pixel_x, pixel_y):
                        canvas[pixel_y, pixel_x] = (118, 125, 130)

        for point in radar_points[::2]:
            if point.get("ground_return", False):
                continue
            pixel = self.to_pixel(point["x_m"], point["y_m"])
            if self.inside_image(pixel[0], pixel[1]):
                cv2.circle(canvas, pixel, 1, (198, 115, 191), -1)

    def draw_driving_tracks(self, canvas, tracks):
        """Takipleri hareket yönü, hız ve güven durumuyla stilize eder."""
        for track in tracks:
            center = self.to_pixel(track["x_m"], track["y_m"])
            if not self.inside_image(center[0], center[1]):
                continue

            velocity_forward = float(track.get("velocity_x_mps", 0.0))
            velocity_right = float(track.get("velocity_y_mps", 0.0))
            object_speed = math.hypot(velocity_forward, velocity_right)
            heading = math.atan2(velocity_right, velocity_forward) if object_speed > 0.5 else 0.0
            corners = self.metric_box_corners(track, heading)
            polygon = np.array(
                [self.to_pixel(forward, right) for forward, right in corners],
                dtype=np.int32,
            )

            confirmed = bool(track.get("confirmed", False))
            body_color = (190, 194, 198) if confirmed else (92, 145, 190)
            outline_color = (238, 241, 244) if confirmed else (96, 190, 255)
            shadow = polygon + np.array([3, 4], dtype=np.int32)
            cv2.fillPoly(canvas, [shadow], (15, 16, 18), lineType=cv2.LINE_AA)
            cv2.fillPoly(canvas, [polygon], body_color, lineType=cv2.LINE_AA)
            cv2.polylines(canvas, [polygon], True, outline_color, 1, cv2.LINE_AA)

            # Ön cam ve tavan, kutuyu ham bbox yerine araç siluetine yaklaştırır.
            roof_radius = max(2, int(0.18 * max(8, abs(polygon[0][1] - polygon[2][1]))))
            cv2.circle(canvas, center, roof_radius, (101, 108, 114), -1, cv2.LINE_AA)
            self.draw_track_motion(canvas, track, object_speed)
            self.draw_track_label(canvas, track, center, object_speed)

    def metric_box_corners(self, track, heading):
        """Nesnenin metre cinsinden dört köşesini hareket yönüne çevirir."""
        center_forward = float(track["x_m"])
        center_right = float(track["y_m"])
        half_length = max(1.0, float(track.get("length_m", 4.5)) / 2.0)
        half_width = max(0.5, float(track.get("width_m", 1.9)) / 2.0)
        axis_forward = (math.cos(heading), math.sin(heading))
        axis_right = (-math.sin(heading), math.cos(heading))
        corners = []
        for length_sign, width_sign in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
            corners.append(
                (
                    center_forward
                    + length_sign * half_length * axis_forward[0]
                    + width_sign * half_width * axis_right[0],
                    center_right
                    + length_sign * half_length * axis_forward[1]
                    + width_sign * half_width * axis_right[1],
                )
            )
        return corners

    def draw_track_motion(self, canvas, track, object_speed):
        """Hareketli nesnenin yaklaşık 1,2 saniyelik yön vektörünü çizer."""
        if object_speed < 0.6:
            return
        start = self.to_pixel(track["x_m"], track["y_m"])
        end = self.to_pixel(
            float(track["x_m"]) + float(track.get("velocity_x_mps", 0.0)) * 1.2,
            float(track["y_m"]) + float(track.get("velocity_y_mps", 0.0)) * 1.2,
        )
        cv2.arrowedLine(canvas, start, end, (255, 184, 80), 2, cv2.LINE_AA, tipLength=0.22)

    def draw_track_label(self, canvas, track, center, object_speed):
        """Onaylı takip için kısa kimlik ve hız etiketi yazar."""
        if not track.get("confirmed", False):
            return
        class_name = str(track.get("class_name", "object")).upper()[:10]
        label = f"#{int(track.get('track_id', 0))} {class_name} {object_speed * 3.6:.0f} km/h"
        text_x = min(self.width - 175, center[0] + 9)
        text_y = max(70, center[1] - 10)
        cv2.rectangle(
            canvas,
            (text_x - 4, text_y - 14),
            (text_x + 164, text_y + 5),
            (24, 26, 29),
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (223, 226, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_driving_ego(self, canvas, geometry):
        """Ego aracı gölge, gövde, cam ve tekerlerle sade biçimde çizer."""
        half_length = float(geometry.get("half_length_m", 2.35))
        half_width = float(geometry.get("half_width_m", 0.95))
        corners = np.array(
            [
                self.to_pixel(half_length, -half_width),
                self.to_pixel(half_length, half_width),
                self.to_pixel(-half_length, half_width),
                self.to_pixel(-half_length, -half_width),
            ],
            dtype=np.int32,
        )
        shadow = corners + np.array([4, 5], dtype=np.int32)
        cv2.fillPoly(canvas, [shadow], (10, 11, 13), lineType=cv2.LINE_AA)
        cv2.fillPoly(canvas, [corners], (230, 233, 236), lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [corners], True, (255, 255, 255), 2, cv2.LINE_AA)

        windshield_front = self.to_pixel(0.85, -0.68 * half_width)
        windshield_rear = self.to_pixel(-0.55, 0.68 * half_width)
        cv2.rectangle(canvas, windshield_front, windshield_rear, (70, 79, 87), -1)
        nose = self.to_pixel(half_length - 0.20, 0.0)
        center = self.to_pixel(0.0, 0.0)
        cv2.line(canvas, center, nose, (190, 197, 203), 1, cv2.LINE_AA)

        # Dört koyu teker, aracın yönünü uzaktan da okunur yapar.
        for forward_sign in (-0.62, 0.62):
            for right_sign in (-1.0, 1.0):
                wheel = self.to_pixel(
                    forward_sign * half_length,
                    right_sign * (half_width + 0.08),
                )
                cv2.circle(canvas, wheel, 3, (25, 27, 30), -1, cv2.LINE_AA)

    def draw_driving_header(self, canvas, scene, current_frame_id):
        """Hız, hedef, sürüş modu ve sensör sağlığını özetler."""
        driving = scene.get("driving_state", {}) or {}
        speed_kmh = float(scene.get("ego_speed_mps", 0.0)) * 3.6
        target_kmh = float(driving.get("target_speed_mps", 0.0)) * 3.6
        mode = str(driving.get("mode", "BEKLENIYOR")).replace("_", " ")
        active = int(scene.get("active_sensor_count", 0))
        total = int(scene.get("total_sensor_count", 0))

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.width, 56), (15, 17, 20), -1)
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0.0, dst=canvas)
        cv2.putText(
            canvas,
            f"{speed_kmh:3.0f}",
            (16, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.02,
            (248, 249, 250),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "km/h",
            (78, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (155, 160, 166),
            1,
            cv2.LINE_AA,
        )
        cv2.line(canvas, (126, 12), (126, 44), (67, 70, 74), 1)
        cv2.putText(
            canvas,
            f"HEDEF {target_kmh:.0f}",
            (145, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (177, 182, 187),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            mode[:28],
            (145, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 178, 70),
            1,
            cv2.LINE_AA,
        )

        sensor_ok = total > 0 and active >= total
        sensor_color = (92, 205, 125) if sensor_ok else (75, 180, 245)
        sensor_text = f"SENSOR {active}/{total}"
        # Sağ üstteki Sürüş/Debug switch'i için boş alan bırakırız.
        text_x = max(330, self.width - 310)
        cv2.circle(canvas, (text_x, 27), 5, sensor_color, -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            sensor_text,
            (text_x + 12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (207, 211, 215),
            1,
            cv2.LINE_AA,
        )

    def draw_driving_footer(self, canvas, scene):
        """Alt köşede nesne ve occupancy renklerini kısa biçimde açıklar."""
        tracks = len(scene.get("tracks", []))
        text = f"360 SURROUND  |  {tracks} TAKIP  |  MAVI ROTA  |  TURUNCU DOLU ALAN"
        cv2.rectangle(canvas, (0, self.height - 27), (self.width, self.height), (17, 19, 22), -1)
        cv2.putText(
            canvas,
            text,
            (14, self.height - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            (157, 162, 168),
            1,
            cv2.LINE_AA,
        )

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

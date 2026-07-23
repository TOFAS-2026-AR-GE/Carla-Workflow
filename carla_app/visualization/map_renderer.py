"""CARLA yol ağını modern ve etkileşimli navigasyon paneline çizer."""

import math

import cv2
import numpy as np


class MapRenderer:
    """CARLA haritasını dünya koordinatından OpenCV pikseline dönüştürür."""

    def __init__(self, carla_map, width=540, height=760, waypoint_spacing=3.0):
        self.map = carla_map
        self.width = int(width)
        self.height = int(height)
        self.map_name = str(getattr(carla_map, "name", "CARLA")).split("/")[-1]
        self.padding = 26
        self.header_height = 86
        self.footer_height = 126
        self.button_rect = None
        self.cancel_rect = None
        self.map_rect = None
        self.cached_route_id = None
        self.cached_route_points = None

        self.waypoints = list(
            carla_map.generate_waypoints(float(waypoint_spacing))
        )
        self.road_lines = self._build_road_lines(self.waypoints)
        self.min_x, self.max_x, self.min_y, self.max_y = self._world_bounds()
        self.base_map = self._draw_base_map()

    def render(self, navigation, vehicle_location, vehicle_yaw, speed_kmh):
        """Navigasyon durumunu ve canlı ego konumunu tek panelde döndürür."""
        panel = self.base_map.copy()
        navigation = navigation or {}

        self._draw_route(panel, navigation.get("route", []))
        self._draw_marker(
            panel,
            navigation.get("start_location"),
            (205, 145, 72),
            "BASLANGIC",
        )
        self._draw_marker(
            panel,
            navigation.get("destination"),
            (56, 72, 230),
            "HEDEF",
        )
        self._draw_marker(
            panel,
            navigation.get("pending_destination"),
            (64, 190, 255),
            "SECILI",
        )
        self._draw_ego(panel, vehicle_location, vehicle_yaw)
        self._draw_status(panel, navigation, speed_kmh)
        return panel

    def screen_to_world(self, x, y):
        """Panel içindeki pikseli CARLA dünya koordinatına dönüştürür."""
        if self.map_rect is None:
            return None
        x1, y1, x2, y2 = self.map_rect
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return None

        scale_x, scale_y, offset_x, offset_y = self._transform()
        world_x = (float(x) - offset_x) / scale_x + self.min_x
        world_y = self.max_y - (float(y) - offset_y) / scale_y
        return world_x, world_y

    def is_confirm_button(self, x, y):
        return self._inside(self.button_rect, x, y)

    def is_cancel_button(self, x, y):
        return self._inside(self.cancel_rect, x, y)

    def _draw_base_map(self):
        panel = np.full(
            (self.height, self.width, 3),
            (18, 22, 28),
            dtype=np.uint8,
        )
        cv2.rectangle(
            panel,
            (0, 0),
            (self.width, self.header_height),
            (24, 29, 37),
            -1,
        )
        cv2.putText(
            panel,
            "NAVIGASYON",
            (24, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (112, 204, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            self.map_name.upper(),
            (24, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (244, 247, 250),
            2,
            cv2.LINE_AA,
        )

        map_top = self.header_height + 10
        map_bottom = self.height - self.footer_height - 10
        self.map_rect = (
            self.padding,
            map_top,
            self.width - self.padding,
            map_bottom,
        )
        x1, y1, x2, y2 = self.map_rect
        cv2.rectangle(panel, (x1, y1), (x2, y2), (28, 34, 42), -1)

        for line in self.road_lines:
            points = np.array(
                [self.world_to_screen(item) for item in line],
                dtype=np.int32,
            )
            if len(points) < 2:
                continue
            cv2.polylines(
                panel,
                [points],
                False,
                (50, 58, 68),
                7,
                cv2.LINE_AA,
            )
            cv2.polylines(
                panel,
                [points],
                False,
                (95, 104, 115),
                1,
                cv2.LINE_AA,
            )
        return panel

    def _draw_route(self, panel, route):
        if len(route) < 2:
            return
        route_id = id(route)
        if route_id != self.cached_route_id:
            self.cached_route_points = np.array(
                [self.world_to_screen(location) for location in route],
                dtype=np.int32,
            )
            self.cached_route_id = route_id
        points = self.cached_route_points
        cv2.polylines(
            panel,
            [points],
            False,
            (24, 31, 45),
            8,
            cv2.LINE_AA,
        )
        cv2.polylines(
            panel,
            [points],
            False,
            (48, 68, 235),
            4,
            cv2.LINE_AA,
        )

    def _draw_ego(self, panel, location, yaw):
        if location is None:
            return
        x, y = self.world_to_screen(location)
        angle = math.radians(float(yaw))
        direction = np.array([math.cos(angle), -math.sin(angle)])
        side = np.array([-direction[1], direction[0]])
        tip = np.array([x, y]) + direction * 13
        left = np.array([x, y]) - direction * 8 + side * 7
        right = np.array([x, y]) - direction * 8 - side * 7
        polygon = np.array([tip, left, right], dtype=np.int32)
        cv2.circle(panel, (x, y), 15, (22, 28, 36), -1, cv2.LINE_AA)
        cv2.fillConvexPoly(panel, polygon, (255, 209, 92), cv2.LINE_AA)
        cv2.polylines(
            panel,
            [polygon],
            True,
            (255, 244, 215),
            1,
            cv2.LINE_AA,
        )

    def _draw_marker(self, panel, location, color, label):
        if location is None:
            return
        x, y = self.world_to_screen(location)
        cv2.circle(panel, (x, y), 11, (18, 22, 28), -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(panel, (x, y), 3, (250, 250, 250), -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            label,
            (x + 12, y - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 239, 244),
            1,
            cv2.LINE_AA,
        )

    def _draw_status(self, panel, navigation, speed_kmh):
        top = self.height - self.footer_height
        cv2.rectangle(
            panel,
            (0, top),
            (self.width, self.height),
            (24, 29, 37),
            -1,
        )
        status = navigation.get("status", "WAITING")
        status_text = {
            "WAITING": "HEDEF BEKLENIYOR",
            "PENDING": "ONAY BEKLENIYOR",
            "DRIVING": "ROTA AKTIF",
            "ARRIVED": "HEDEFE VARILDI",
            "ERROR": "ROTA HATASI",
        }.get(status, status)
        status_color = {
            "WAITING": (145, 154, 166),
            "PENDING": (64, 190, 255),
            "DRIVING": (100, 220, 145),
            "ARRIVED": (100, 220, 145),
            "ERROR": (92, 92, 245),
        }.get(status, (220, 220, 220))

        cv2.putText(
            panel,
            status_text,
            (24, top + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            1,
            cv2.LINE_AA,
        )

        remaining = navigation.get("remaining_distance_m")
        remaining_text = "--" if remaining is None else f"{remaining:.0f} m"
        info = f"{float(speed_kmh):.0f} km/h   |   Kalan {remaining_text}"
        cv2.putText(
            panel,
            info,
            (24, top + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (224, 229, 235),
            1,
            cv2.LINE_AA,
        )

        button_y1 = top + 70
        button_y2 = self.height - 16
        self.cancel_rect = (24, button_y1, 122, button_y2)
        self.button_rect = (134, button_y1, self.width - 24, button_y2)
        self._button(
            panel,
            self.cancel_rect,
            "IPTAL",
            (47, 53, 63),
            enabled=navigation.get("pending_destination") is not None,
        )
        self._button(
            panel,
            self.button_rect,
            "ROTAYI ONAYLA",
            (38, 73, 219),
            enabled=navigation.get("pending_destination") is not None,
        )

    @staticmethod
    def _button(panel, rect, text, color, enabled):
        x1, y1, x2, y2 = rect
        fill = color if enabled else (43, 48, 57)
        text_color = (250, 251, 252) if enabled else (105, 112, 122)
        cv2.rectangle(panel, (x1, y1), (x2, y2), fill, -1)
        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            1,
        )[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(
            panel,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            text_color,
            1,
            cv2.LINE_AA,
        )

    def world_to_screen(self, location):
        scale_x, scale_y, offset_x, offset_y = self._transform()
        x = offset_x + (float(location.x) - self.min_x) * scale_x
        y = offset_y + (self.max_y - float(location.y)) * scale_y
        return int(round(x)), int(round(y))

    def _transform(self):
        x1, y1, x2, y2 = self.map_rect
        world_width = max(1.0, self.max_x - self.min_x)
        world_height = max(1.0, self.max_y - self.min_y)
        available_width = max(1.0, x2 - x1)
        available_height = max(1.0, y2 - y1)
        scale = min(
            available_width / world_width,
            available_height / world_height,
        )
        drawn_width = world_width * scale
        drawn_height = world_height * scale
        offset_x = x1 + (available_width - drawn_width) * 0.5
        offset_y = y1 + (available_height - drawn_height) * 0.5
        return scale, scale, offset_x, offset_y

    def _world_bounds(self):
        locations = [waypoint.transform.location for waypoint in self.waypoints]
        if not locations:
            return -100.0, 100.0, -100.0, 100.0
        margin = 12.0
        return (
            min(item.x for item in locations) - margin,
            max(item.x for item in locations) + margin,
            min(item.y for item in locations) - margin,
            max(item.y for item in locations) + margin,
        )

    @staticmethod
    def _build_road_lines(waypoints):
        grouped = {}
        for waypoint in waypoints:
            key = (
                int(waypoint.road_id),
                int(waypoint.section_id),
                int(waypoint.lane_id),
            )
            grouped.setdefault(key, []).append(waypoint)

        lines = []
        for lane_waypoints in grouped.values():
            lane_waypoints.sort(key=lambda item: float(item.s))
            current_line = []
            previous = None
            for waypoint in lane_waypoints:
                location = waypoint.transform.location
                if previous is not None:
                    gap = math.hypot(
                        location.x - previous.x,
                        location.y - previous.y,
                    )
                    if gap > 12.0 and len(current_line) >= 2:
                        lines.append(current_line)
                        current_line = []
                current_line.append(location)
                previous = location
            if len(current_line) >= 2:
                lines.append(current_line)
        return lines

    @staticmethod
    def _inside(rect, x, y):
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

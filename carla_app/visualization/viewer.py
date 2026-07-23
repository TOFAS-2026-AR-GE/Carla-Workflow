"""Kamera görüntüsünü ve algılama kutularını OpenCV penceresinde gösterir."""

import cv2
import numpy as np

from carla_app.visualization.navigation_panel import NavigationPanel


class PerceptionViewer:
    """Algılama sonuçlarını gösterir ve kullanıcı çıkışını izler."""

    def __init__(
        self,
        carla_map=None,
        navigation=None,
        window_name="CARLA Akilli Surus",
        dashboard_width=1580,
        dashboard_height=780,
        navigation_render_every_n_frames=2,
    ):
        self.window_name = window_name
        self.closed = False
        self.bev_mode = "driving"
        self.bev_button_rect = None
        self.dashboard_width = max(1100, int(dashboard_width))
        self.dashboard_height = max(650, int(dashboard_height))
        self.map_width = min(600, max(430, self.dashboard_width // 3))
        self.camera_width = self.dashboard_width - self.map_width - 4
        self.map_offset_x = self.camera_width + 4
        self.last_vehicle_location = None
        self.last_navigation = {}
        self.navigation_panel = None
        if carla_map is not None and navigation is not None:
            self.navigation_panel = NavigationPanel(
                carla_map,
                navigation,
                width=self.map_width,
                height=self.dashboard_height,
                render_every_n_frames=navigation_render_every_n_frames,
            )
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            window_name,
            self.dashboard_width,
            self.dashboard_height,
        )
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def show(
        self,
        result,
        fallback_image=None,
        fallback_frame_id=None,
        current_frame_id=None,
        bev_image=None,
        road_context=None,
        vehicle_state=None,
        navigation_state=None,
    ):
        if self.closed:
            return False

        if result is None:
            image = fallback_image
            result_frame_id = fallback_frame_id
            vehicles = []
            signs = []
            errors = {}
            elapsed_ms = 0.0
            lane_detection = {}
        else:
            image = result.get("image", fallback_image)
            result_frame_id = result.get("frame_id", fallback_frame_id)
            vehicles = result.get("vehicles", [])
            signs = result.get("signs", [])
            errors = result.get("errors", {})
            elapsed_ms = float(result.get("elapsed_ms", 0.0))
            lane_detection = result.get("lane_detection", {})

        if image is None and bev_image is None:
            return self._window_is_open() and self._read_key()

        if image is None:
            image = np.zeros((600, 800, 3), dtype=np.uint8)

        frame = np.ascontiguousarray(image[:, :, :3][:, :, ::-1])
        for detection in vehicles:
            self._draw(frame, detection, (0, 220, 0), "VEH")
        for detection in signs:
            self._draw(frame, detection, (0, 165, 255), "SIGN")
        self._draw_lanes(frame, lane_detection)
        road_context = road_context or {}
        for detection in road_context.get("detections", []):
            category = detection.get("category")
            if category in {"vehicle", "two_wheeler", "speed_sign"}:
                continue
            color = (0, 0, 255) if category == "traffic_light" else (255, 0, 255)
            self._draw(frame, detection, color, category.upper())

        lag = 0
        if current_frame_id is not None and result_frame_id is not None:
            lag = max(0, int(current_frame_id) - int(result_frame_id))

        header = (
            f"Frame {result_frame_id} | Vehicles {len(vehicles)} | "
            f"Signs {len(signs)} | Lag {lag} | {elapsed_ms:.1f} ms"
        )
        if lane_detection.get("available"):
            header += f" | Lanes {lane_detection.get('detected_count', 0)}"
        lead_light = road_context.get("lead_traffic_light")
        if lead_light is not None:
            header += f" | Lead {lead_light.get('color', '?')}"
        if road_context.get("speed_limit_kmh") is not None:
            header += f" | Limit {road_context['speed_limit_kmh']}"
        pedestrian_risk = road_context.get("pedestrian_risk", "NONE")
        if pedestrian_risk != "NONE":
            header += f" | Ped {pedestrian_risk}"
        header_color = (40, 40, 160) if errors else (20, 20, 20)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), header_color, -1)
        cv2.putText(
            frame,
            header,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if errors:
            error_parts = []
            for name, message in errors.items():
                error_parts.append(f"{name}: {message}")
            error_text = " | ".join(error_parts)
            cv2.rectangle(frame, (0, 32), (frame.shape[1], 62), (20, 20, 180), -1)
            cv2.putText(
                frame,
                error_text[:130],
                (10, 53),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        vehicle_state = vehicle_state or {}
        self.last_vehicle_location = vehicle_state.get("location")
        self.last_navigation = navigation_state or {}
        map_panel = None
        if self.navigation_panel is not None:
            map_panel = self.navigation_panel.render(
                self.last_navigation,
                vehicle_state,
                current_frame_id,
            )

        display_frame = self.combine_panels(frame, bev_image, map_panel)
        cv2.imshow(self.window_name, display_frame)
        self.closed = not (self._window_is_open() and self._read_key())
        return not self.closed

    def combine_panels(self, perception_frame, bev_image, map_panel=None):
        """Kamera, isteğe bağlı BEV ve navigasyonu tek büyük pencerede birleştirir."""
        if not hasattr(self, "dashboard_height"):
            if bev_image is None:
                self.bev_button_rect = None
                return perception_frame
            target_width = perception_frame.shape[1]
            target_height = perception_frame.shape[0]
            bev_panel = cv2.resize(
                bev_image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
            divider = np.full((target_height, 3, 3), 210, dtype=np.uint8)
            combined = np.hstack((perception_frame, divider, bev_panel))
            self.draw_bev_switch(combined)
            return combined

        source_height, source_width = perception_frame.shape[:2]
        interpolation = (
            cv2.INTER_AREA
            if source_width > self.camera_width
            or source_height > self.dashboard_height
            else cv2.INTER_LINEAR
        )
        camera_panel = cv2.resize(
            perception_frame,
            (self.camera_width, self.dashboard_height),
            interpolation=interpolation,
        )

        if bev_image is not None:
            inset_width = min(360, self.camera_width // 3)
            inset_height = int(inset_width * 0.70)
            inset = cv2.resize(
                bev_image,
                (inset_width, inset_height),
                interpolation=cv2.INTER_AREA,
            )
            x1 = self.camera_width - inset_width - 18
            y1 = self.dashboard_height - inset_height - 18
            cv2.rectangle(
                camera_panel,
                (x1 - 3, y1 - 27),
                (self.camera_width - 15, self.dashboard_height - 15),
                (20, 24, 30),
                -1,
            )
            camera_panel[y1 : y1 + inset_height, x1 : x1 + inset_width] = inset
            cv2.putText(
                camera_panel,
                f"BEV {self.bev_mode.upper()}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                (230, 235, 240),
                1,
                cv2.LINE_AA,
            )
            self.draw_bev_switch(camera_panel)
        else:
            self.bev_button_rect = None

        if map_panel is None:
            self.map_offset_x = 0
            return camera_panel

        divider = np.full(
            (self.dashboard_height, 4, 3),
            (8, 11, 15),
            dtype=np.uint8,
        )
        self.map_offset_x = self.camera_width + divider.shape[1]
        combined = np.hstack((camera_panel, divider, map_panel))
        return combined

    def draw_bev_switch(self, frame):
        """BEV panelinin sağ üstüne iki konumlu tıklanabilir switch çizer."""
        width = 174
        height = 30
        margin = 10
        panel_width = getattr(self, "camera_width", frame.shape[1])
        x1 = panel_width - width - margin
        y1 = 10
        x2 = x1 + width
        y2 = y1 + height
        middle = x1 + width // 2
        self.bev_button_rect = (x1, y1, x2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (31, 34, 38), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 96, 102), 1)
        mode = getattr(self, "bev_mode", "driving")
        if mode == "driving":
            cv2.rectangle(frame, (x1 + 2, y1 + 2), (middle, y2 - 2), (208, 132, 42), -1)
        else:
            cv2.rectangle(frame, (middle, y1 + 2), (x2 - 2, y2 - 2), (82, 88, 96), -1)

        cv2.putText(
            frame,
            "SURUS",
            (x1 + 17, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (248, 249, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "DEBUG",
            (middle + 17, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (248, 249, 250),
            1,
            cv2.LINE_AA,
        )

    def _mouse_callback(self, event, x, y, _flags, _parameter):
        """Harita hedefini, onayı ve BEV düğmesini aynı pencerede işler."""
        if event != cv2.EVENT_LBUTTONUP:
            return

        if self.bev_button_rect is not None:
            x1, y1, x2, y2 = self.bev_button_rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                middle = (x1 + x2) // 2
                self.bev_mode = "driving" if x < middle else "debug"
                return

        navigation_panel = getattr(self, "navigation_panel", None)
        if navigation_panel is None:
            return
        map_x = x - self.map_offset_x
        navigation_panel.handle_left_click(
            map_x,
            y,
            getattr(self, "last_vehicle_location", None),
        )

    def toggle_bev_mode(self):
        """Klavye için sürüş ve debug ekranı arasında geçiş yapar."""
        current = getattr(self, "bev_mode", "driving")
        self.bev_mode = "debug" if current == "driving" else "driving"

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass

    def _window_is_open(self):
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def _read_key(self):
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("b"), ord("B")):
            self.toggle_bev_mode()
            return True
        return key not in (27, ord("q"), ord("Q"))

    def _draw(self, frame, detection, color, prefix):
        box = detection["bbox"]
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        confidence = float(detection.get("confidence", 0.0))
        class_name = str(detection.get("class_name", "unknown"))
        label = f"{prefix} {class_name} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_lanes(frame, lane_detection):
        colors = ((0, 80, 255), (0, 220, 0), (255, 80, 0), (0, 220, 220))
        for lane in lane_detection.get("lanes", []):
            if not lane.get("detected") or len(lane.get("points", [])) < 2:
                continue
            points = np.asarray(lane["points"], dtype=np.int32).reshape(-1, 1, 2)
            lane_index = int(lane.get("lane_index", 0))
            cv2.polylines(
                frame,
                [points],
                False,
                colors[lane_index % len(colors)],
                3,
                cv2.LINE_AA,
            )

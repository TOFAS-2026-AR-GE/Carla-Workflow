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
        dashboard_width=1920,
        dashboard_height=700,
        navigation_render_every_n_frames=2,
    ):
        self.window_name = window_name
        self.closed = False
        self.bev_mode = "driving"
        self.bev_button_rect = None
        self.dashboard_width = max(1100, int(dashboard_width))
        self.dashboard_height = max(650, int(dashboard_height))
        self.map_width = min(520, max(430, self.dashboard_width // 4))
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
        road_context = road_context or {}
        self.draw_perception_overlay(
            frame,
            vehicles,
            signs,
            lane_detection,
            road_context,
        )

        lag = 0
        if current_frame_id is not None and result_frame_id is not None:
            lag = max(0, int(current_frame_id) - int(result_frame_id))

        header = (
            f"Frame {result_frame_id} | Vehicles {len(vehicles)} | "
            f"Signs {len(signs)} | Lag {lag} | {elapsed_ms:.1f} ms"
        )
        if lane_detection.get("available"):
            header += (
                f" | UFLD {lane_detection.get('detected_count', 0)}/4 "
                f"{float(lane_detection.get('elapsed_ms', 0.0)):.1f} ms"
            )
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

        camera_panel = self.resize_with_letterbox(
            perception_frame,
            self.camera_width,
            self.dashboard_height,
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

    @staticmethod
    def resize_with_letterbox(image, target_width, target_height):
        """Kamera geometrisini bozmadan görüntüyü hedef panele yerleştirir."""
        source_height, source_width = image.shape[:2]
        if source_height <= 0 or source_width <= 0:
            raise ValueError("Kamera paneli icin bos goruntu verilemez.")

        scale = min(
            float(target_width) / float(source_width),
            float(target_height) / float(source_height),
        )
        resized_width = max(1, int(round(source_width * scale)))
        resized_height = max(1, int(round(source_height * scale)))
        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        canvas = np.full(
            (int(target_height), int(target_width), 3),
            (8, 11, 15),
            dtype=np.uint8,
        )
        offset_x = (int(target_width) - resized_width) // 2
        offset_y = (int(target_height) - resized_height) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        return canvas

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
        box = detection.get("bbox")
        if box is None or len(box) != 4:
            return False
        try:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
        except (TypeError, ValueError):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        try:
            confidence = float(detection.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
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
        return True

    def draw_perception_overlay(
        self,
        frame,
        vehicles,
        signs,
        lane_detection,
        road_context,
    ):
        """Ana kameraya kutuları ve şerit çizgilerini tek yerde çizer."""
        drawn_boxes = 0
        for detection in vehicles:
            drawn_boxes += int(
                self._draw(frame, detection, (0, 220, 0), "VEH")
            )
        for detection in signs:
            drawn_boxes += int(
                self._draw(frame, detection, (0, 165, 255), "SIGN")
            )

        drawn_lanes = self._draw_lanes(frame, lane_detection)
        for detection in (road_context or {}).get("detections", []):
            category = str(detection.get("category", "")).strip().lower()
            if category in {"vehicle", "two_wheeler", "speed_sign"}:
                continue
            if not category:
                continue
            color = (
                (0, 0, 255)
                if category == "traffic_light"
                else (255, 0, 255)
            )
            drawn_boxes += int(
                self._draw(frame, detection, color, category.upper())
            )
        return {
            "boxes": drawn_boxes,
            "lanes": drawn_lanes,
        }

    @staticmethod
    def _draw_lanes(frame, lane_detection):
        lanes = [
            lane
            for lane in (lane_detection or {}).get("lanes", [])
            if lane.get("detected") and len(lane.get("points", [])) >= 2
        ]
        PerceptionViewer._draw_ego_lane_corridor(frame, lanes)
        # BGR renkleri modelin yayımlandığı CARLA UFLD uygulamasındaki dört
        # lane kimliğiyle aynıdır: kırmızı, yeşil, mavi ve sarı.
        colors = (
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
            (0, 255, 255),
        )
        drawn = 0
        for lane in lanes:
            points = np.asarray(lane["points"], dtype=np.int32).reshape(-1, 1, 2)
            lane_index = int(lane.get("lane_index", 0))
            color = colors[lane_index % len(colors)]
            for raw_point in lane.get("raw_points", []):
                if len(raw_point) != 2:
                    continue
                point = (int(raw_point[0]), int(raw_point[1]))
                cv2.circle(
                    frame,
                    point,
                    4,
                    (10, 12, 15),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    point,
                    2,
                    color,
                    -1,
                    cv2.LINE_AA,
                )
            cv2.polylines(
                frame,
                [points],
                False,
                (12, 15, 18),
                7,
                cv2.LINE_AA,
            )
            cv2.polylines(
                frame,
                [points],
                False,
                color,
                3,
                cv2.LINE_AA,
            )
            bottom = max(lane["points"], key=lambda point: point[1])
            confidence = float(lane.get("confidence", 0.0))
            hold_marker = "*" if lane.get("temporally_held") else ""
            cv2.putText(
                frame,
                f"L{lane_index + 1}{hold_marker} {confidence:.2f}",
                (int(bottom[0]) + 7, max(45, int(bottom[1]) - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
            drawn += 1
        return drawn

    @staticmethod
    def _draw_ego_lane_corridor(frame, lanes):
        """UFLD'nin ego şeridi olan 1-2 sınırlarının arasını saydam doldurur."""
        lanes_by_index = {
            int(lane.get("lane_index", -1)): lane
            for lane in lanes
        }
        left = lanes_by_index.get(1)
        right = lanes_by_index.get(2)
        if left is None or right is None:
            return False

        left_points = np.asarray(left["points"], dtype=np.float64)
        right_points = np.asarray(right["points"], dtype=np.float64)
        y_min = max(float(np.min(left_points[:, 1])), float(np.min(right_points[:, 1])))
        y_max = min(float(np.max(left_points[:, 1])), float(np.max(right_points[:, 1])))
        if y_max - y_min < max(20.0, 0.08 * frame.shape[0]):
            return False

        sample_y = np.arange(y_min, y_max + 1.0, 4.0)
        left_order = np.argsort(left_points[:, 1])
        right_order = np.argsort(right_points[:, 1])
        left_x = np.interp(
            sample_y,
            left_points[left_order, 1],
            left_points[left_order, 0],
        )
        right_x = np.interp(
            sample_y,
            right_points[right_order, 1],
            right_points[right_order, 0],
        )
        lane_width = right_x - left_x
        if np.count_nonzero(lane_width > 4.0) < 0.90 * len(lane_width):
            return False

        polygon = np.vstack(
            (
                np.column_stack((left_x, sample_y)),
                np.column_stack((right_x[::-1], sample_y[::-1])),
            )
        )
        polygon[:, 0] = np.clip(polygon[:, 0], 0, frame.shape[1] - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, frame.shape[0] - 1)
        polygon = np.rint(polygon).astype(np.int32).reshape(-1, 1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], (60, 180, 90), lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.22, frame, 0.78, 0.0, dst=frame)
        return True

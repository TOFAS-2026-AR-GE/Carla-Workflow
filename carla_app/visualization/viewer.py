"""OpenCV camera and controller-performance dashboard."""

from __future__ import annotations

from collections import deque
import math

import cv2
import numpy as np


class PerceptionViewer:
    def __init__(
        self,
        window_name="CARLA Controller Dashboard",
        history_seconds=20.0,
        fixed_delta_seconds=0.05,
    ):
        self.window_name = window_name
        self.closed = False
        self.history_size = max(
            50,
            int(round(float(history_seconds) / max(1e-3, fixed_delta_seconds))),
        )
        self.history = {
            name: deque(maxlen=self.history_size)
            for name in (
                "speed_kmh",
                "reference_kmh",
                "target_kmh",
                "throttle",
                "brake",
                "gap_m",
            )
        }
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1360, 720)

    def show(
        self,
        result,
        fallback_image=None,
        fallback_frame_id=None,
        current_frame_id=None,
        state=None,
        control_info=None,
        lead_vehicle=None,
        traffic_light=None,
    ):
        if self.closed:
            return False

        state = state or {}
        control_info = control_info or {}
        traffic_light = traffic_light or {}
        if result is None:
            image = fallback_image
            result_frame_id = fallback_frame_id
            vehicles = []
            traffic_lights = []
            signs = []
            errors = {}
            elapsed_ms = 0.0
        else:
            image = result.get("image", fallback_image)
            result_frame_id = result.get("frame_id", fallback_frame_id)
            vehicles = result.get("vehicles", [])
            traffic_lights = result.get("traffic_lights", [])
            signs = result.get("signs", [])
            errors = result.get("errors", {})
            elapsed_ms = float(result.get("elapsed_ms", 0.0))

        if image is None:
            return self._window_is_open() and self._read_key()

        camera = np.ascontiguousarray(image[:, :, :3][:, :, ::-1]).copy()
        for detection in vehicles:
            self._draw(camera, detection, (0, 220, 0), "VEH")
        for detection in traffic_lights:
            light_state = self._light_state(detection.get("class_name"))
            color = {
                "red": (0, 0, 255),
                "yellow": (0, 220, 255),
                "green": (0, 220, 0),
            }.get(light_state, (200, 200, 200))
            self._draw(camera, detection, color, "TL")
        for detection in signs:
            self._draw(camera, detection, (0, 165, 255), "SIGN")

        lag = 0
        if current_frame_id is not None and result_frame_id is not None:
            lag = max(0, int(current_frame_id) - int(result_frame_id))
        header = (
            f"Frame {result_frame_id} | Vehicles {len(vehicles)} | "
            f"Traffic lights {len(traffic_lights)} | Lag {lag} | {elapsed_ms:.1f} ms"
        )
        header_color = (40, 40, 160) if errors else (20, 20, 20)
        cv2.rectangle(camera, (0, 0), (camera.shape[1], 34), header_color, -1)
        cv2.putText(
            camera,
            header,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if errors:
            error_text = " | ".join(
                f"{name}: {message}" for name, message in errors.items()
            )
            cv2.rectangle(camera, (0, 34), (camera.shape[1], 64), (20, 20, 180), -1)
            cv2.putText(
                camera,
                error_text[:130],
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        camera_panel = self._fit(camera, 840, 700)
        dashboard = np.full((700, 500, 3), 245, dtype=np.uint8)
        self._append_history(state, control_info, lead_vehicle)
        self._draw_telemetry(dashboard, state, control_info, lead_vehicle, traffic_light)
        self._draw_route(dashboard, state, top=245, height=205)
        self._draw_speed_plot(dashboard, top=465, height=205)

        canvas = np.full((720, 1360, 3), 20, dtype=np.uint8)
        canvas[10:710, 10:850] = camera_panel
        canvas[10:710, 860:1360] = dashboard
        cv2.imshow(self.window_name, canvas)
        self.closed = not (self._window_is_open() and self._read_key())
        return not self.closed

    def _append_history(self, state, control_info, lead_vehicle):
        speed_kmh = float(state.get("speed_kmh", 0.0))
        reference_kmh = float(control_info.get("requested_speed_mps", 0.0)) * 3.6
        target_kmh = float(control_info.get("target_speed_mps", 0.0)) * 3.6
        self.history["speed_kmh"].append(speed_kmh)
        self.history["reference_kmh"].append(reference_kmh)
        self.history["target_kmh"].append(target_kmh)
        self.history["throttle"].append(float(control_info.get("throttle", 0.0)))
        self.history["brake"].append(float(control_info.get("brake", 0.0)))
        self.history["gap_m"].append(
            float(lead_vehicle["distance_m"]) if lead_vehicle is not None else math.nan
        )

    def _draw_telemetry(self, panel, state, control_info, lead, traffic_light):
        longitudinal = control_info.get("longitudinal", {})
        speed_plan = control_info.get("speed_plan", {})
        lines = [
            ("MODE", str(control_info.get("mode", "-"))),
            ("SPEED", f"{float(state.get('speed_kmh', 0.0)):.1f} km/h"),
            ("REFERENCE", f"{float(control_info.get('requested_speed_mps', 0.0))*3.6:.1f} km/h"),
            ("TARGET", f"{float(control_info.get('target_speed_mps', 0.0))*3.6:.1f} km/h"),
            ("PEDALS", f"T {float(control_info.get('throttle', 0.0)):.2f}  B {float(control_info.get('brake', 0.0)):.2f}"),
            ("ACC CMD", f"{float(longitudinal.get('acceleration_mps2', 0.0)):+.2f} m/s2"),
            ("SPEED LIMIT", str(speed_plan.get("speed_reason", "-"))),
        ]
        if lead is None:
            lines.append(("LEAD", "none"))
        else:
            desired = longitudinal.get("desired_gap_m")
            desired_text = "-" if desired is None else f" / desired {float(desired):.1f}"
            lines.append(("LEAD", f"{float(lead['distance_m']):.1f} m{desired_text}"))
        distance = traffic_light.get("distance_m")
        distance_text = "-" if distance is None else f"{float(distance):.1f} m"
        lines.append(("LIGHT", f"{traffic_light.get('state', 'unknown')} / {distance_text}"))

        cv2.putText(panel, "CONTROLLER", (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
        y = 55
        for name, value in lines:
            cv2.putText(panel, name, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (90, 90, 90), 1, cv2.LINE_AA)
            cv2.putText(panel, value, (165, y), cv2.FONT_HERSHEY_SIMPLEX, 0.51, (20, 20, 20), 1, cv2.LINE_AA)
            y += 22

    def _draw_route(self, panel, state, top, height):
        left, right = 18, panel.shape[1] - 18
        bottom = top + height
        cv2.rectangle(panel, (left, top), (right, bottom), (220, 220, 220), 1)
        cv2.putText(panel, "ROUTE PREVIEW (ego frame)", (left + 8, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

        path = state.get("reference_path", []) or []
        ego = state.get("location")
        yaw = math.radians(float(state.get("yaw", 0.0)))
        if ego is None or not path:
            return

        center_x = (left + right) // 2
        plot_top = top + 28
        plot_bottom = bottom - 10
        lateral_span_m = 14.0
        forward_span_m = 60.0
        points = []
        for point in path:
            dx = float(point.x - ego.x)
            dy = float(point.y - ego.y)
            forward = math.cos(yaw) * dx + math.sin(yaw) * dy
            lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if forward < -2.0 or forward > forward_span_m:
                continue
            px = int(center_x + lateral / lateral_span_m * (right - left))
            py = int(plot_bottom - forward / forward_span_m * (plot_bottom - plot_top))
            points.append((px, py))
        if len(points) >= 2:
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, (60, 120, 210), 2, cv2.LINE_AA)
        cv2.circle(panel, (center_x, plot_bottom), 7, (30, 30, 30), -1)
        cv2.line(panel, (center_x, plot_bottom), (center_x, plot_bottom - 18), (30, 30, 30), 2)

    def _draw_speed_plot(self, panel, top, height):
        left, right = 18, panel.shape[1] - 18
        bottom = top + height
        cv2.rectangle(panel, (left, top), (right, bottom), (220, 220, 220), 1)
        cv2.putText(panel, "SPEED HISTORY", (left + 8, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1, cv2.LINE_AA)

        series = {
            "reference_kmh": (160, 160, 160),
            "target_kmh": (0, 150, 230),
            "speed_kmh": (40, 80, 200),
        }
        maximum = max(
            20.0,
            *(max(values, default=0.0) for name, values in self.history.items() if name in series),
        )
        maximum = math.ceil(maximum / 10.0) * 10.0
        plot_left, plot_right = left + 38, right - 8
        plot_top, plot_bottom = top + 30, bottom - 24

        for fraction in (0.0, 0.5, 1.0):
            y = int(plot_bottom - fraction * (plot_bottom - plot_top))
            cv2.line(panel, (plot_left, y), (plot_right, y), (225, 225, 225), 1)
            cv2.putText(panel, f"{fraction*maximum:.0f}", (left + 2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90, 90, 90), 1, cv2.LINE_AA)

        for name, color in series.items():
            values = list(self.history[name])
            if len(values) < 2:
                continue
            points = []
            for index, value in enumerate(values):
                x = int(plot_left + index / max(1, self.history_size - 1) * (plot_right - plot_left))
                y = int(plot_bottom - float(value) / maximum * (plot_bottom - plot_top))
                points.append((x, y))
            cv2.polylines(panel, [np.asarray(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)

        legend_y = bottom - 7
        labels = [("ref", series["reference_kmh"]), ("target", series["target_kmh"]), ("actual", series["speed_kmh"])]
        x = plot_left
        for label, color in labels:
            cv2.line(panel, (x, legend_y - 4), (x + 18, legend_y - 4), color, 2)
            cv2.putText(panel, label, (x + 23, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (60, 60, 60), 1, cv2.LINE_AA)
            x += 95

    @staticmethod
    def _fit(image, width, height):
        scale = min(width / image.shape[1], height / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        y = (height - resized.shape[0]) // 2
        x = (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        return canvas

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

    @staticmethod
    def _read_key():
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"), ord("Q"))

    @staticmethod
    def _draw(frame, detection, color, prefix):
        x1, y1, x2, y2 = [int(value) for value in detection["bbox"]]
        confidence = float(detection.get("confidence", 0.0))
        class_name = str(detection.get("class_name", "unknown"))
        label = f"{prefix} {class_name} {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    @staticmethod
    def _light_state(class_name):
        text = str(class_name or "").lower()
        if "red" in text:
            return "red"
        if "yellow" in text or "orange" in text or "amber" in text:
            return "yellow"
        if "green" in text:
            return "green"
        return "unknown"

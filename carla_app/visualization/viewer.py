import cv2
import numpy as np


class PerceptionViewer:
    def __init__(self, window_name="CARLA Perception"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def show(self, result, fallback_image=None, fallback_frame_id=None):
        if result is None:
            image = fallback_image
            frame_id = fallback_frame_id
            vehicles = []
            signs = []
            elapsed_ms = 0.0
        else:
            image = result["image"]
            frame_id = result["frame_id"]
            vehicles = result["vehicles"]
            signs = result["signs"]
            elapsed_ms = result["elapsed_ms"]

        if image is None:
            return self._read_key()

        frame = np.ascontiguousarray(image[:, :, ::-1]).copy()
        for detection in vehicles:
            self._draw(frame, detection, (0, 220, 0), "VEH")
        for detection in signs:
            self._draw(frame, detection, (0, 165, 255), "SIGN")

        header = (
            f"Frame {frame_id} | Vehicles {len(vehicles)} | "
            f"Signs {len(signs)} | {elapsed_ms:.1f} ms"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 32), (20, 20, 20), -1)
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

        cv2.imshow(self.window_name, frame)
        return self._read_key()

    def close(self):
        cv2.destroyAllWindows()

    def _read_key(self):
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"), ord("Q"))

    @staticmethod
    def _draw(frame, detection, color, prefix):
        x1, y1, x2, y2 = detection["bbox"]
        confidence = detection["confidence"]
        label = f"{prefix} {detection['class_name']} {confidence:.2f}"
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

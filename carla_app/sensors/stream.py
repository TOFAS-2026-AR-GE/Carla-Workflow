import threading
import time


class CameraStream:
    def __init__(self, max_frames=10):
        self.max_frames = max_frames
        self.frames = {}
        self.condition = threading.Condition()

    def push(self, frame_id, rgb_image):
        with self.condition:
            self.frames[int(frame_id)] = rgb_image
            while len(self.frames) > self.max_frames:
                del self.frames[min(self.frames)]
            self.condition.notify_all()

    def wait(self, frame_id, timeout=0.5):
        frame_id = int(frame_id)
        end_time = time.monotonic() + timeout

        with self.condition:
            while frame_id not in self.frames:
                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(remaining)

            image = self.frames.pop(frame_id)
            for old_id in [item for item in self.frames if item < frame_id]:
                del self.frames[old_id]
            return image

    def clear(self):
        with self.condition:
            self.frames.clear()

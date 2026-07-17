"""Veri kaydı için bütün sensörlerin aynı karede buluşmasını bekler."""

import threading
import time


class SensorSync:
    def __init__(self, sensor_names, max_frames=20):
        self.sensor_names = set(sensor_names)
        self.max_frames = max_frames
        self.frames = {}
        self.condition = threading.Condition()

    def push(self, sensor_name, frame_id, data):
        with self.condition:
            frame_id = int(frame_id)
            if frame_id not in self.frames:
                self.frames[frame_id] = {}
            self.frames[frame_id][sensor_name] = data
            while len(self.frames) > self.max_frames:
                del self.frames[min(self.frames)]
            self.condition.notify_all()

    def wait(self, frame_id, timeout=1.0):
        frame_id = int(frame_id)
        end_time = time.monotonic() + timeout

        with self.condition:
            while True:
                packet = self.frames.get(frame_id, {})
                packet_complete = True
                for sensor_name in self.sensor_names:
                    if sensor_name not in packet:
                        packet_complete = False
                        break

                if packet_complete:
                    result = self.frames.pop(frame_id)
                    self._remove_older(frame_id)
                    return result

                remaining = end_time - time.monotonic()
                if remaining <= 0:
                    self._remove_older(frame_id)
                    return None
                self.condition.wait(remaining)

    def clear(self):
        with self.condition:
            self.frames.clear()

    def _remove_older(self, frame_id):
        old_frame_ids = []
        for item in self.frames:
            if item < frame_id:
                old_frame_ids.append(item)
        for old_id in old_frame_ids:
            del self.frames[old_id]

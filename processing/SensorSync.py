import time
from collections import defaultdict

class SensorSync:
    def __init__(self, sensors):
        self.sensors = set(sensors)
        self.buffer = defaultdict(dict)
        
    def push(self, sensor_name, frame, data):
        self.buffer[frame][sensor_name] = data
    
    def wait_for_frame(self, frame_id, timeout=2.0, poll_interval=0.005):
        start = time.time()
        while time.time() - start < timeout:
            frame_data = self.buffer.get(frame_id, {})
            if self.sensors.issubset(frame_data.keys()):
                self.cleanup(frame_id)
                return frame_data
            time.sleep(poll_interval)

        frame_data = self.buffer.get(frame_id, {})
        missing = self.sensors - frame_data.keys()
        if missing:
            print(f"[WARN] Frame {frame_id}: eksik sensor verisi -> {missing}")
            self.cleanup(frame_id)
        return frame_data
    
    def cleanup(self, upto_frame):
        stale = [f for f in self.buffer if f <= upto_frame]
        for f in stale:
            del self.buffer[f]
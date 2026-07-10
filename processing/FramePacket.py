
class FramePacket:
    def __init__(self, frame_id, sensor_data, expected_sensors):
        self.frame_id = int(frame_id)
        self.sensor_data = dict(sensor_data)
        self.expected_sensors = list(expected_sensors)
        
    
    def is_complete(self):
        for sensor_name in self.expected_sensors:
            if sensor_name not in self.sensor_data:
                return False
        
        return True
    
    def missing_sensors(self):
        missing = []
        for sensor_name in self.expected_sensors:
            if sensor_name not in self.sensor_data:
                missing.append(sensor_name)

        return missing
    
    def get(self, sensor_name):
        return self.sensor_data.get(sensor_name)
    
    def to_dict(self):
        return {
            "frame_id": self.frame_id,
            "complete": self.is_complete,
            "received_sensors": sorted(self.sensor_data.keys()),
            "expected_sensors": sorted(self.expected_sensors),
            "missing_sensors": sorted(self.missing_sensors()),
        }
    
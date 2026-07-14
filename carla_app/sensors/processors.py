import math

import numpy as np


def image_to_rgb(image):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape(image.height, image.width, 4)
    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def lidar_to_array(lidar):
    return np.frombuffer(lidar.raw_data, dtype=np.float32).reshape(-1, 4).copy()


def gnss_to_dict(gnss):
    return {
        "latitude": float(gnss.latitude),
        "longitude": float(gnss.longitude),
        "altitude": float(gnss.altitude),
    }


def imu_to_dict(imu):
    return {
        "accelerometer": {
            "x": float(imu.accelerometer.x),
            "y": float(imu.accelerometer.y),
            "z": float(imu.accelerometer.z),
        },
        "gyroscope": {
            "x": float(imu.gyroscope.x),
            "y": float(imu.gyroscope.y),
            "z": float(imu.gyroscope.z),
        },
        "compass": float(imu.compass),
    }


def radar_to_list(radar):
    return [
        {
            "depth": float(point.depth),
            "velocity": float(point.velocity),
            "azimuth_deg": math.degrees(float(point.azimuth)),
            "altitude_deg": math.degrees(float(point.altitude)),
        }
        for point in radar
    ]


def process_packet(packet):
    return {
        "rgb": image_to_rgb(packet["rgb_camera"]),
        "lidar": lidar_to_array(packet["lidar"]),
        "gnss": gnss_to_dict(packet["gnss"]),
        "imu": imu_to_dict(packet["imu"]),
        "radars": {
            name: radar_to_list(packet[name])
            for name in (
                "radar_front",
                "radar_rear",
                "radar_left",
                "radar_right",
            )
        },
    }

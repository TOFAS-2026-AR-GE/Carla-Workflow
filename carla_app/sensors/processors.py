"""CARLA sensör nesnelerini sade Python ve NumPy verilerine çevirir."""

import math

import numpy as np


def image_to_rgb(image):
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        image.height,
        image.width,
        4,
    )

    return np.ascontiguousarray(array[:, :, :3][:, :, ::-1])


def lidar_to_array(lidar):
    array = np.frombuffer(lidar.raw_data, dtype=np.float32)
    array = array.reshape(-1, 4)
    return array.copy()


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
    points = []
    for point in radar:
        points.append(
            {
                "depth_m": float(point.depth),
                "relative_velocity_mps": float(point.velocity),
                "azimuth_deg": math.degrees(float(point.azimuth)),
                "altitude_deg": math.degrees(float(point.altitude)),
            }
        )
    return points


def process_packet(
    packet,
    layout,
):
    """Aynı kareye ait tam sensör paketini kayıt biçimine dönüştürür."""
    cameras = {}
    for camera in layout.cameras:
        cameras[camera.name] = image_to_rgb(packet[camera.name])

    radars = {}
    for radar in layout.radars:
        radars[radar.name] = radar_to_list(packet[radar.name])

    return {
        "primary_camera": layout.primary_camera_name,
        "cameras": cameras,
        "lidar": lidar_to_array(packet[layout.lidar.name]),
        "gnss": gnss_to_dict(packet[layout.gnss.name]),
        "imu": imu_to_dict(packet[layout.imu.name]),
        "radars": radars,
    }

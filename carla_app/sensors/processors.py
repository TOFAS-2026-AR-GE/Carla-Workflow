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
    return (
        np.frombuffer(
            lidar.raw_data,
            dtype=np.float32,
        )
        .reshape(-1, 4)
        .copy()
    )


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
            "depth_m": float(point.depth),
            "relative_velocity_mps": float(point.velocity),
            "azimuth_deg": math.degrees(float(point.azimuth)),
            "altitude_deg": math.degrees(float(point.altitude)),
        }
        for point in radar
    ]


def ultrasonic_to_dict(
    measurement,
    maximum_range,
):
    nearest = None

    for point in measurement:
        if nearest is None or point.depth < nearest.depth:
            nearest = point

    if nearest is None:
        return {
            "detected": False,
            "distance_m": float(maximum_range),
            "relative_velocity_mps": 0.0,
            "azimuth_deg": 0.0,
            "altitude_deg": 0.0,
        }

    return {
        "detected": True,
        "distance_m": float(nearest.depth),
        "relative_velocity_mps": float(nearest.velocity),
        "azimuth_deg": math.degrees(float(nearest.azimuth)),
        "altitude_deg": math.degrees(float(nearest.altitude)),
    }


def process_packet(
    packet,
    layout,
):
    cameras = {spec.name: image_to_rgb(packet[spec.name]) for spec in layout.cameras}

    radars = {spec.name: radar_to_list(packet[spec.name]) for spec in layout.radars}

    ultrasonics = {
        spec.name: ultrasonic_to_dict(
            packet[spec.name],
            float(
                spec.attributes.get(
                    "range",
                    "4.5",
                )
            ),
        )
        for spec in layout.ultrasonics
    }

    return {
        "primary_camera": (layout.primary_camera_name),
        "cameras": cameras,
        "lidar": lidar_to_array(packet[layout.lidar.name]),
        "gnss": gnss_to_dict(packet[layout.gnss.name]),
        "imu": imu_to_dict(packet[layout.imu.name]),
        "radars": radars,
        "ultrasonics": ultrasonics,
    }

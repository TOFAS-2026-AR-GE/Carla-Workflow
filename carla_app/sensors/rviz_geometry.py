"""CARLA sensor yerlesimini ROS koordinatlarina ceviren sade yardimcilar."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SensorVisual:
    name: str
    kind: str
    active: bool
    position_xyz: tuple
    orientation_xyzw: tuple
    horizontal_fov_deg: float
    display_range_m: float


def carla_location_to_ros_xyz(location):
    """CARLA x-ileri/y-sag konumunu ROS x-ileri/y-sol konumuna cevirir."""
    return (
        float(location.x),
        -float(location.y),
        float(location.z),
    )


def carla_rotation_to_ros_quaternion(rotation):
    """CARLA sol-el rotasyonunu ROS sag-el quaternion'una cevirir."""
    roll = math.radians(float(rotation.roll))
    pitch = -math.radians(float(rotation.pitch))
    yaw = -math.radians(float(rotation.yaw))

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def sensor_display_range(spec):
    """Gercek menzili RViz'de okunabilir bir uzunlukta sinirlar."""
    if spec.kind == "camera":
        return 12.0
    if spec.kind == "lidar":
        return 5.0
    if spec.kind in ("radar", "ultrasonic"):
        return min(25.0, float(spec.attributes.get("range", 10.0)))
    return 0.0


def sensor_horizontal_fov(spec):
    if spec.kind == "camera":
        return float(spec.attributes.get("fov", 0.0))
    return float(spec.attributes.get("horizontal_fov", 0.0))


def build_sensor_visuals(layout, active_sensor_names):
    """Layout kayitlarini ROS frame'inde cizilebilir kayitlara donusturur."""
    active_sensor_names = set(active_sensor_names)
    visuals = []

    for spec in layout.all_specs:
        location = spec.transform.location
        visuals.append(
            SensorVisual(
                name=spec.name,
                kind=spec.kind,
                active=spec.name in active_sensor_names,
                position_xyz=carla_location_to_ros_xyz(location),
                orientation_xyzw=carla_rotation_to_ros_quaternion(
                    spec.transform.rotation
                ),
                horizontal_fov_deg=sensor_horizontal_fov(spec),
                display_range_m=sensor_display_range(spec),
            )
        )

    return visuals

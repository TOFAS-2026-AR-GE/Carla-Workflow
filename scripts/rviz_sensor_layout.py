#!/usr/bin/env python3
"""Planlanan ve aktif CARLA sensorlerini RViz MarkerArray olarak yayinlar."""

import math
import sys
from pathlib import Path

import carla
import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_app.sensors.layout import build_sensor_layout  # noqa: E402
from carla_app.sensors.rviz_geometry import (  # noqa: E402
    build_sensor_visuals,
    carla_location_to_ros_xyz,
    carla_rotation_to_ros_quaternion,
)


SENSOR_COLORS = {
    "camera": (0.20, 1.00, 0.25),
    "radar": (1.00, 0.20, 0.15),
    "lidar": (0.15, 0.55, 1.00),
    "ultrasonic": (1.00, 0.80, 0.10),
    "gnss": (0.90, 0.25, 1.00),
    "imu": (0.10, 1.00, 1.00),
}


def set_color(marker, kind, active, alpha=1.0):
    red, green, blue = SENSOR_COLORS.get(kind, (1.0, 1.0, 1.0))
    marker.color.r = red
    marker.color.g = green
    marker.color.b = blue
    marker.color.a = alpha if active else min(alpha, 0.22)


def set_sensor_pose(marker, visual):
    x, y, z = visual.position_xyz
    qx, qy, qz, qw = visual.orientation_xyzw
    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = z
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw


def point(x, y, z=0.0):
    value = Point()
    value.x = float(x)
    value.y = float(y)
    value.z = float(z)
    return value


class SensorLayoutPublisher(Node):
    """CARLA layout.py icindeki sensorleri ego araci etrafinda cizer."""

    def __init__(self):
        super().__init__("carla_sensor_layout_visualizer")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("frame_id", "ego_vehicle")
        self.declare_parameter("show_inactive", True)
        self.declare_parameter("camera_width", 800)
        self.declare_parameter("camera_height", 600)
        self.declare_parameter("camera_fov", 90.0)
        self.declare_parameter("fixed_delta_seconds", 0.05)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.publisher = self.create_publisher(
            MarkerArray,
            "/carla/ego_vehicle/sensor_layout",
            qos,
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.connection_warning_printed = False
        self.vehicle_warning_printed = False
        self.last_signature = None
        self.world = None
        self.ego_vehicle = None
        self.layout_timer = self.create_timer(1.0, self.update_layout)
        self.tf_timer = self.create_timer(0.05, self.publish_ego_transform)

    def update_layout(self):
        try:
            world = self.client.get_world()
        except RuntimeError as error:
            if not self.connection_warning_printed:
                self.get_logger().warning(f"CARLA bekleniyor: {error}")
                self.connection_warning_printed = True
            return

        self.connection_warning_printed = False
        self.world = world
        vehicle = self.find_ego_vehicle(world)
        if vehicle is None:
            self.ego_vehicle = None
            if not self.vehicle_warning_printed:
                role = self.get_parameter("ego_role_name").value
                self.get_logger().warning(f"role_name={role} ego araci bekleniyor")
                self.vehicle_warning_printed = True
            return

        self.vehicle_warning_printed = False
        self.ego_vehicle = vehicle
        active_names = self.active_sensor_names(world, vehicle)
        signature = (vehicle.id, tuple(sorted(active_names)))
        if signature == self.last_signature:
            return

        layout = build_sensor_layout(
            vehicle=vehicle,
            camera_width=int(self.get_parameter("camera_width").value),
            camera_height=int(self.get_parameter("camera_height").value),
            front_wide_fov=float(self.get_parameter("camera_fov").value),
            fixed_delta_seconds=float(
                self.get_parameter("fixed_delta_seconds").value
            ),
        )
        visuals = build_sensor_visuals(layout, active_names)
        if not bool(self.get_parameter("show_inactive").value):
            visuals = [visual for visual in visuals if visual.active]

        self.publisher.publish(
            self.create_markers(visuals, layout.vehicle_geometry)
        )
        self.publish_sensor_transforms(visuals)
        self.last_signature = signature
        self.get_logger().info(
            f"RViz layout: {len(active_names)} aktif, "
            f"{len(layout.all_specs)} planli sensor"
        )

    def publish_ego_transform(self):
        vehicle = self.ego_vehicle
        if vehicle is None:
            return

        try:
            if not vehicle.is_alive:
                return
            carla_transform = vehicle.get_transform()
        except RuntimeError:
            self.ego_vehicle = None
            return
        x, y, z = carla_location_to_ros_xyz(carla_transform.location)
        qx, qy, qz, qw = carla_rotation_to_ros_quaternion(
            carla_transform.rotation
        )

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = self.get_parameter("frame_id").value
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

    def publish_sensor_transforms(self, visuals):
        parent_frame = self.get_parameter("frame_id").value
        transforms = []
        for visual in visuals:
            x, y, z = visual.position_xyz
            qx, qy, qz, qw = visual.orientation_xyzw
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = parent_frame
            transform.child_frame_id = f"{parent_frame}/layout/{visual.name}"
            transform.transform.translation.x = x
            transform.transform.translation.y = y
            transform.transform.translation.z = z
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw
            transforms.append(transform)
        if transforms:
            self.static_tf_broadcaster.sendTransform(transforms)

    def find_ego_vehicle(self, world):
        role_name = self.get_parameter("ego_role_name").value
        for actor in world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == role_name:
                return actor
        return None

    def active_sensor_names(self, world, vehicle):
        names = set()
        for actor in world.get_actors().filter("sensor.*"):
            parent = getattr(actor, "parent", None)
            if parent is None or parent.id != vehicle.id:
                continue
            name = actor.attributes.get("role_name")
            if name:
                names.add(name)
        return names

    def create_markers(self, visuals, vehicle_geometry):
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        markers.markers.append(self.vehicle_body_marker(vehicle_geometry))

        marker_id = 1
        for visual in visuals:
            markers.markers.extend(
                self.sensor_markers(visual, marker_id)
            )
            marker_id += 10
        return markers

    def vehicle_body_marker(self, geometry):
        marker = self.base_marker(0, "vehicle_body", Marker.CUBE)
        marker.pose.position.x = geometry["bounding_box_center_x_m"]
        marker.pose.position.y = -geometry["bounding_box_center_y_m"]
        marker.pose.position.z = geometry["bounding_box_center_z_m"]
        marker.pose.orientation.w = 1.0
        marker.scale.x = geometry["length_m"]
        marker.scale.y = geometry["width_m"]
        marker.scale.z = geometry["height_m"]
        marker.color.r = 0.55
        marker.color.g = 0.58
        marker.color.b = 0.62
        marker.color.a = 0.24
        return marker

    def base_marker(self, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = self.get_parameter("frame_id").value
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        return marker

    def sensor_markers(self, visual, marker_id):
        markers = []

        body_type = (
            Marker.SPHERE
            if visual.kind in ("lidar", "gnss", "imu")
            else Marker.CUBE
        )
        body = self.base_marker(marker_id, "sensor_body", body_type)
        set_sensor_pose(body, visual)
        body.scale.x = 0.22
        body.scale.y = 0.16
        body.scale.z = 0.14
        set_color(body, visual.kind, visual.active)
        markers.append(body)

        direction = self.base_marker(marker_id + 1, "sensor_direction", Marker.ARROW)
        set_sensor_pose(direction, visual)
        direction.scale.x = 0.75
        direction.scale.y = 0.09
        direction.scale.z = 0.09
        set_color(direction, visual.kind, visual.active)
        markers.append(direction)

        label = self.base_marker(marker_id + 2, "sensor_name", Marker.TEXT_VIEW_FACING)
        x, y, z = visual.position_xyz
        label.pose.position.x = x
        label.pose.position.y = y
        label.pose.position.z = z + 0.28
        label.pose.orientation.w = 1.0
        label.scale.z = 0.16
        label.text = visual.name if visual.active else f"{visual.name} [plan]"
        set_color(label, visual.kind, visual.active)
        markers.append(label)

        if visual.kind == "lidar":
            markers.append(self.lidar_circle(visual, marker_id + 3))
        elif visual.horizontal_fov_deg > 0.0 and visual.display_range_m > 0.0:
            markers.append(self.fov_marker(visual, marker_id + 3))

        return markers

    def fov_marker(self, visual, marker_id):
        marker = self.base_marker(marker_id, "sensor_fov", Marker.LINE_LIST)
        set_sensor_pose(marker, visual)
        marker.scale.x = 0.035
        set_color(marker, visual.kind, visual.active, alpha=0.70)

        half_angle = math.radians(visual.horizontal_fov_deg * 0.5)
        forward = visual.display_range_m * math.cos(half_angle)
        side = visual.display_range_m * math.sin(half_angle)
        origin = point(0.0, 0.0)
        left = point(forward, side)
        right = point(forward, -side)
        marker.points = [origin, left, origin, right, left, right]
        return marker

    def lidar_circle(self, visual, marker_id):
        marker = self.base_marker(marker_id, "sensor_fov", Marker.LINE_STRIP)
        set_sensor_pose(marker, visual)
        marker.scale.x = 0.035
        set_color(marker, visual.kind, visual.active, alpha=0.70)

        segments = 48
        marker.points = [
            point(
                visual.display_range_m * math.cos(2.0 * math.pi * i / segments),
                visual.display_range_m * math.sin(2.0 * math.pi * i / segments),
            )
            for i in range(segments + 1)
        ]
        return marker


def main():
    rclpy.init()
    node = SensorLayoutPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

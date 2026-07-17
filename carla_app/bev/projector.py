"""Sensör ölçümlerini ego aracına göre metre cinsinden X-Y düzlemine taşır."""

import math

import numpy as np


def rotate_point(x, y, yaw_deg):
    """Bir noktayı verilen yaw açısı kadar döndürür."""
    yaw = math.radians(float(yaw_deg))
    rotated_x = x * math.cos(yaw) - y * math.sin(yaw)
    rotated_y = x * math.sin(yaw) + y * math.cos(yaw)
    return rotated_x, rotated_y


class BevProjector:
    """Kamera, radar, LiDAR ve rotayı ortak ego koordinatına dönüştürür."""

    def __init__(self, layout, maximum_lidar_points=6000):
        self.layout = layout
        self.maximum_lidar_points = int(maximum_lidar_points)

        geometry = layout.vehicle_geometry
        self.vehicle_bottom_z = (
            float(geometry["bounding_box_center_z_m"])
            - float(geometry["half_height_m"])
        )

    def build_scene(self, sensor_snapshot, perception_result, vehicle_state):
        """Renderer tarafından çizilecek sade BEV sahnesini oluşturur."""
        return {
            "lidar_points": self.project_lidar(sensor_snapshot),
            "radar_points": self.project_radars(sensor_snapshot),
            "detections": self.project_camera_detections(perception_result),
            "route_points": self.project_route(vehicle_state),
            "active_sensor_count": len(sensor_snapshot),
            "total_sensor_count": len(self.layout.all_specs),
            "vehicle_geometry": dict(self.layout.vehicle_geometry),
        }

    def project_lidar(self, sensor_snapshot):
        """LiDAR noktalarını sensör konumundan ego araç koordinatına taşır."""
        entry = sensor_snapshot.get(self.layout.lidar.name)
        if entry is None:
            return np.empty((0, 2), dtype=np.float32)

        points = np.asarray(entry["data"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)

        if len(points) > self.maximum_lidar_points:
            step = int(math.ceil(len(points) / self.maximum_lidar_points))
            points = points[::step]

        transform = self.layout.lidar.transform
        yaw = math.radians(float(transform.rotation.yaw))
        x = points[:, 0]
        y = points[:, 1]
        rotated_x = x * math.cos(yaw) - y * math.sin(yaw)
        rotated_y = x * math.sin(yaw) + y * math.cos(yaw)
        ego_x = rotated_x + float(transform.location.x)
        ego_y = rotated_y + float(transform.location.y)

        valid = np.isfinite(ego_x) & np.isfinite(ego_y)
        valid &= np.hypot(ego_x, ego_y) <= 90.0
        return np.column_stack((ego_x[valid], ego_y[valid])).astype(np.float32)

    def project_radars(self, sensor_snapshot):
        """Beş radarın kutupsal noktalarını ego X-Y koordinatına çevirir."""
        projected = []

        for radar in self.layout.radars:
            entry = sensor_snapshot.get(radar.name)
            if entry is None:
                continue

            sensor_x = float(radar.transform.location.x)
            sensor_y = float(radar.transform.location.y)
            sensor_yaw = float(radar.transform.rotation.yaw)

            for point in entry["data"]:
                try:
                    depth = float(point["depth_m"])
                    azimuth = float(point["azimuth_deg"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(depth) or not math.isfinite(azimuth):
                    continue
                if not 0.0 < depth <= 90.0:
                    continue

                angle = math.radians(azimuth)
                local_x = depth * math.cos(angle)
                local_y = depth * math.sin(angle)
                rotated_x, rotated_y = rotate_point(
                    local_x,
                    local_y,
                    sensor_yaw,
                )
                projected.append(
                    {
                        "x_m": sensor_x + rotated_x,
                        "y_m": sensor_y + rotated_y,
                        "sensor_name": radar.name,
                        "relative_velocity_mps": float(
                            point.get("relative_velocity_mps", 0.0)
                        ),
                    }
                )

        return projected

    def project_camera_detections(self, perception_result):
        """BBox alt merkezini zeminle kesiştirerek yaklaşık BEV konumu bulur."""
        if not perception_result:
            return []

        camera_results = perception_result.get("camera_results", {})
        projected = []

        for camera in self.layout.cameras:
            result = camera_results.get(camera.name)
            if result is None or result.get("image") is None:
                continue

            image = result["image"]
            image_height, image_width = image.shape[:2]
            for detection in result.get("vehicles", []):
                position = self.project_detection(
                    camera,
                    detection,
                    image_width,
                    image_height,
                )
                if position is None:
                    continue
                position["camera_name"] = camera.name
                position["class_name"] = detection.get(
                    "class_name", "vehicle"
                )
                position["confidence"] = float(
                    detection.get("confidence", 0.0)
                )
                projected.append(position)

        return projected

    def project_detection(self, camera, detection, image_width, image_height):
        """Tek bbox'ın alt merkezinden zemin üzerindeki yaklaşık noktayı bulur."""
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None

        pixel_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        pixel_y = float(bbox[3])
        fov_deg = float(camera.attributes.get("fov", 90.0))
        focal_length = float(image_width) / (
            2.0 * math.tan(math.radians(fov_deg) / 2.0)
        )

        ray_x = 1.0
        ray_y = (pixel_x - 0.5 * image_width) / focal_length
        ray_z = -(pixel_y - 0.5 * image_height) / focal_length

        pitch = math.radians(float(camera.transform.rotation.pitch))
        pitched_x = math.cos(pitch) * ray_x - math.sin(pitch) * ray_z
        pitched_z = math.sin(pitch) * ray_x + math.cos(pitch) * ray_z
        if pitched_z >= -0.01:
            return None

        camera_height = (
            float(camera.transform.location.z) - self.vehicle_bottom_z
        )
        distance_scale = camera_height / -pitched_z
        local_x = distance_scale * pitched_x
        local_y = distance_scale * ray_y
        rotated_x, rotated_y = rotate_point(
            local_x,
            local_y,
            camera.transform.rotation.yaw,
        )
        ego_x = float(camera.transform.location.x) + rotated_x
        ego_y = float(camera.transform.location.y) + rotated_y

        if not math.isfinite(ego_x) or not math.isfinite(ego_y):
            return None
        if math.hypot(ego_x, ego_y) > 100.0:
            return None
        return {"x_m": ego_x, "y_m": ego_y}

    def project_route(self, vehicle_state):
        """Dünya koordinatındaki referans rotayı ego koordinatına çevirir."""
        if not vehicle_state:
            return []

        location = vehicle_state.get("location")
        path = vehicle_state.get("reference_path", [])
        if location is None:
            return []

        yaw = math.radians(float(vehicle_state.get("yaw", 0.0)))
        projected = []
        for point in path:
            dx = float(point.x) - float(location.x)
            dy = float(point.y) - float(location.y)
            forward = dx * math.cos(yaw) + dy * math.sin(yaw)
            right = -dx * math.sin(yaw) + dy * math.cos(yaw)
            projected.append((forward, right))
        return projected

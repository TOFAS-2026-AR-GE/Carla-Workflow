"""Bütün sensör ölçümlerini kalibrasyonla ortak ego koordinatına taşır."""

import math

import numpy as np

from carla_app.bev.calibration import CalibrationSet


class BevProjector:
    """Kamera, radar, LiDAR ve rotayı metre tabanlı ego düzlemine taşır."""

    def __init__(
        self,
        layout,
        calibrations=None,
        motion_compensator=None,
        maximum_lidar_points=12000,
    ):
        self.layout = layout
        self.calibrations = calibrations or CalibrationSet(layout)
        self.motion_compensator = motion_compensator
        self.maximum_lidar_points = int(maximum_lidar_points)
        self.ground_z_m = self.calibrations.ground_z_m

    def build_scene(
        self,
        sensor_snapshot,
        perception_result,
        vehicle_state,
        current_frame_id=None,
    ):
        """Füzyon ve çizim katmanlarının kullanacağı kalibre sahneyi üretir."""
        return {
            "lidar_points": self.project_lidar(
                sensor_snapshot,
                current_frame_id,
            ),
            "lidar_origin": self.sensor_origin(
                self.layout.lidar.name,
                sensor_snapshot,
                current_frame_id,
            ),
            "radar_points": self.project_radars(
                sensor_snapshot,
                current_frame_id,
            ),
            "camera_objects": self.project_camera_detections(
                perception_result,
                current_frame_id,
            ),
            "route_points": self.project_route(vehicle_state),
            "active_sensor_count": len(sensor_snapshot),
            "total_sensor_count": len(self.layout.all_specs),
            "vehicle_geometry": dict(self.layout.vehicle_geometry),
            "ground_z_m": self.ground_z_m,
        }

    def project_lidar(self, sensor_snapshot, current_frame_id=None):
        """LiDAR XYZ noktalarını tam R-T dönüşümüyle ego karesine taşır."""
        entry = sensor_snapshot.get(self.layout.lidar.name)
        if entry is None:
            return np.empty((0, 3), dtype=np.float32)

        points = np.asarray(entry["data"], dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)

        if len(points) > self.maximum_lidar_points:
            step = int(math.ceil(len(points) / self.maximum_lidar_points))
            points = points[::step]

        calibration = self.calibrations.get_sensor(self.layout.lidar.name)
        ego_points = calibration.sensor_points_to_ego(points[:, :3])
        ego_points = self.compensate_points(
            ego_points,
            entry.get("frame_id"),
            current_frame_id,
        )

        finite = np.all(np.isfinite(ego_points), axis=1)
        finite &= np.hypot(ego_points[:, 0], ego_points[:, 1]) <= 90.0
        finite &= ego_points[:, 2] >= self.ground_z_m - 1.0
        finite &= ego_points[:, 2] <= self.ground_z_m + 5.0
        return ego_points[finite].astype(np.float32)

    def project_radars(self, sensor_snapshot, current_frame_id=None):
        """Beş radarın kutupsal 3B noktalarını ortak ego karesine taşır."""
        projected = []

        for radar in self.layout.radars:
            entry = sensor_snapshot.get(radar.name)
            if entry is None:
                continue

            local_points = []
            valid_metadata = []
            maximum_range = float(radar.attributes.get("range", 90.0))

            for point in entry["data"]:
                try:
                    depth = float(point["depth_m"])
                    azimuth = math.radians(float(point["azimuth_deg"]))
                    altitude = math.radians(float(point["altitude_deg"]))
                except (KeyError, TypeError, ValueError):
                    continue
                if not math.isfinite(depth) or not 0.0 < depth <= maximum_range:
                    continue

                horizontal_depth = depth * math.cos(altitude)
                local_points.append(
                    (
                        horizontal_depth * math.cos(azimuth),
                        horizontal_depth * math.sin(azimuth),
                        depth * math.sin(altitude),
                    )
                )
                valid_metadata.append(point)

            if not local_points:
                continue

            calibration = self.calibrations.get_sensor(radar.name)
            ego_points = calibration.sensor_points_to_ego(local_points)
            ego_points = self.compensate_points(
                ego_points,
                entry.get("frame_id"),
                current_frame_id,
            )

            origin = np.array(
                [[calibration.T[0], calibration.T[1], calibration.T[2]]],
                dtype=np.float64,
            )
            origin = self.compensate_points(
                origin,
                entry.get("frame_id"),
                current_frame_id,
            )[0]

            for index, ego_point in enumerate(ego_points):
                if not np.all(np.isfinite(ego_point)):
                    continue
                metadata = valid_metadata[index]
                projected.append(
                    {
                        "x_m": float(ego_point[0]),
                        "y_m": float(ego_point[1]),
                        "z_m": float(ego_point[2]),
                        "origin_x_m": float(origin[0]),
                        "origin_y_m": float(origin[1]),
                        "sensor_name": radar.name,
                        "frame_id": int(entry["frame_id"]),
                        "relative_velocity_mps": float(
                            metadata.get("relative_velocity_mps", 0.0)
                        ),
                        "ground_return": bool(
                            ego_point[2] < self.ground_z_m + 0.15
                        ),
                        "measurement_id": (
                            f"{radar.name}:{int(entry['frame_id'])}:{index}"
                        ),
                    }
                )

        return projected

    def project_camera_detections(
        self,
        perception_result,
        current_frame_id=None,
    ):
        """Yedi kameranın bbox tabanlarını kalibre zemin düzlemine taşır."""
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

                point = np.array(
                    [[position["x_m"], position["y_m"], self.ground_z_m]],
                    dtype=np.float64,
                )
                point = self.compensate_points(
                    point,
                    result.get("frame_id"),
                    current_frame_id,
                )[0]
                distance = math.hypot(float(point[0]), float(point[1]))
                confidence = float(detection.get("confidence", 0.0))
                uncertainty = 0.70 + 0.035 * distance

                projected.append(
                    {
                        "x_m": float(point[0]),
                        "y_m": float(point[1]),
                        "source": "camera",
                        "camera_name": camera.name,
                        "sensor_names": [camera.name],
                        "frame_ids": [int(result["frame_id"])],
                        "class_name": detection.get("class_name", "vehicle"),
                        "confidence": confidence,
                        "uncertainty_m": uncertainty,
                        "measurement_ids": [
                            self.camera_measurement_id(
                                camera.name,
                                result["frame_id"],
                                detection.get("bbox"),
                            )
                        ],
                    }
                )

        return projected

    def project_detection(self, camera, detection, image_width, image_height):
        """Tek bbox'ın alt orta pikselini H^-1 ile zemin konumuna çevirir."""
        bbox = detection.get("bbox")
        if bbox is None or len(bbox) != 4:
            return None

        calibration = self.calibrations.get_camera(camera.name)
        scale_x = calibration.width / max(1.0, float(image_width))
        scale_y = calibration.height / max(1.0, float(image_height))
        pixel_x = 0.5 * (float(bbox[0]) + float(bbox[2])) * scale_x
        pixel_y = float(bbox[3]) * scale_y
        position = calibration.pixel_to_ground_point(pixel_x, pixel_y)
        if position is None:
            return None

        forward_m, right_m = position
        if math.hypot(forward_m, right_m) > 100.0:
            return None
        return {"x_m": forward_m, "y_m": right_m}

    def camera_measurement_id(self, camera_name, frame_id, bbox):
        if bbox is None:
            box_text = "no-box"
        else:
            box_values = []
            for value in bbox:
                box_values.append(str(int(round(value))))
            box_text = ",".join(box_values)
        return f"{camera_name}:{int(frame_id)}:{box_text}"

    def compensate_points(self, points, old_frame_id, current_frame_id):
        if self.motion_compensator is None or current_frame_id is None:
            return np.asarray(points).copy()
        return self.motion_compensator.compensate_points(
            points,
            old_frame_id,
            current_frame_id,
        )

    def sensor_origin(self, sensor_name, sensor_snapshot, current_frame_id):
        """Sensör merkezini ölçüm anından güncel ego karesine taşır."""
        calibration = self.calibrations.get_sensor(sensor_name)
        if calibration is None:
            return (0.0, 0.0)

        frame_id = current_frame_id
        entry = sensor_snapshot.get(sensor_name)
        if entry is not None:
            frame_id = entry.get("frame_id", current_frame_id)

        origin = np.array(
            [[calibration.T[0], calibration.T[1], calibration.T[2]]],
            dtype=np.float64,
        )
        origin = self.compensate_points(origin, frame_id, current_frame_id)[0]
        return float(origin[0]), float(origin[1])

    def project_route(self, vehicle_state):
        """Dünya koordinatındaki referans rotayı güncel ego karesine taşır."""
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

"""Kamera, radar ve LiDAR ölçümlerini güven ağırlıklarıyla birleştirir."""

import math

import numpy as np

from carla_app.bev.association import (
    group_associated_measurements,
    measurement_covariance,
)
from carla_app.bev.clustering import cluster_summary, cluster_xy_points


class SensorFusion:
    """Aynı nesneyi gören sensörlerden tek belirsizlikli ölçüm üretir."""

    def build_fused_objects(
        self,
        camera_objects,
        radar_points,
        lidar_points,
        lidar_frame_id=None,
    ):
        measurements = []
        measurements.extend(camera_objects)
        measurements.extend(self.radar_measurements(radar_points))
        groups = group_associated_measurements(measurements)
        lidar_measurements = self.lidar_measurements(
            lidar_points,
            lidar_frame_id=lidar_frame_id,
        )
        used_lidar = self.attach_lidar(groups, lidar_measurements)
        for index, lidar in enumerate(lidar_measurements):
            if index not in used_lidar:
                groups.append([lidar])

        fused_objects = []
        for group in groups:
            fused_objects.append(self.fuse_group(group))
        return fused_objects

    def radar_measurements(self, radar_points):
        usable = []
        metadata = []
        for point in radar_points:
            if point.get("ground_return", False):
                continue
            usable.append((point["x_m"], point["y_m"]))
            metadata.append(point)

        if not usable:
            return []

        xy = np.asarray(usable, dtype=np.float64)
        clusters = cluster_xy_points(xy, cell_size_m=1.0, minimum_points=2)
        measurements = []

        for indices in clusters:
            summary = cluster_summary(xy, indices)
            if summary["length_m"] > 8.0 or summary["width_m"] > 5.0:
                continue

            velocities = []
            sensor_names = []
            frame_ids = []
            measurement_ids = []
            for index in indices:
                point = metadata[index]
                velocities.append(float(point["relative_velocity_mps"]))
                sensor_names.append(point["sensor_name"])
                frame_ids.append(int(point["frame_id"]))
                measurement_id = point.get("measurement_id")
                if measurement_id is None:
                    measurement_id = (
                        f"{point['sensor_name']}:{point['frame_id']}:{index}"
                    )
                measurement_ids.append(measurement_id)

            summary.update(
                {
                    "source": "radar",
                    "sensor_names": sorted(set(sensor_names)),
                    "frame_ids": sorted(set(frame_ids)),
                    "relative_velocity_mps": float(np.median(velocities)),
                    "confidence": min(0.92, 0.55 + 0.06 * len(indices)),
                    "uncertainty_m": 0.65,
                    "covariance_xy": np.eye(2, dtype=np.float64) * 0.65**2,
                    "class_name": "obstacle",
                    "measurement_ids": measurement_ids,
                }
            )
            measurements.append(summary)

        return measurements

    def lidar_measurements(self, lidar_points, lidar_frame_id=None):
        points = np.asarray(lidar_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return []

        # Araç yüzeyleri arasında küçük LiDAR boşlukları kalabildiği için
        # 80 cm hücre komşuluğu kullanılır; büyük duvarlar boyutla elenir.
        clusters = cluster_xy_points(points[:, :2], 0.80, minimum_points=8)
        measurements = []
        for indices in clusters:
            summary = cluster_summary(points[:, :2], indices)
            if summary["length_m"] > 14.0 or summary["width_m"] > 8.0:
                continue
            if summary["length_m"] < 0.15 and summary["width_m"] < 0.15:
                continue

            frame_ids = [] if lidar_frame_id is None else [int(lidar_frame_id)]
            measurement_ids = []
            if lidar_frame_id is not None:
                measurement_ids.append(
                    f"lidar_roof:{int(lidar_frame_id)}:{indices[0]}"
                )
            summary.update(
                {
                    "source": "lidar",
                    "sensor_names": ["lidar_roof"],
                    "frame_ids": frame_ids,
                    "confidence": min(0.96, 0.60 + 0.01 * len(indices)),
                    "uncertainty_m": 0.25,
                    "covariance_xy": np.eye(2, dtype=np.float64) * 0.25**2,
                    "class_name": "obstacle",
                    "measurement_ids": measurement_ids,
                }
            )
            measurements.append(summary)
        return measurements

    def attach_lidar(self, groups, lidar_measurements):
        """LiDAR kümesini yalnızca kamera/radar nesnesine yakınsa ekler."""
        used_lidar = set()
        for group in groups:
            center_x = 0.0
            center_y = 0.0
            for item in group:
                center_x += float(item["x_m"])
                center_y += float(item["y_m"])
            center_x /= len(group)
            center_y /= len(group)

            best_index = None
            best_distance = None
            for index, lidar in enumerate(lidar_measurements):
                if index in used_lidar:
                    continue
                distance = math.hypot(
                    float(lidar["x_m"]) - center_x,
                    float(lidar["y_m"]) - center_y,
                )
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_index = index

            if best_index is not None and best_distance <= 3.0:
                group.append(lidar_measurements[best_index])
                used_lidar.add(best_index)
        return used_lidar

    def fuse_group(self, group):
        """Ters varyans ağırlığıyla tek konum ve ortak güven değeri üretir."""
        information = np.zeros((2, 2), dtype=np.float64)
        information_vector = np.zeros(2, dtype=np.float64)
        missed_probability = 1.0
        sources = []
        sensor_names = []
        frame_ids = []
        measurement_ids = []
        velocities = []
        velocity_weights = []
        class_name = "obstacle"
        best_class_confidence = -1.0
        length_m = 0.0
        width_m = 0.0
        source_frames = {}

        for measurement in group:
            uncertainty = max(0.10, float(measurement["uncertainty_m"]))
            weight = 1.0 / (uncertainty * uncertainty)
            covariance = measurement_covariance(measurement)
            inverse_covariance = np.linalg.inv(covariance)
            position = np.array(
                [float(measurement["x_m"]), float(measurement["y_m"])],
                dtype=np.float64,
            )
            information += inverse_covariance
            information_vector += inverse_covariance @ position

            confidence = min(0.99, max(0.0, float(measurement["confidence"])))
            missed_probability *= 1.0 - confidence
            sources.append(measurement["source"])
            sensor_names.extend(measurement.get("sensor_names", []))
            frame_ids.extend(measurement.get("frame_ids", []))
            measurement_ids.extend(measurement.get("measurement_ids", []))
            if measurement.get("frame_ids"):
                source_frames[str(measurement["source"])] = max(
                    int(value) for value in measurement["frame_ids"]
                )

            if measurement["source"] == "camera" and confidence > best_class_confidence:
                class_name = measurement.get("class_name", "vehicle")
                best_class_confidence = confidence

            if "relative_velocity_mps" in measurement:
                velocities.append(float(measurement["relative_velocity_mps"]))
                velocity_weights.append(weight)

            if measurement["source"] == "lidar":
                length_m = max(length_m, float(measurement["length_m"]))
                width_m = max(width_m, float(measurement["width_m"]))

        relative_velocity = None
        if velocities:
            relative_velocity = float(
                np.average(velocities, weights=velocity_weights)
            )

        unique_sources = sorted(set(sources))
        unique_sensors = sorted(set(sensor_names))
        unique_frames_set = set()
        for value in frame_ids:
            unique_frames_set.add(int(value))
        unique_frames = sorted(unique_frames_set)
        unique_measurements = sorted(set(measurement_ids))
        fused_covariance = np.linalg.inv(information)
        fused_position = fused_covariance @ information_vector
        if unique_measurements:
            measurement_key = "|".join(unique_measurements)
        else:
            frame_texts = []
            for value in unique_frames:
                frame_texts.append(str(value))
            measurement_key = "|".join(
                [
                    ",".join(unique_sensors),
                    ",".join(frame_texts),
                    f"{fused_position[0]:.1f}",
                    f"{fused_position[1]:.1f}",
                ]
            )

        uncertainty = math.sqrt(
            float(np.max(np.linalg.eigvalsh(fused_covariance)))
        )
        return {
            "x_m": float(fused_position[0]),
            "y_m": float(fused_position[1]),
            "uncertainty_m": uncertainty,
            "covariance_xy": fused_covariance,
            "confidence": 1.0 - missed_probability,
            "class_name": class_name,
            "sources": unique_sources,
            "sensor_names": unique_sensors,
            "frame_ids": unique_frames,
            "relative_velocity_mps": relative_velocity,
            "length_m": length_m if length_m > 0.2 else 4.5,
            "width_m": width_m if width_m > 0.2 else 1.9,
            "measurement_key": measurement_key,
            "source_frames": source_frames,
        }

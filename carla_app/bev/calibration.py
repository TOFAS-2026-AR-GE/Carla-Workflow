"""Sensörlerin K, R ve T kalibrasyonlarını açık matrislerle oluşturur."""

import math

import numpy as np


def build_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """CARLA roll-pitch-yaw açılarını sensörden ego aracına döndürür."""
    roll = math.radians(float(roll_deg))
    pitch = math.radians(float(pitch_deg))
    yaw = math.radians(float(yaw_deg))

    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    cos_pitch = math.cos(pitch)
    sin_pitch = math.sin(pitch)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    return np.array(
        [
            [
                cos_pitch * cos_yaw,
                cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll,
                -cos_yaw * sin_pitch * cos_roll - sin_yaw * sin_roll,
            ],
            [
                cos_pitch * sin_yaw,
                sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll,
                -sin_yaw * sin_pitch * cos_roll + cos_yaw * sin_roll,
            ],
            [
                sin_pitch,
                -cos_pitch * sin_roll,
                cos_pitch * cos_roll,
            ],
        ],
        dtype=np.float64,
    )


def build_transform_matrix(transform):
    """Sensör yerel koordinatından ego koordinatına 4x4 matris üretir."""
    rotation = build_rotation_matrix(
        transform.rotation.roll,
        transform.rotation.pitch,
        transform.rotation.yaw,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[0, 3] = float(transform.location.x)
    matrix[1, 3] = float(transform.location.y)
    matrix[2, 3] = float(transform.location.z)
    return matrix


def build_camera_matrix(width, height, horizontal_fov_deg):
    """Kamera çözünürlüğü ve yatay FOV değerinden K matrisi üretir."""
    width = int(width)
    height = int(height)
    horizontal_fov_deg = float(horizontal_fov_deg)

    if width <= 0 or height <= 0:
        raise ValueError("Kamera çözünürlüğü sıfırdan büyük olmalı.")
    if not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("Kamera FOV değeri 1 ile 179 derece arasında olmalı.")

    focal_length = width / (
        2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0)
    )
    camera_matrix = np.eye(3, dtype=np.float64)
    camera_matrix[0, 0] = focal_length
    camera_matrix[1, 1] = focal_length
    camera_matrix[0, 2] = width / 2.0
    camera_matrix[1, 2] = height / 2.0
    return camera_matrix


class SensorCalibration:
    """Tek sensörün ego aracına göre dış kalibrasyonunu tutar."""

    def __init__(self, sensor_spec):
        self.name = sensor_spec.name
        self.sensor_to_ego = build_transform_matrix(sensor_spec.transform)
        self.ego_to_sensor = np.linalg.inv(self.sensor_to_ego)
        self.R = self.sensor_to_ego[:3, :3].copy()
        self.T = self.sensor_to_ego[:3, 3].copy()

    def sensor_points_to_ego(self, points):
        """Nx3 sensör noktasını ego aracının Nx3 koordinatına taşır."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 3:
            return np.empty((0, 3), dtype=np.float64)

        xyz = points[:, :3]
        return xyz @ self.R.T + self.T


class CameraCalibration(SensorCalibration):
    """Kamera iç kalibrasyonunu ve zemin homografisini birlikte tutar."""

    def __init__(self, camera_spec, ground_z_m):
        super().__init__(camera_spec)
        self.width = int(camera_spec.attributes["image_size_x"])
        self.height = int(camera_spec.attributes["image_size_y"])
        self.fov_deg = float(camera_spec.attributes["fov"])
        self.ground_z_m = float(ground_z_m)
        self.K = build_camera_matrix(self.width, self.height, self.fov_deg)

        # CARLA kamera eksenleri x-ileri, y-sağ, z-yukarıdır. OpenCV ise
        # görüntü izdüşümünde x-sağ, y-aşağı, z-ileri eksenlerini kullanır.
        self.carla_to_opencv = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.ground_to_image = self.build_ground_homography()
        self.image_to_ground = np.linalg.inv(self.ground_to_image)

    def build_ground_homography(self):
        """Ego zeminindeki [X,Y,1] noktasını görüntüye taşıyan H matrisi."""
        transform = self.ego_to_sensor
        ground_columns = np.zeros((3, 3), dtype=np.float64)
        ground_columns[:, 0] = transform[:3, 0]
        ground_columns[:, 1] = transform[:3, 1]
        ground_columns[:, 2] = (
            transform[:3, 2] * self.ground_z_m + transform[:3, 3]
        )
        camera_columns = self.carla_to_opencv @ ground_columns
        return self.K @ camera_columns

    def project_ground_points(self, ground_points):
        """Nx2 ego zemin noktasını piksele ve kamera derinliğine çevirir."""
        points = np.asarray(ground_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] < 2:
            empty = np.empty((0, 2), dtype=np.float64)
            return empty, np.empty(0, dtype=np.float64)

        homogeneous = np.column_stack(
            (points[:, 0], points[:, 1], np.ones(len(points)))
        )
        image_homogeneous = homogeneous @ self.ground_to_image.T
        depth = image_homogeneous[:, 2]
        pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
        valid = np.abs(depth) > 1e-8
        pixels[valid, 0] = image_homogeneous[valid, 0] / depth[valid]
        pixels[valid, 1] = image_homogeneous[valid, 1] / depth[valid]
        return pixels, depth

    def pixel_to_ground_point(self, pixel_x, pixel_y):
        """Bir görüntü pikselinin düz zemin üzerindeki ego konumunu bulur."""
        image_point = np.array(
            [float(pixel_x), float(pixel_y), 1.0],
            dtype=np.float64,
        )
        ground_point = self.image_to_ground @ image_point
        scale = ground_point[2]
        if abs(scale) < 1e-8:
            return None

        forward_m = ground_point[0] / scale
        right_m = ground_point[1] / scale
        if not math.isfinite(forward_m) or not math.isfinite(right_m):
            return None

        ego_point = np.array(
            [forward_m, right_m, self.ground_z_m, 1.0],
            dtype=np.float64,
        )
        sensor_point = self.ego_to_sensor @ ego_point
        if sensor_point[0] <= 0.1:
            return None
        return float(forward_m), float(right_m)


class CalibrationSet:
    """Yerleşimdeki bütün sensör kalibrasyonlarını adlarıyla hazırlar."""

    def __init__(self, layout):
        geometry = layout.vehicle_geometry
        ground_z_m = (
            float(geometry["bounding_box_center_z_m"])
            - float(geometry["half_height_m"])
        )
        self.ground_z_m = ground_z_m
        self.cameras = {}
        self.sensors = {}

        for sensor in layout.all_specs:
            calibration = SensorCalibration(sensor)
            self.sensors[sensor.name] = calibration

        for camera in layout.cameras:
            calibration = CameraCalibration(camera, ground_z_m)
            self.cameras[camera.name] = calibration
            self.sensors[camera.name] = calibration

    def get_sensor(self, sensor_name):
        return self.sensors.get(sensor_name)

    def get_camera(self, camera_name):
        return self.cameras.get(camera_name)

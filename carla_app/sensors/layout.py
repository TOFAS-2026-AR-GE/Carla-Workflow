from dataclasses import dataclass
from typing import Dict, List, Tuple

import carla


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: str
    blueprint_id: str
    transform: carla.Transform
    attributes: Dict[str, str]
    physical_sensor: str
    primary: bool = False

    def to_dict(self) -> Dict[str, object]:
        location = self.transform.location
        rotation = self.transform.rotation

        return {
            "name": self.name,
            "kind": self.kind,
            "physical_sensor": self.physical_sensor,
            "blueprint_id": self.blueprint_id,
            "primary": self.primary,
            "transform_vehicle_frame": {
                "x_m": float(location.x),
                "y_m": float(location.y),
                "z_m": float(location.z),
                "roll_deg": float(rotation.roll),
                "pitch_deg": float(rotation.pitch),
                "yaw_deg": float(rotation.yaw),
            },
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class SensorLayout:
    cameras: Tuple[SensorSpec, ...]
    lidar: SensorSpec
    gnss: SensorSpec
    imu: SensorSpec
    radars: Tuple[SensorSpec, ...]
    ultrasonics: Tuple[SensorSpec, ...]
    vehicle_geometry: Dict[str, float]

    @property
    def all_specs(self) -> Tuple[SensorSpec, ...]:
        return (
            self.cameras
            + (self.lidar, self.gnss, self.imu)
            + self.radars
            + self.ultrasonics
        )

    @property
    def sensor_names(self) -> List[str]:
        return [spec.name for spec in self.all_specs]

    @property
    def control_specs(self) -> Tuple[SensorSpec, ...]:
        """Sensors required by the live controller and perception viewer."""
        primary_camera = next(camera for camera in self.cameras if camera.primary)
        return primary_camera, self.front_radar

    @property
    def front_radar(self) -> SensorSpec:
        return next(radar for radar in self.radars if radar.name == "radar_front_long")

    @property
    def front_radar_geometry(self) -> Dict[str, float]:
        """Geometry needed to reject radar rays that intersect the road."""
        bottom_z = (
            self.vehicle_geometry["bounding_box_center_z_m"]
            - self.vehicle_geometry["half_height_m"]
        )
        return {
            "height_above_ground_m": float(
                self.front_radar.transform.location.z - bottom_z
            ),
            "pitch_deg": float(self.front_radar.transform.rotation.pitch),
        }

    @property
    def primary_camera_name(self) -> str:
        for camera in self.cameras:
            if camera.primary:
                return camera.name

        raise RuntimeError("Birincil kamera tanimlanmamis.")

    def to_manifest(self, vehicle_type_id: str) -> Dict[str, object]:
        return {
            "profile": "cost_effective_360_research_vehicle_v1",
            "vehicle_type_id": vehicle_type_id,
            "coordinate_system": {
                "x": "forward",
                "y": "right",
                "z": "up",
                "rotation_unit": "degree",
            },
            "vehicle_geometry": dict(self.vehicle_geometry),
            "primary_camera": self.primary_camera_name,
            "sensor_count": {
                "cameras": len(self.cameras),
                "lidars": 1,
                "automotive_radars": len(self.radars),
                "ultrasonics": len(self.ultrasonics),
                "gnss": 1,
                "imu": 1,
                "total": len(self.all_specs),
            },
            "simulation_note": (
                "CARLA'da yerlesik ultrasonik blueprint bulunmadigi icin fiziksel "
                "ultrasonikler, 4.5 m menzilli dusuk yogunluklu radar konileriyle "
                "simule edilmistir."
            ),
            "sensors": [spec.to_dict() for spec in self.all_specs],
        }


def _transform(
    x: float,
    y: float,
    z: float,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
) -> carla.Transform:
    return carla.Transform(
        carla.Location(
            x=float(x),
            y=float(y),
            z=float(z),
        ),
        carla.Rotation(
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        ),
    )


def _camera_attributes(
    width: int,
    height: int,
    fov: float,
) -> Dict[str, str]:
    return {
        "image_size_x": str(int(width)),
        "image_size_y": str(int(height)),
        "fov": f"{float(fov):.1f}",
        "sensor_tick": "0.0",
        "gamma": "2.2",
        "motion_blur_intensity": "0.0",
        "enable_postprocess_effects": "true",
    }


def _radar_attributes(
    horizontal_fov: float,
    vertical_fov: float,
    sensor_range: float,
    points_per_second: int,
) -> Dict[str, str]:
    return {
        "horizontal_fov": f"{float(horizontal_fov):.1f}",
        "vertical_fov": f"{float(vertical_fov):.1f}",
        "range": f"{float(sensor_range):.1f}",
        "points_per_second": str(int(points_per_second)),
        "sensor_tick": "0.0",
    }


def build_sensor_layout(
    vehicle,
    camera_width: int,
    camera_height: int,
    front_wide_fov: float,
    fixed_delta_seconds: float,
) -> SensorLayout:
    box = vehicle.bounding_box
    center = box.location
    extent = box.extent

    # CARLA bounding-box extent değerleri yarı boyutlardır.
    half_length = max(float(extent.x), 1.50)
    half_width = max(float(extent.y), 0.70)
    half_height = max(float(extent.z), 0.50)

    front_x = float(center.x) + half_length + 0.05
    rear_x = float(center.x) - half_length - 0.05

    left_y = float(center.y) - half_width - 0.04
    right_y = float(center.y) + half_width + 0.04

    bottom_z = float(center.z) - half_height
    top_z = float(center.z) + half_height

    windshield_x = float(center.x) + 0.42 * half_length
    windshield_z = top_z - 0.08

    side_camera_z = float(center.z) + 0.45 * half_height
    rear_camera_z = float(center.z) + 0.25 * half_height

    bumper_z = bottom_z + min(
        0.55,
        max(0.38, 0.52 * half_height),
    )

    # Uzun menzil radari birkac metre ondeki yolu degil, ufka yakin arac
    # govdelerini gormelidir. Kaput seviyesine yerlestirilir ve hafif yukari
    # bakar. Yukseklik secilen aracin boyutuna gore otomatik hesaplanir.
    front_radar_height_above_ground = min(
        1.15,
        max(0.85, 0.70 * (2.0 * half_height)),
    )
    front_radar_z = bottom_z + front_radar_height_above_ground

    roof_z = top_z + 0.16

    surround_fov = 100.0
    rear_fov = 110.0

    cameras = (
        SensorSpec(
            name="camera_front_wide",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                windshield_x,
                float(center.y),
                windshield_z,
                yaw=0.0,
                pitch=-4.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                front_wide_fov,
            ),
            physical_sensor="2 MP HDR wide-angle automotive camera",
            primary=True,
        ),
        SensorSpec(
            name="camera_front_narrow",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                windshield_x + 0.02,
                float(center.y),
                windshield_z + 0.02,
                yaw=0.0,
                pitch=-2.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                50.0,
            ),
            physical_sensor="2 MP HDR long-range narrow camera",
        ),
        SensorSpec(
            name="camera_front_left",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                float(center.x) + 0.48 * half_length,
                left_y,
                side_camera_z,
                yaw=-60.0,
                pitch=-4.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                surround_fov,
            ),
            physical_sensor="2 MP HDR surround camera",
        ),
        SensorSpec(
            name="camera_front_right",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                float(center.x) + 0.48 * half_length,
                right_y,
                side_camera_z,
                yaw=60.0,
                pitch=-4.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                surround_fov,
            ),
            physical_sensor="2 MP HDR surround camera",
        ),
        SensorSpec(
            name="camera_rear_left",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                float(center.x) - 0.38 * half_length,
                left_y,
                side_camera_z,
                yaw=-120.0,
                pitch=-4.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                surround_fov,
            ),
            physical_sensor="2 MP HDR surround camera",
        ),
        SensorSpec(
            name="camera_rear_right",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                float(center.x) - 0.38 * half_length,
                right_y,
                side_camera_z,
                yaw=120.0,
                pitch=-4.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                surround_fov,
            ),
            physical_sensor="2 MP HDR surround camera",
        ),
        SensorSpec(
            name="camera_rear_center",
            kind="camera",
            blueprint_id="sensor.camera.rgb",
            transform=_transform(
                rear_x,
                float(center.y),
                rear_camera_z,
                yaw=180.0,
                pitch=-5.0,
            ),
            attributes=_camera_attributes(
                camera_width,
                camera_height,
                rear_fov,
            ),
            physical_sensor="2 MP HDR rear camera",
        ),
    )

    lidar = SensorSpec(
        name="lidar_roof",
        kind="lidar",
        blueprint_id="sensor.lidar.ray_cast",
        transform=_transform(
            float(center.x),
            float(center.y),
            roof_z,
        ),
        attributes={
            "channels": "32",
            "range": "80.0",
            "points_per_second": "560000",
            "rotation_frequency": (f"{1.0 / fixed_delta_seconds:.3f}"),
            "upper_fov": "10.0",
            "lower_fov": "-25.0",
            "horizontal_fov": "360.0",
            "sensor_tick": "0.0",
            "dropoff_general_rate": "0.10",
        },
        physical_sensor=("single 360-degree 32-channel roof LiDAR"),
    )

    gnss = SensorSpec(
        name="gnss_roof",
        kind="gnss",
        blueprint_id="sensor.other.gnss",
        transform=_transform(
            float(center.x) - 0.10 * half_length,
            float(center.y),
            roof_z + 0.06,
        ),
        attributes={
            "sensor_tick": "0.0",
            "noise_lat_stddev": "0.0000005",
            "noise_lon_stddev": "0.0000005",
            "noise_alt_stddev": "0.10",
            "noise_seed": "41",
        },
        physical_sensor=("multi-band RTK-capable GNSS antenna/receiver"),
    )

    imu = SensorSpec(
        name="imu_cg",
        kind="imu",
        blueprint_id="sensor.other.imu",
        transform=_transform(
            float(center.x),
            float(center.y),
            float(center.z) - 0.15 * half_height,
        ),
        attributes={
            "sensor_tick": "0.0",
            "noise_accel_stddev_x": "0.02",
            "noise_accel_stddev_y": "0.02",
            "noise_accel_stddev_z": "0.02",
            "noise_gyro_stddev_x": "0.001",
            "noise_gyro_stddev_y": "0.001",
            "noise_gyro_stddev_z": "0.001",
            "noise_seed": "43",
        },
        physical_sensor=("automotive six-axis IMU near vehicle center of gravity"),
    )

    radars = (
        SensorSpec(
            name="radar_front_long",
            kind="radar",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                front_x,
                float(center.y),
                front_radar_z,
                yaw=0.0,
                pitch=2.0,
            ),
            attributes=_radar_attributes(
                horizontal_fov=30.0,
                vertical_fov=6.0,
                sensor_range=150.0,
                points_per_second=6000,
            ),
            physical_sensor="front 77 GHz long-range radar",
        ),
        SensorSpec(
            name="radar_front_left",
            kind="radar",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                float(center.x) + 0.88 * half_length,
                float(center.y) - 0.88 * half_width,
                bumper_z,
                yaw=-45.0,
            ),
            attributes=_radar_attributes(
                horizontal_fov=100.0,
                vertical_fov=20.0,
                sensor_range=70.0,
                points_per_second=3500,
            ),
            physical_sensor=("front-left 77 GHz corner radar"),
        ),
        SensorSpec(
            name="radar_front_right",
            kind="radar",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                float(center.x) + 0.88 * half_length,
                float(center.y) + 0.88 * half_width,
                bumper_z,
                yaw=45.0,
            ),
            attributes=_radar_attributes(
                horizontal_fov=100.0,
                vertical_fov=20.0,
                sensor_range=70.0,
                points_per_second=3500,
            ),
            physical_sensor=("front-right 77 GHz corner radar"),
        ),
        SensorSpec(
            name="radar_rear_left",
            kind="radar",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                float(center.x) - 0.88 * half_length,
                float(center.y) - 0.88 * half_width,
                bumper_z,
                yaw=-135.0,
            ),
            attributes=_radar_attributes(
                horizontal_fov=100.0,
                vertical_fov=20.0,
                sensor_range=70.0,
                points_per_second=3500,
            ),
            physical_sensor=("rear-left 77 GHz corner radar"),
        ),
        SensorSpec(
            name="radar_rear_right",
            kind="radar",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                float(center.x) - 0.88 * half_length,
                float(center.y) + 0.88 * half_width,
                bumper_z,
                yaw=135.0,
            ),
            attributes=_radar_attributes(
                horizontal_fov=100.0,
                vertical_fov=20.0,
                sensor_range=70.0,
                points_per_second=3500,
            ),
            physical_sensor=("rear-right 77 GHz corner radar"),
        ),
    )

    ultrasonic_positions = (
        (
            "ultrasonic_front_left_outer",
            front_x,
            -0.90,
            -70.0,
        ),
        (
            "ultrasonic_front_left_corner",
            front_x,
            -0.55,
            -40.0,
        ),
        (
            "ultrasonic_front_left_center",
            front_x,
            -0.20,
            -15.0,
        ),
        (
            "ultrasonic_front_right_center",
            front_x,
            0.20,
            15.0,
        ),
        (
            "ultrasonic_front_right_corner",
            front_x,
            0.55,
            40.0,
        ),
        (
            "ultrasonic_front_right_outer",
            front_x,
            0.90,
            70.0,
        ),
        (
            "ultrasonic_rear_left_outer",
            rear_x,
            -0.90,
            -110.0,
        ),
        (
            "ultrasonic_rear_left_corner",
            rear_x,
            -0.55,
            -140.0,
        ),
        (
            "ultrasonic_rear_left_center",
            rear_x,
            -0.20,
            -165.0,
        ),
        (
            "ultrasonic_rear_right_center",
            rear_x,
            0.20,
            165.0,
        ),
        (
            "ultrasonic_rear_right_corner",
            rear_x,
            0.55,
            140.0,
        ),
        (
            "ultrasonic_rear_right_outer",
            rear_x,
            0.90,
            110.0,
        ),
    )

    ultrasonics = tuple(
        SensorSpec(
            name=name,
            kind="ultrasonic",
            blueprint_id="sensor.other.radar",
            transform=_transform(
                x,
                float(center.y) + y_ratio * half_width,
                bumper_z - 0.05,
                yaw=yaw,
            ),
            attributes=_radar_attributes(
                horizontal_fov=50.0,
                vertical_fov=24.0,
                sensor_range=4.5,
                points_per_second=450,
            ),
            physical_sensor=("40 kHz parking ultrasonic transducer"),
        )
        for name, x, y_ratio, yaw in ultrasonic_positions
    )

    geometry = {
        "bounding_box_center_x_m": float(center.x),
        "bounding_box_center_y_m": float(center.y),
        "bounding_box_center_z_m": float(center.z),
        "length_m": 2.0 * half_length,
        "width_m": 2.0 * half_width,
        "height_m": 2.0 * half_height,
        "half_length_m": half_length,
        "half_width_m": half_width,
        "half_height_m": half_height,
    }

    return SensorLayout(
        cameras=cameras,
        lidar=lidar,
        gnss=gnss,
        imu=imu,
        radars=radars,
        ultrasonics=ultrasonics,
        vehicle_geometry=geometry,
    )

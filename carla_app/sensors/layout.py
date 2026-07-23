"""Araç üzerindeki sensörlerin konumunu ve CARLA ayarlarını tanımlar.

Bu dosya yalnızca sensör yerleşiminden sorumludur. Sensör oluşturma işlemi
``factory.py``, canlı veriyi saklama işlemi ise ``stream.py`` içindedir.
"""

import carla


class SensorSpec:
    """Tek bir sensörün adı, türü, konumu ve CARLA ayarları."""

    def __init__(
        self,
        name,
        kind,
        blueprint_id,
        transform,
        attributes,
        physical_sensor,
        primary=False,
    ):
        self.name = name
        self.kind = kind
        self.blueprint_id = blueprint_id
        self.transform = transform
        self.attributes = attributes
        self.physical_sensor = physical_sensor
        self.primary = primary

    def to_dict(self):
        """Sensör tanımını kayıt dosyasına uygun sözlüğe çevirir."""
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


class SensorLayout:
    """Araçta kullanılabilecek bütün gerçek CARLA sensörlerini tutar."""

    def __init__(self, cameras, lidar, gnss, imu, radars, vehicle_geometry):
        self.cameras = cameras
        self.lidar = lidar
        self.gnss = gnss
        self.imu = imu
        self.radars = radars
        self.vehicle_geometry = vehicle_geometry

        # Sık kullanılan değerler bir kez bulunur ve normal sınıf alanlarında
        # tutulur. Böylece bu sınıfı okuyan kişi gizli özellik yöntemlerini
        # takip etmek zorunda kalmaz.
        self.all_specs = self.cameras + (
            self.lidar,
            self.gnss,
            self.imu,
        ) + self.radars

        self.sensor_names = []
        for sensor in self.all_specs:
            self.sensor_names.append(sensor.name)

        primary_camera = None
        for camera in self.cameras:
            if camera.primary:
                primary_camera = camera
                break
        if primary_camera is None:
            raise RuntimeError("Birincil kamera tanımlanmamış.")

        self.primary_camera_name = primary_camera.name
        self.front_radar = None
        for radar in self.radars:
            if radar.name == "radar_front_long":
                self.front_radar = radar
                break
        if self.front_radar is None:
            raise RuntimeError("Ön uzun menzil radarı tanımlanmamış.")

        self.control_specs = (primary_camera, self.front_radar)
        bottom_z = (
            self.vehicle_geometry["bounding_box_center_z_m"]
            - self.vehicle_geometry["half_height_m"]
        )
        self.front_radar_geometry = {
            "height_above_ground_m": float(
                self.front_radar.transform.location.z - bottom_z
            ),
            "pitch_deg": float(self.front_radar.transform.rotation.pitch),
        }

    def to_manifest(self, vehicle_type_id):
        """Sensör yerleşimini veri kaydının kalibrasyon dosyasına hazırlar."""
        sensors = []
        for sensor in self.all_specs:
            sensors.append(sensor.to_dict())

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
                "gnss": 1,
                "imu": 1,
                "total": len(self.all_specs),
            },
            "sensors": sensors,
        }


def make_transform(x, y, z, yaw=0.0, pitch=0.0, roll=0.0):
    """Araç merkezine göre verilen değerlerden CARLA dönüşümü oluşturur."""
    location = carla.Location(x=float(x), y=float(y), z=float(z))
    rotation = carla.Rotation(
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
    )
    return carla.Transform(location, rotation)


def camera_attributes(width, height, fov):
    """Bütün RGB kameraların ortak CARLA ayarları."""
    return {
        "image_size_x": str(int(width)),
        "image_size_y": str(int(height)),
        "fov": f"{float(fov):.1f}",
        "sensor_tick": "0.0",
        "gamma": "2.2",
        "motion_blur_intensity": "0.0",
        "enable_postprocess_effects": "true",
    }


def radar_attributes(horizontal_fov, vertical_fov, sensor_range, point_count):
    """Bütün radarların ortak CARLA ayarları."""
    return {
        "horizontal_fov": f"{float(horizontal_fov):.1f}",
        "vertical_fov": f"{float(vertical_fov):.1f}",
        "range": f"{float(sensor_range):.1f}",
        "points_per_second": str(int(point_count)),
        "sensor_tick": "0.0",
    }


def make_camera(
    name,
    x,
    y,
    z,
    yaw,
    pitch,
    fov,
    width,
    height,
    physical_sensor,
    primary=False,
):
    """Tekrarlanan kamera tanımlarını açık ve kısa biçimde oluşturur."""
    return SensorSpec(
        name=name,
        kind="camera",
        blueprint_id="sensor.camera.rgb",
        transform=make_transform(x, y, z, yaw=yaw, pitch=pitch),
        attributes=camera_attributes(width, height, fov),
        physical_sensor=physical_sensor,
        primary=primary,
    )


def make_radar(
    name,
    x,
    y,
    z,
    yaw,
    pitch,
    horizontal_fov,
    vertical_fov,
    sensor_range,
    point_count,
    physical_sensor,
):
    """Tekrarlanan radar tanımlarını açık ve kısa biçimde oluşturur."""
    return SensorSpec(
        name=name,
        kind="radar",
        blueprint_id="sensor.other.radar",
        transform=make_transform(x, y, z, yaw=yaw, pitch=pitch),
        attributes=radar_attributes(
            horizontal_fov,
            vertical_fov,
            sensor_range,
            point_count,
        ),
        physical_sensor=physical_sensor,
    )


def build_sensor_layout(
    vehicle,
    camera_width,
    camera_height,
    front_wide_fov,
    fixed_delta_seconds,
    surround_camera_width=None,
    surround_camera_height=None,
):
    """Araç boyutunu okuyup bütün sensörlerin yerleşimini oluşturur."""
    if fixed_delta_seconds <= 0.0:
        raise ValueError("Sensör zaman adımı sıfırdan büyük olmalı.")
    if surround_camera_width is None:
        surround_camera_width = camera_width
    if surround_camera_height is None:
        surround_camera_height = camera_height

    box = vehicle.bounding_box
    center = box.location
    extent = box.extent

    # CARLA extent değerleri tam boy değil, aracın yarı boyutlarıdır.
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
    bumper_z = bottom_z + min(0.55, max(0.38, 0.52 * half_height))
    roof_z = top_z + 0.16

    # Ön radar kaput seviyesinde ve iki derece yukarı bakar. Bu yerleşim,
    # aracın hemen önündeki yol yüzeyinden gelen yanlış dönüşleri azaltır.
    front_radar_height = min(1.15, max(0.85, 0.70 * (2.0 * half_height)))
    front_radar_z = bottom_z + front_radar_height

    cameras = (
        make_camera(
            "camera_front_wide",
            1.5,
            0.0,
            2.4,
            yaw=0.0,
            pitch=0.0,
            fov=front_wide_fov,
            width=camera_width,
            height=camera_height,
            physical_sensor="2 MP HDR geniş açılı otomotiv kamerası",
            primary=True,
        ),
        make_camera(
            "camera_front_narrow",
            windshield_x + 0.02,
            center.y,
            windshield_z + 0.02,
            yaw=0.0,
            pitch=-2.0,
            fov=50.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR uzun menzil ön kamerası",
        ),
        make_camera(
            "camera_front_left",
            center.x + 0.48 * half_length,
            left_y,
            side_camera_z,
            yaw=-60.0,
            pitch=-4.0,
            fov=100.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR çevre görüş kamerası",
        ),
        make_camera(
            "camera_front_right",
            center.x + 0.48 * half_length,
            right_y,
            side_camera_z,
            yaw=60.0,
            pitch=-4.0,
            fov=100.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR çevre görüş kamerası",
        ),
        make_camera(
            "camera_rear_left",
            center.x - 0.38 * half_length,
            left_y,
            side_camera_z,
            yaw=-120.0,
            pitch=-4.0,
            fov=100.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR çevre görüş kamerası",
        ),
        make_camera(
            "camera_rear_right",
            center.x - 0.38 * half_length,
            right_y,
            side_camera_z,
            yaw=120.0,
            pitch=-4.0,
            fov=100.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR çevre görüş kamerası",
        ),
        make_camera(
            "camera_rear_center",
            rear_x,
            center.y,
            rear_camera_z,
            yaw=180.0,
            pitch=-5.0,
            fov=110.0,
            width=surround_camera_width,
            height=surround_camera_height,
            physical_sensor="2 MP HDR arka görüş kamerası",
        ),
    )

    lidar = SensorSpec(
        name="lidar_roof",
        kind="lidar",
        blueprint_id="sensor.lidar.ray_cast",
        transform=make_transform(center.x, center.y, roof_z),
        attributes={
            "channels": "32",
            "range": "80.0",
            "points_per_second": "560000",
            "rotation_frequency": f"{1.0 / fixed_delta_seconds:.3f}",
            "upper_fov": "10.0",
            "lower_fov": "-25.0",
            "horizontal_fov": "360.0",
            "sensor_tick": "0.0",
            "dropoff_general_rate": "0.10",
        },
        physical_sensor="Tavan tipi 32 kanallı 360 derece LiDAR",
    )

    gnss = SensorSpec(
        name="gnss_roof",
        kind="gnss",
        blueprint_id="sensor.other.gnss",
        transform=make_transform(
            center.x - 0.10 * half_length,
            center.y,
            roof_z + 0.06,
        ),
        attributes={
            "sensor_tick": "0.0",
            "noise_lat_stddev": "0.0000005",
            "noise_lon_stddev": "0.0000005",
            "noise_alt_stddev": "0.10",
            "noise_seed": "41",
        },
        physical_sensor="Çok bantlı RTK destekli GNSS alıcısı",
    )

    imu = SensorSpec(
        name="imu_cg",
        kind="imu",
        blueprint_id="sensor.other.imu",
        transform=make_transform(center.x, center.y, center.z - 0.15 * half_height),
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
        physical_sensor="Araç ağırlık merkezine yakın altı eksenli IMU",
    )

    radars = (
        make_radar(
            "radar_front_long",
            front_x,
            center.y,
            front_radar_z,
            yaw=0.0,
            pitch=2.0,
            horizontal_fov=30.0,
            vertical_fov=6.0,
            sensor_range=150.0,
            point_count=6000,
            physical_sensor="Ön 77 GHz uzun menzil radarı",
        ),
        make_radar(
            "radar_front_left",
            center.x + 0.88 * half_length,
            center.y - 0.88 * half_width,
            bumper_z,
            yaw=-45.0,
            pitch=0.0,
            horizontal_fov=100.0,
            vertical_fov=20.0,
            sensor_range=70.0,
            point_count=3500,
            physical_sensor="Ön sol 77 GHz köşe radarı",
        ),
        make_radar(
            "radar_front_right",
            center.x + 0.88 * half_length,
            center.y + 0.88 * half_width,
            bumper_z,
            yaw=45.0,
            pitch=0.0,
            horizontal_fov=100.0,
            vertical_fov=20.0,
            sensor_range=70.0,
            point_count=3500,
            physical_sensor="Ön sağ 77 GHz köşe radarı",
        ),
        make_radar(
            "radar_rear_left",
            center.x - 0.88 * half_length,
            center.y - 0.88 * half_width,
            bumper_z,
            yaw=-135.0,
            pitch=0.0,
            horizontal_fov=100.0,
            vertical_fov=20.0,
            sensor_range=70.0,
            point_count=3500,
            physical_sensor="Arka sol 77 GHz köşe radarı",
        ),
        make_radar(
            "radar_rear_right",
            center.x - 0.88 * half_length,
            center.y + 0.88 * half_width,
            bumper_z,
            yaw=135.0,
            pitch=0.0,
            horizontal_fov=100.0,
            vertical_fov=20.0,
            sensor_range=70.0,
            point_count=3500,
            physical_sensor="Arka sağ 77 GHz köşe radarı",
        ),
    )

    vehicle_geometry = {
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
        vehicle_geometry=vehicle_geometry,
    )

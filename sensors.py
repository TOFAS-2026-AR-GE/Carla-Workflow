import numpy as np
import carla

def spawn_rgb_camera(world: carla.World, vehicle: carla.Vehicle, sync, width=800, height=600, fov=90):
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    
    transform = carla.Transform(carla.Location(
        x=1.5,
        z=1.6
    ))
    
    camera = world.spawn_actor(
        blueprint=cam_bp,
        transform=transform,
        attach_to=vehicle
    )
    
    camera.listen(lambda image: sync.push("rgb_camera", image.frame, image))
    print("[OK] RGB camera spawned.")
    return camera

def spawn_semantic_camera(world: carla.World, vehicle: carla.Vehicle, sync, width=800, height=600, fov=90):

  bp_lib = world.get_blueprint_library()
  seg_bp = bp_lib.find("sensor.camera.semantic_segmentation")
  seg_bp.set_attribute("image_size_x", str(width))
  seg_bp.set_attribute("image_size_y", str(height))
  seg_bp.set_attribute("fov", str(fov))

  transform = carla.Transform(carla.Location(x=1.5, z=1.6))
  camera = world.spawn_actor(seg_bp, transform, attach_to=vehicle)

  camera.listen(lambda image: sync.push("semantic_camera", image.frame, image))
  print("[OK] Semantic segmentation camera spawned.")
  return camera


def spawn_lidar(world: carla.World, vehicle: carla.Vehicle, sync,
                 channels=32, range_m=50, points_per_second=200000, rotation_frequency=20):
  bp_lib = world.get_blueprint_library()
  lidar_bp = bp_lib.find("sensor.lidar.ray_cast")
  lidar_bp.set_attribute("channels", str(channels))
  lidar_bp.set_attribute("range", str(range_m))
  lidar_bp.set_attribute("points_per_second", str(points_per_second))
  lidar_bp.set_attribute("rotation_frequency", str(rotation_frequency))

  transform = carla.Transform(carla.Location(x=0.0, z=2.4))
  lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)

  lidar.listen(lambda cloud: sync.push("lidar", cloud.frame, cloud))
  print("[OK] LiDAR spawned.")
  return lidar

def spawn_gnss(world: carla.World, vehicle: carla.Vehicle, sync):
  bp_lib = world.get_blueprint_library()
  gnss_bp = bp_lib.find("sensor.other.gnss")
  transform = carla.Transform(carla.Location(x=0.0, z=1.0))
  gnss = world.spawn_actor(gnss_bp, transform, attach_to=vehicle)

  gnss.listen(lambda data: sync.push("gnss", data.frame, data))
  print("[OK] GNSS spawned.")
  return gnss


def spawn_imu(world: carla.World, vehicle: carla.Vehicle, sync):
  bp_lib = world.get_blueprint_library()
  imu_bp = bp_lib.find("sensor.other.imu")
  transform = carla.Transform(carla.Location(x=0.0, z=1.0))
  imu = world.spawn_actor(imu_bp, transform, attach_to=vehicle)

  imu.listen(lambda data: sync.push("imu", data.frame, data))
  print("[OK] IMU spawned.")
  return imu

def spawn_collision_sensor(world: carla.World, vehicle: carla.Vehicle, on_collision):

  bp_lib = world.get_blueprint_library()
  col_bp = bp_lib.find("sensor.other.collision")
  transform = carla.Transform(carla.Location(x=0.0, z=0.0))
  collision_sensor = world.spawn_actor(col_bp, transform, attach_to=vehicle)

  collision_sensor.listen(on_collision)
  print("[OK] Collision sensor spawned.")
  return collision_sensor

def rgb_image_to_array(image: carla.Image) -> np.ndarray:
  """CARLA carla.Image -> (H, W, 3) BGR numpy array."""
  raw = np.frombuffer(image.raw_data, dtype=np.uint8)
  raw = raw.reshape((image.height, image.width, 4))  # BGRA
  return raw[:, :, :3]


def semantic_image_to_labels(image: carla.Image) -> np.ndarray:
  """
  Semantic segmentation goruntusunde sinif ID'si Kirmizi (R) kanalinda
  saklanir (CARLA'nin dokumantasyonundaki convention). (H, W) label array
  dondurur. Onemli ID'ler: 6=RoadLine, 7=Road, 10=Vehicle, 4=Pedestrian.
  """
  raw = np.frombuffer(image.raw_data, dtype=np.uint8)
  raw = raw.reshape((image.height, image.width, 4))
  return raw[:, :, 2]  # BGRA -> R kanali index 2

def lidar_cloud_to_array(cloud: carla.LidarMeasurement) -> np.ndarray:
  """CARLA lidar point cloud -> (N, 4) numpy array [x, y, z, intensity]."""
  points = np.frombuffer(cloud.raw_data, dtype=np.float32)
  return points.reshape((-1, 4))

def spawn_radar(
    world,
    vehicle,
    sync,
    sensor_name,
    transform,
    range_m,
    horizontal_fov,
    vertical_fov,
    points_per_second=3000,
):
    """
    Araca tek bir radar bağlar.

    sensor_name örnekleri:
        radar_front
        radar_rear
        radar_left
        radar_right
    """

    blueprint_library = world.get_blueprint_library()
    radar_bp = blueprint_library.find("sensor.other.radar")

    radar_bp.set_attribute(
        "range",
        str(range_m),
    )

    radar_bp.set_attribute(
        "horizontal_fov",
        str(horizontal_fov),
    )

    radar_bp.set_attribute(
        "vertical_fov",
        str(vertical_fov),
    )

    radar_bp.set_attribute(
        "points_per_second",
        str(points_per_second),
    )

    # 0.0 olduğu için radar her simülasyon frame'inde veri üretir.
    radar_bp.set_attribute(
        "sensor_tick",
        "0.0",
    )

    radar = world.spawn_actor(
        blueprint=radar_bp,
        transform=transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    radar.listen(
        lambda radar_data: sync.push(
            sensor_name,
            radar_data.frame,
            radar_data,
        )
    )

    print(f"[OK] {sensor_name} oluşturuldu.")

    return radar


def spawn_four_radars(world, vehicle, sync):
    """
    Ego araca dört radar bağlar:

        1. Ön radar
        2. Arka radar
        3. Sol kör nokta radarı
        4. Sağ kör nokta radarı

    Radar konumları aracın boyutuna göre otomatik belirlenir.
    """

    # Aracın yarı uzunluğu, yarı genişliği ve yarı yüksekliği.
    vehicle_extent = vehicle.bounding_box.extent

    vehicle_half_length = vehicle_extent.x
    vehicle_half_width = vehicle_extent.y
    vehicle_half_height = vehicle_extent.z

    # Radarın yerden yüksekliği.
    # Bazı araçlar alçak olabileceği için minimum 0.70 metre kullanıyoruz.
    radar_height = max(
        0.70,
        vehicle_half_height,
    )

    # --------------------------------------------------------
    # Ön radar
    # --------------------------------------------------------

    front_transform = carla.Transform(
        carla.Location(
            x=vehicle_half_length + 0.10,
            y=0.0,
            z=radar_height,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
    )

    front_radar = spawn_radar(
        world=world,
        vehicle=vehicle,
        sync=sync,
        sensor_name="radar_front",
        transform=front_transform,
        range_m=60,
        horizontal_fov=70,
        vertical_fov=15,
        points_per_second=5000,
    )

    # --------------------------------------------------------
    # Arka radar
    # --------------------------------------------------------

    rear_transform = carla.Transform(
        carla.Location(
            x=-(vehicle_half_length + 0.10),
            y=0.0,
            z=radar_height,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=180.0,
            roll=0.0,
        ),
    )

    rear_radar = spawn_radar(
        world=world,
        vehicle=vehicle,
        sync=sync,
        sensor_name="radar_rear",
        transform=rear_transform,
        range_m=40,
        horizontal_fov=90,
        vertical_fov=15,
        points_per_second=3000,
    )

    # --------------------------------------------------------
    # Sol kör nokta radarı
    # --------------------------------------------------------

    left_transform = carla.Transform(
        carla.Location(
            # Yan radarları biraz arkaya koyuyoruz.
            x=-vehicle_half_length * 0.40,
            y=-(vehicle_half_width + 0.10),
            z=radar_height,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=-90.0,
            roll=0.0,
        ),
    )

    left_radar = spawn_radar(
        world=world,
        vehicle=vehicle,
        sync=sync,
        sensor_name="radar_left",
        transform=left_transform,
        range_m=30,
        horizontal_fov=100,
        vertical_fov=20,
        points_per_second=3000,
    )

    # --------------------------------------------------------
    # Sağ kör nokta radarı
    # --------------------------------------------------------

    right_transform = carla.Transform(
        carla.Location(
            x=-vehicle_half_length * 0.40,
            y=vehicle_half_width + 0.10,
            z=radar_height,
        ),
        carla.Rotation(
            pitch=0.0,
            yaw=90.0,
            roll=0.0,
        ),
    )

    right_radar = spawn_radar(
        world=world,
        vehicle=vehicle,
        sync=sync,
        sensor_name="radar_right",
        transform=right_transform,
        range_m=30,
        horizontal_fov=100,
        vertical_fov=20,
        points_per_second=3000,
    )

    # İsimleriyle birlikte döndürüyoruz.
    return {
        "radar_front": front_radar,
        "radar_rear": rear_radar,
        "radar_left": left_radar,
        "radar_right": right_radar,
    }
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
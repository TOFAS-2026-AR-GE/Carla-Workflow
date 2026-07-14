import carla

from carla_app.sensors.processors import image_to_rgb


RADAR_TRANSFORMS = {
    "radar_front": carla.Transform(carla.Location(x=2.2, z=1.0)),
    "radar_rear": carla.Transform(
        carla.Location(x=-2.2, z=1.0), carla.Rotation(yaw=180)
    ),
    "radar_left": carla.Transform(
        carla.Location(z=1.0), carla.Rotation(yaw=-90)
    ),
    "radar_right": carla.Transform(
        carla.Location(z=1.0), carla.Rotation(yaw=90)
    ),
}


def _spawn(world, vehicle, blueprint_id, transform):
    blueprint = world.get_blueprint_library().find(blueprint_id)
    return world.spawn_actor(blueprint, transform, attach_to=vehicle)


def spawn_camera(world, vehicle, sync, stream, width, height, fov):
    blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
    blueprint.set_attribute("image_size_x", str(width))
    blueprint.set_attribute("image_size_y", str(height))
    blueprint.set_attribute("fov", str(fov))
    blueprint.set_attribute("sensor_tick", "0.0")

    camera = world.spawn_actor(
        blueprint,
        carla.Transform(carla.Location(x=1.5, z=1.6)),
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )

    def callback(image):
        sync.push("rgb_camera", image.frame, image)
        stream.push(image.frame, image_to_rgb(image))

    camera.listen(callback)
    return camera


def spawn_lidar(world, vehicle, sync):
    blueprint = world.get_blueprint_library().find("sensor.lidar.ray_cast")
    blueprint.set_attribute("channels", "32")
    blueprint.set_attribute("range", "50")
    blueprint.set_attribute("points_per_second", "200000")
    blueprint.set_attribute("rotation_frequency", "20")
    lidar = world.spawn_actor(
        blueprint,
        carla.Transform(carla.Location(z=2.2)),
        attach_to=vehicle,
    )
    lidar.listen(lambda data: sync.push("lidar", data.frame, data))
    return lidar


def spawn_gnss(world, vehicle, sync):
    sensor = _spawn(
        world,
        vehicle,
        "sensor.other.gnss",
        carla.Transform(carla.Location(z=1.5)),
    )
    sensor.listen(lambda data: sync.push("gnss", data.frame, data))
    return sensor


def spawn_imu(world, vehicle, sync):
    sensor = _spawn(world, vehicle, "sensor.other.imu", carla.Transform())
    sensor.listen(lambda data: sync.push("imu", data.frame, data))
    return sensor


def spawn_radars(world, vehicle, sync):
    actors = {}
    for name, transform in RADAR_TRANSFORMS.items():
        blueprint = world.get_blueprint_library().find("sensor.other.radar")
        blueprint.set_attribute("horizontal_fov", "35")
        blueprint.set_attribute("vertical_fov", "20")
        blueprint.set_attribute("range", "50")
        radar = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        radar.listen(lambda data, sensor_name=name: sync.push(sensor_name, data.frame, data))
        actors[name] = radar
    return actors

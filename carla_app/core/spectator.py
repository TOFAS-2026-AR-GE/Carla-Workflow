import carla

FOLLOW_DISTANCE = 6.0
FOLLOW_HEIGHT = 2.5
FOLLOW_PITCH = -10.0


def update_spectator(world, vehicle):
    transform = vehicle.get_transform()
    forward = transform.get_forward_vector()
    location = carla.Location(
        x=transform.location.x - forward.x * FOLLOW_DISTANCE,
        y=transform.location.y - forward.y * FOLLOW_DISTANCE,
        z=transform.location.z + FOLLOW_HEIGHT,
    )
    rotation = carla.Rotation(
        pitch=FOLLOW_PITCH,
        yaw=transform.rotation.yaw,
    )
    world.get_spectator().set_transform(carla.Transform(location, rotation))

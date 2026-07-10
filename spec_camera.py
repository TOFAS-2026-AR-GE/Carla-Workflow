import carla


FOLLOW_DISTANCE_METERS = 8.0
FOLLOW_HEIGHT_METERS = 4.0
CAMERA_PITCH_DEGREES = -15.0

def update_spectator_camera(world: carla.World, vehicle: carla.Vehicle) -> None:

    spectator = world.get_spectator()

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation

    forward_vector = vehicle_transform.get_forward_vector()

    camera_location = carla.Location(
        x=vehicle_location.x - forward_vector.x * FOLLOW_DISTANCE_METERS,
        y=vehicle_location.y - forward_vector.y * FOLLOW_DISTANCE_METERS,
        z=vehicle_location.z + FOLLOW_HEIGHT_METERS,
    )

    camera_rotation = carla.Rotation(
        pitch=CAMERA_PITCH_DEGREES,
        yaw=vehicle_rotation.yaw,
        roll=0.0,
    )

    camera_transform = carla.Transform(camera_location, camera_rotation)
    spectator.set_transform(camera_transform)
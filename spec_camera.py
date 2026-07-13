import carla


FOLLOW_DISTANCE_METERS = 6.0
FOLLOW_HEIGHT_METERS = 2.5
CAMERA_PITCH_DEGREES = -10.0
LOCATION_SMOOTHING = 0.20
ROTATION_SMOOTHING = 0.15


def _lerp(start: float, end: float, alpha: float) -> float:
    return start + (end - start) * alpha


def _normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


def _lerp_angle(start: float, end: float, alpha: float) -> float:
    angle_delta = _normalize_angle(end - start)
    return _normalize_angle(start + angle_delta * alpha)


def _smooth_location(
    current: carla.Location,
    target: carla.Location,
) -> carla.Location:
    return carla.Location(
        x=_lerp(current.x, target.x, LOCATION_SMOOTHING),
        y=_lerp(current.y, target.y, LOCATION_SMOOTHING),
        z=_lerp(current.z, target.z, LOCATION_SMOOTHING),
    )


def _smooth_rotation(
    current: carla.Rotation,
    target: carla.Rotation,
) -> carla.Rotation:
    return carla.Rotation(
        pitch=_lerp_angle(
            current.pitch,
            target.pitch,
            ROTATION_SMOOTHING,
        ),
        yaw=_lerp_angle(
            current.yaw,
            target.yaw,
            ROTATION_SMOOTHING,
        ),
        roll=0.0,
    )


def update_spectator_camera(
    world: carla.World,
    vehicle: carla.Vehicle,
) -> None:
    spectator = world.get_spectator()
    current_transform = spectator.get_transform()

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_rotation = vehicle_transform.rotation
    forward_vector = vehicle_transform.get_forward_vector()

    target_location = carla.Location(
        x=vehicle_location.x - forward_vector.x * FOLLOW_DISTANCE_METERS,
        y=vehicle_location.y - forward_vector.y * FOLLOW_DISTANCE_METERS,
        z=vehicle_location.z + FOLLOW_HEIGHT_METERS,
    )

    target_rotation = carla.Rotation(
        pitch=CAMERA_PITCH_DEGREES,
        yaw=vehicle_rotation.yaw,
        roll=0.0,
    )

    camera_location = _smooth_location(
        current=current_transform.location,
        target=target_location,
    )

    camera_rotation = _smooth_rotation(
        current=current_transform.rotation,
        target=target_rotation,
    )

    spectator.set_transform(
        carla.Transform(
            camera_location,
            camera_rotation,
        )
    )

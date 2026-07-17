"""Sensör tanımlarını CARLA aktörlerine dönüştürür ve veriyi yönlendirir."""

import carla

from carla_app.sensors.processors import (
    gnss_to_dict,
    image_to_rgb,
    imu_to_dict,
    lidar_to_array,
    radar_to_list,
)


def set_supported_attributes(blueprint, attributes):
    """Yalnızca CARLA planının desteklediği sensör ayarlarını uygular."""
    for name, value in attributes.items():
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, str(value))


def spawn_sensor(world, vehicle, spec):
    """Bir sensörü araca sabit biçimde bağlar."""
    blueprint = world.get_blueprint_library().find(spec.blueprint_id)
    set_supported_attributes(blueprint, spec.attributes)

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute("role_name", spec.name)

    return world.spawn_actor(
        blueprint,
        spec.transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def start_sensor_listener(
    actor,
    spec,
    sync,
    camera_stream,
    radar_stream,
    live_stream=None,
):
    """Sensör verisini kamera, radar veya kayıt akışına gönderir."""
    if spec.kind == "camera":

        def camera_callback(image):
            if sync is not None:
                sync.push(spec.name, image.frame, image)
            if spec.primary or live_stream is not None:
                rgb_image = image_to_rgb(image)
                if spec.primary:
                    camera_stream.push(image.frame, rgb_image)
                if live_stream is not None:
                    live_stream.push(spec.name, image.frame, rgb_image)

        actor.listen(camera_callback)
        return

    if spec.kind == "radar":

        def radar_callback(data):
            if sync is not None:
                sync.push(spec.name, data.frame, data)
            points = radar_to_list(data)
            radar_stream.push(spec.name, data.frame, points)
            if live_stream is not None:
                live_stream.push(spec.name, data.frame, points)

        actor.listen(radar_callback)
        return

    if spec.kind == "lidar":

        def lidar_callback(data):
            if sync is not None:
                sync.push(spec.name, data.frame, data)
            if live_stream is not None:
                live_stream.push(spec.name, data.frame, lidar_to_array(data))

        actor.listen(lidar_callback)
        return

    if spec.kind == "gnss":

        def gnss_callback(data):
            if sync is not None:
                sync.push(spec.name, data.frame, data)
            if live_stream is not None:
                live_stream.push(spec.name, data.frame, gnss_to_dict(data))

        actor.listen(gnss_callback)
        return

    if spec.kind == "imu":

        def imu_callback(data):
            if sync is not None:
                sync.push(spec.name, data.frame, data)
            if live_stream is not None:
                live_stream.push(spec.name, data.frame, imu_to_dict(data))

        actor.listen(imu_callback)
        return

    if sync is not None:

        def recording_callback(data):
            sync.push(spec.name, data.frame, data)

        actor.listen(recording_callback)


def spawn_layout(
    world,
    vehicle,
    layout,
    sync,
    camera_stream,
    radar_stream,
    live_stream=None,
    specs=None,
):
    """Seçilen sensörleri oluşturur; hata olursa oluşturulanları temizler."""
    actors = []

    try:
        active_specs = layout.all_specs if specs is None else tuple(specs)

        for spec in active_specs:
            actor = spawn_sensor(world, vehicle, spec)
            start_sensor_listener(
                actor,
                spec,
                sync,
                camera_stream,
                radar_stream,
                live_stream,
            )
            actors.append(actor)
            print(
                f"[OK] Sensör: {spec.name:<32} "
                f"tip={spec.kind:<10} "
                f"blueprint={spec.blueprint_id}"
            )

        return actors

    except Exception:
        for actor in actors:
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        raise

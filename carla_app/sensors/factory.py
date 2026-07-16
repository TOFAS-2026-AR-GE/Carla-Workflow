from typing import Dict, List

import carla

from carla_app.sensors.layout import (
    SensorLayout,
    SensorSpec,
)
from carla_app.sensors.processors import image_to_rgb, radar_to_list


def _set_supported_attributes(
    blueprint,
    attributes: Dict[str, str],
) -> None:
    for name, value in attributes.items():
        if blueprint.has_attribute(name):
            blueprint.set_attribute(name, str(value))


def _spawn_actor(
    world,
    vehicle,
    spec: SensorSpec,
):
    blueprint = (
        world
        .get_blueprint_library()
        .find(spec.blueprint_id)
    )

    _set_supported_attributes(
        blueprint,
        spec.attributes,
    )

    if blueprint.has_attribute("role_name"):
        blueprint.set_attribute(
            "role_name",
            spec.name,
        )

    return world.spawn_actor(
        blueprint,
        spec.transform,
        attach_to=vehicle,
        attachment_type=carla.AttachmentType.Rigid,
    )


def _listen(
    actor,
    spec: SensorSpec,
    sync,
    camera_stream,
    radar_stream,
) -> None:
    if spec.kind == "camera":

        def camera_callback(image):
            sync.push(
                spec.name,
                image.frame,
                image,
            )

            if spec.primary:
                camera_stream.push(
                    image.frame,
                    image_to_rgb(image),
                )

        actor.listen(camera_callback)
        return

    if spec.kind == "radar":

        def radar_callback(data, sensor_name=spec.name):
            sync.push(sensor_name, data.frame, data)
            radar_stream.push(
                sensor_name,
                data.frame,
                radar_to_list(data),
            )

        actor.listen(radar_callback)
        return

    actor.listen(
        lambda data, sensor_name=spec.name: sync.push(
            sensor_name,
            data.frame,
            data,
        )
    )


def spawn_layout(
    world,
    vehicle,
    layout: SensorLayout,
    sync,
    camera_stream,
    radar_stream,
) -> List[object]:
    actors = []

    try:
        for spec in layout.all_specs:
            actor = _spawn_actor(
                world,
                vehicle,
                spec,
            )

            _listen(
                actor,
                spec,
                sync,
                camera_stream,
                radar_stream,
            )

            actors.append(actor)

            print(
                f"[OK] Sensor: {spec.name:<32} "
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
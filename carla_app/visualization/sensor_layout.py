"""Sensör yerleşimini tarayıcı ekranının kullanacağı veriye çevirir."""

import json
from pathlib import Path

TEMPLATE_FILE = Path(__file__).with_name("sensor_layout.html")


def sensor_fov(spec):
    """Sensör türüne göre yatay görüş açısını okur."""
    if spec.kind == "camera":
        return float(spec.attributes.get("fov", 0.0))
    return float(spec.attributes.get("horizontal_fov", 0.0))


def sensor_range(spec):
    """Sensörde menzil ayarı varsa metre cinsinden döndürür."""
    value = spec.attributes.get("range")
    return float(value) if value is not None else None


def build_web_view_data(layout, active_sensor_names, vehicle_type_id):
    """Yerleşimi tarayıcıya gömülebilecek sade bir sözlüğe çevirir."""
    active_sensor_names = set(active_sensor_names)
    sensors = []

    for spec in layout.all_specs:
        location = spec.transform.location
        rotation = spec.transform.rotation
        sensors.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "active": spec.name in active_sensor_names,
                "physical_sensor": spec.physical_sensor,
                "blueprint_id": spec.blueprint_id,
                "position": {
                    "x": float(location.x),
                    "y": float(location.y),
                    "z": float(location.z),
                },
                "rotation": {
                    "roll": float(rotation.roll),
                    "pitch": float(rotation.pitch),
                    "yaw": float(rotation.yaw),
                },
                "horizontal_fov_deg": sensor_fov(spec),
                "range_m": sensor_range(spec),
            }
        )

    return {
        "vehicle_type_id": vehicle_type_id,
        "vehicle": dict(layout.vehicle_geometry),
        "sensors": sensors,
    }


def render_web_view(data):
    """Sensör verisini tek başına açılabilen HTML dosyasına yerleştirir."""
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return template.replace("__SENSOR_LAYOUT_DATA__", encoded)

#!/usr/bin/env python3
"""CARLA sensör yerleşimini sade bir tarayıcı ekranında gösterir."""

import argparse
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carla_app.visualization.sensor_layout import (  # noqa: E402
    build_web_view_data,
    render_web_view,
)


def parse_arguments():
    """Komut satırından çıktı yolu ve bekleme süresini okur."""
    parser = argparse.ArgumentParser(
        description=(
            "Ego aracını ve sensörleri tarayıcıda üstten/yandan göster."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/carla_sensor_layout.html"),
        help="Oluşturulacak HTML dosyası",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=30.0,
        help="Ego aracını bekleme süresi",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="HTML dosyasını oluştur ama tarayıcıyı otomatik açma",
    )
    return parser.parse_args()


def find_ego_vehicle(world, role_name):
    """Verilen rol adına sahip ego aracını CARLA dünyasında bulur."""
    for vehicle in world.get_actors().filter("vehicle.*"):
        if vehicle.attributes.get("role_name") == role_name:
            return vehicle
    return None


def wait_for_ego_vehicle(client, role_name, wait_seconds):
    """Normal uygulamanın ego aracını oluşturmasını belirli süre bekler."""
    deadline = time.monotonic() + max(0.0, wait_seconds)

    while True:
        world = client.get_world()
        vehicle = find_ego_vehicle(world, role_name)
        if vehicle is not None:
            return world, vehicle

        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"role_name={role_name!r} ego aracı bulunamadı. "
                "Önce başka terminalde 'bash run.sh' çalıştır."
            )

        print(f"[BEKLE] role_name={role_name} ego aracı bekleniyor...")
        time.sleep(0.5)


def active_sensor_names(world, vehicle):
    """Ego aracına gerçekten bağlı olan sensör adlarını döndürür."""
    names = set()
    for actor in world.get_actors().filter("sensor.*"):
        parent = getattr(actor, "parent", None)
        if parent is None or parent.id != vehicle.id:
            continue

        name = actor.attributes.get("role_name")
        if name:
            names.add(name)

    return names


def main():
    """CARLA'dan yerleşimi okuyup HTML dosyasını ve tarayıcıyı açar."""
    args = parse_arguments()

    try:
        import carla
        from carla_app.config import Settings
        from carla_app.sensors.layout import build_sensor_layout
    except ModuleNotFoundError as error:
        missing_package = error.name or "bilinmeyen paket"
        raise RuntimeError(
            f"Python paketi {missing_package!r} bu ortamda bulunamadı. "
            "Normal uygulamayı çalıştırdığın Python ortamını kullan veya "
            "'pip install -r requirements.txt' çalıştır."
        ) from error

    settings = Settings.load()
    client = carla.Client(settings.host, settings.port)
    client.set_timeout(min(5.0, max(1.0, settings.timeout)))

    print(f"[BAGLAN] CARLA {settings.host}:{settings.port}")
    world, vehicle = wait_for_ego_vehicle(
        client,
        settings.ego_role_name,
        args.wait_seconds,
    )

    layout = build_sensor_layout(
        vehicle=vehicle,
        camera_width=settings.camera_width,
        camera_height=settings.camera_height,
        front_wide_fov=settings.camera_fov,
        fixed_delta_seconds=settings.fixed_delta_seconds,
    )
    active_names = active_sensor_names(world, vehicle)
    data = build_web_view_data(layout, active_names, vehicle.type_id)
    active_count = sum(sensor["active"] for sensor in data["sensors"])

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_web_view(data), encoding="utf-8")

    print(
        f"[HAZIR] {active_count} aktif / "
        f"{len(layout.all_specs)} toplam sensör"
    )
    print(f"[DOSYA] {output}")

    if not args.no_browser:
        opened = webbrowser.open(output.as_uri())
        if opened:
            print("[AÇILDI] Sensör ekranı tarayıcıda açıldı.")
        else:
            print("[BİLGİ] Tarayıcı açılmadıysa şu komutu çalıştır:")
            print(f"xdg-open {output}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as error:
        raise SystemExit(f"[HATA] {error}") from error

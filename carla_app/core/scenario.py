"""YAML senaryo dosyasındaki harita, zaman ve trafik ayarlarını okur."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Scenario:
    map_name: str | None
    synchronous_mode: bool
    fixed_delta_seconds: float
    traffic_manager_port: int
    npc_count: int
    safe_distance_m: float
    speed_difference_percent: float


def load_scenario(path: Path, default_delta: float) -> Scenario:
    if not path.is_file():
        raise FileNotFoundError(f"Senaryo bulunamadi: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    simulation = data.get("simulation", {})
    traffic = data.get("traffic", {})

    return Scenario(
        map_name=data.get("map"),
        synchronous_mode=bool(simulation.get("synchronous_mode", True)),
        fixed_delta_seconds=float(simulation.get("fixed_delta_seconds", default_delta)),
        traffic_manager_port=int(traffic.get("tm_port", 8000)),
        npc_count=int(traffic.get("npc_count", 0)),
        safe_distance_m=float(traffic.get("safe_distance_m", 3.0)),
        speed_difference_percent=float(traffic.get("speed_difference_percent", 0.0)),
    )

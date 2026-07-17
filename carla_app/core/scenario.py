"""YAML senaryo dosyasındaki harita, zaman ve trafik ayarlarını okur."""

import yaml


class Scenario:
    """Senaryo dosyasından okunan değerleri normal sınıf alanlarında tutar."""

    def __init__(
        self,
        map_name,
        synchronous_mode,
        fixed_delta_seconds,
        traffic_manager_port,
        npc_count,
        safe_distance_m,
        speed_difference_percent,
    ):
        self.map_name = map_name
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds
        self.traffic_manager_port = traffic_manager_port
        self.npc_count = npc_count
        self.safe_distance_m = safe_distance_m
        self.speed_difference_percent = speed_difference_percent


def load_scenario(path, default_delta):
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

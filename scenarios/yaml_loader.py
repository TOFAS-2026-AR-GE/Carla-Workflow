from pathlib import Path

import yaml


def load_scenario(path):
    scenario_path = Path(path)

    with scenario_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


import os

import carla

from scenarios.load_map import apply_world_settings
from scenarios.load_map import load_map
from scenarios.traffic_scenario.traffic import spawn_traffic
from scenarios.yaml_loader import load_scenario

def connect_carla(host, port, timeout):
  client = carla.Client(host, port)
  client.set_timeout(timeout)
  
  world = client.get_world()

  scenario_file = os.getenv("SCENARIO_FILE")
  scenario = load_scenario(scenario_file)

  world = load_map(
      client=client,
      world=world,
      map_name=scenario["map"],
  )
  apply_world_settings(
      world=world,
      scenario=scenario,
  )
  spawn_traffic(
      client=client,
      world=world,
      scenario=scenario,
  )
  
  print(f"[OK] Carlaya baglandi: {host}:{port}")
  print(f"[OK] Current map: {world.get_map().name}")
  
  return client, world

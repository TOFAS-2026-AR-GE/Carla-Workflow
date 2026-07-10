import carla
import random

def spawn_vehicle(world: carla.World, vehicle_name):
  bp_lib = world.get_blueprint_library()
  tesla_bp = bp_lib.find(vehicle_name)
  spawn_points = world.get_map().get_spawn_points()
  
  if not spawn_points:
    raise RuntimeError("No spawn points found in this map.")
  
  random.shuffle(spawn_points)
  for spawn_point in spawn_points:
    tesla = world.spawn_actor(tesla_bp, spawn_point)
    if tesla is not None:
      print(f"[OK] Spawned vehicle: {tesla.type_id}")
      print(f"[OK] Vehicle id: {tesla.id}")
      return tesla
    
  raise RuntimeError("Could not spawn a vehicle. Try again or change the map.")
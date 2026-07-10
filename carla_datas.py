import carla
import math

def get_datas(world: carla.World, vehicle: carla.Vehicle, lookahead_distance: float = 4.0):
  transform = vehicle.get_transform()
  location = transform.location
  rotation = transform.rotation
  velocity = vehicle.get_velocity()
  
  speed_mps = math.sqrt(
        velocity.x**2 +
        velocity.y**2 +
        velocity.z**2
    )
  speed_kmh = speed_mps * 3.6
  carla_map = world.get_map()
  current_waypoint = carla_map.get_waypoint(
        location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

  next_waypoints = current_waypoint.next(lookahead_distance)
  
  if next_waypoints:
    next_waypoint = next_waypoints[0]
    next_location = next_waypoint.transform.location
  else:
    next_waypoint = None
    next_location = None
    
  data = {
        "location": location,
        "rotation": rotation,
        "velocity": velocity,

        "x": location.x,
        "y": location.y,
        "z": location.z,
        "yaw": rotation.yaw,

        "speed_mps": speed_mps,
        "speed_kmh": speed_kmh,

        "current_waypoint": current_waypoint,
        "next_waypoint": next_waypoint,
        "next_location": next_location,

        "road_id": current_waypoint.road_id,
        "lane_id": current_waypoint.lane_id,
        "lane_width": current_waypoint.lane_width,
        "is_junction": current_waypoint.is_junction,
    }
  
  return data
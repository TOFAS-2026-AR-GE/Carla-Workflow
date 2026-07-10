from dotenv import load_dotenv
import os
import time
from connect_carla import connect_carla
from spec_camera import update_spectator_camera
from vehicle import spawn_vehicle
from carla_datas import get_datas
from controllers import LaneFollowController



load_dotenv()

host = os.getenv('HOST')
port = int(os.getenv('PORT'))
timeout = float(os.getenv('TIMEOUT'))
vehicle_name = os.getenv('VEHICLE_NAME')
  
def main():
  
  client, world = connect_carla(host, port, timeout)
  vehicle = None
  controller = LaneFollowController()
  
  try:
    vehicle = spawn_vehicle(world, vehicle_name)
    print("[OK] Third-person spectator follow started.")
    print("[INFO] Press CTRL+C to stop.")

    while True:
      data = get_datas(world, vehicle, lookahead_distance=4.0)
      control = controller.run_step(data)
      vehicle.apply_control(control)
      update_spectator_camera(world, vehicle)
      time.sleep(0.03)

  except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")

  finally:
    if vehicle is not None:
      vehicle.destroy()
      print("[OK] Vehicle destroyed.")

if __name__ == '__main__':
  main()
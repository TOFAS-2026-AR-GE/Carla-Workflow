import carla

def connect_carla(host, port, timeout):
  client = carla.Client(host, port)
  client.set_timeout(timeout)
  
  world = client.get_world()
  
  print(f"[OK] Carlaya baglandi: {host}:{port}")
  print(f"[OK] Current map: {world.get_map().name}")
  
  return client, world
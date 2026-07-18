"""CARLA bağlantısını açar ve dünya ayarlarını kapanışta geri yükler."""

import carla


class CarlaSession:
    """CARLA istemcisi ile senkron dünya ayarlarının yaşam döngüsü."""

    def __init__(self, settings, scenario):
        self.settings = settings
        self.scenario = scenario
        self.client = None
        self.world = None
        self.original_settings = None

    def open(self):
        self.client = carla.Client(self.settings.host, self.settings.port)
        self.client.set_timeout(self.settings.timeout)
        self.world = self.client.get_world()

        if self.scenario.map_name:
            current_map = self.world.get_map().name
            if not current_map.endswith(self.scenario.map_name):
                print(f"[INFO] Harita yukleniyor: {self.scenario.map_name}")
                self.world = self.client.load_world(self.scenario.map_name)

        self.original_settings = self.world.get_settings()
        world_settings = self.world.get_settings()
        world_settings.synchronous_mode = self.scenario.synchronous_mode
        world_settings.fixed_delta_seconds = self.scenario.fixed_delta_seconds
        self.world.apply_settings(world_settings)

        print(f"[OK] CARLA baglantisi: {self.settings.host}:{self.settings.port}")
        print(f"[OK] Harita: {self.world.get_map().name}")
        return self.client, self.world

    def close(self):
        if self.world is not None and self.original_settings is not None:
            self.world.apply_settings(self.original_settings)
            print("[OK] CARLA ayarlari geri yuklendi.")

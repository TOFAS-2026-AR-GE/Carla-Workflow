import io
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    sys.modules["carla"] = carla

carla = sys.modules["carla"]
carla.Transform = getattr(carla, "Transform", object)
carla.Location = getattr(carla, "Location", object)
carla.Rotation = getattr(carla, "Rotation", object)
carla.AttachmentType = getattr(
    carla,
    "AttachmentType",
    types.SimpleNamespace(Rigid="Rigid"),
)

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from carla_app.sensors.manager import SensorManager  # noqa: E402


class SensorManagerTests(unittest.TestCase):
    def test_live_control_spawns_only_primary_camera_and_front_radar(self):
        settings = types.SimpleNamespace(
            enable_data_recording=False,
            output_folder=None,
            camera_width=800,
            camera_height=600,
            camera_fov=90.0,
            fixed_delta_seconds=0.05,
        )
        camera = types.SimpleNamespace(name="camera_front_wide")
        radar = types.SimpleNamespace(name="radar_front_long")
        unused = types.SimpleNamespace(name="lidar_roof")
        layout = types.SimpleNamespace(
            control_specs=(camera, radar),
            all_specs=(camera, radar, unused),
            sensor_names=[camera.name, radar.name, unused.name],
        )
        captured = {}

        def fake_spawn_layout(**arguments):
            captured.update(arguments)
            return []

        manager = SensorManager(settings)
        with (
            patch(
                "carla_app.sensors.manager.build_sensor_layout",
                return_value=layout,
            ),
            patch(
                "carla_app.sensors.manager.spawn_layout",
                side_effect=fake_spawn_layout,
            ),
            redirect_stdout(io.StringIO()),
        ):
            manager.start(world=object(), vehicle=object())

        self.assertEqual(captured["specs"], (camera, radar))
        self.assertIsNone(captured["sync"])


if __name__ == "__main__":
    unittest.main()

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

from carla_app.sensors import layout as layout_module  # noqa: E402
from carla_app.sensors.manager import SensorManager  # noqa: E402
from carla_app.sensors.stream import CameraStream  # noqa: E402


class CameraStreamTests(unittest.TestCase):
    def test_returns_delayed_camera_frame_instead_of_discarding_it(self):
        stream = CameraStream(max_frames=4)
        image = object()
        stream.push(98, image)

        camera_frame_id, returned_image = stream.wait_latest(100, timeout=0.0)

        self.assertEqual(camera_frame_id, 98)
        self.assertIs(returned_image, image)

    def test_returns_each_camera_frame_only_once(self):
        stream = CameraStream(max_frames=4)
        stream.push(12, object())

        self.assertEqual(stream.wait_latest(14, timeout=0.0)[0], 12)
        self.assertEqual(stream.wait_latest(15, timeout=0.0), (None, None))

    def test_future_frame_is_not_labeled_as_current(self):
        stream = CameraStream(max_frames=4)
        stream.push(21, object())

        self.assertEqual(stream.wait_latest(20, timeout=0.0), (None, None))
        self.assertEqual(stream.wait_latest(21, timeout=0.0)[0], 21)


class SensorManagerTests(unittest.TestCase):
    def test_front_radar_is_mounted_high_and_aimed_above_the_road(self):
        vehicle = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.75),
                extent=types.SimpleNamespace(x=2.35, y=0.95, z=0.75),
            )
        )

        def fake_location(x=0.0, y=0.0, z=0.0):
            return types.SimpleNamespace(x=x, y=y, z=z)

        def fake_rotation(roll=0.0, pitch=0.0, yaw=0.0):
            return types.SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)

        def fake_transform(location, rotation):
            return types.SimpleNamespace(location=location, rotation=rotation)

        with (
            patch.object(layout_module.carla, "Location", side_effect=fake_location),
            patch.object(layout_module.carla, "Rotation", side_effect=fake_rotation),
            patch.object(layout_module.carla, "Transform", side_effect=fake_transform),
        ):
            sensor_layout = layout_module.build_sensor_layout(
                vehicle=vehicle,
                camera_width=800,
                camera_height=600,
                front_wide_fov=90.0,
                fixed_delta_seconds=0.05,
            )

        geometry = sensor_layout.front_radar_geometry
        self.assertGreaterEqual(geometry["height_above_ground_m"], 0.85)
        self.assertAlmostEqual(geometry["pitch_deg"], 2.0)
        self.assertEqual(sensor_layout.front_radar.attributes["vertical_fov"], "6.0")

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

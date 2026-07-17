import math
import types
import unittest

from carla_app.sensors.rviz_geometry import (
    build_sensor_visuals,
    carla_location_to_ros_xyz,
    carla_rotation_to_ros_quaternion,
)


class RvizGeometryTests(unittest.TestCase):
    def test_world_location_flips_carla_right_to_ros_left(self):
        location = types.SimpleNamespace(x=10.0, y=4.0, z=1.0)

        self.assertEqual(
            carla_location_to_ros_xyz(location),
            (10.0, -4.0, 1.0),
        )

    def test_carla_right_becomes_ros_left(self):
        spec = types.SimpleNamespace(
            name="camera_right",
            kind="camera",
            transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=1.0, y=2.0, z=3.0),
                rotation=types.SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
            ),
            attributes={"fov": "90.0"},
        )
        layout = types.SimpleNamespace(all_specs=(spec,))

        visual = build_sensor_visuals(layout, {"camera_right"})[0]

        self.assertEqual(visual.position_xyz, (1.0, -2.0, 3.0))
        self.assertTrue(visual.active)
        self.assertEqual(visual.horizontal_fov_deg, 90.0)

    def test_carla_positive_yaw_becomes_negative_ros_yaw(self):
        rotation = types.SimpleNamespace(roll=0.0, pitch=0.0, yaw=90.0)

        _, _, z, w = carla_rotation_to_ros_quaternion(rotation)

        self.assertAlmostEqual(z, -math.sqrt(0.5))
        self.assertAlmostEqual(w, math.sqrt(0.5))

    def test_inactive_sensor_is_marked_as_planned(self):
        spec = types.SimpleNamespace(
            name="lidar_roof",
            kind="lidar",
            transform=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=2.0),
                rotation=types.SimpleNamespace(roll=0.0, pitch=0.0, yaw=0.0),
            ),
            attributes={"range": "80.0", "horizontal_fov": "360.0"},
        )
        layout = types.SimpleNamespace(all_specs=(spec,))

        visual = build_sensor_visuals(layout, set())[0]

        self.assertFalse(visual.active)
        self.assertEqual(visual.display_range_m, 5.0)


if __name__ == "__main__":
    unittest.main()

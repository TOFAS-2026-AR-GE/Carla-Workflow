import tempfile
import types
import unittest
from pathlib import Path

from carla_app.sensors.web_viewer import build_web_view_data, render_web_view


def sensor(name, kind, attributes):
    return types.SimpleNamespace(
        name=name,
        kind=kind,
        physical_sensor=f"physical {kind}",
        blueprint_id=f"sensor.test.{kind}",
        attributes=attributes,
        transform=types.SimpleNamespace(
            location=types.SimpleNamespace(x=1.25, y=-0.5, z=1.6),
            rotation=types.SimpleNamespace(roll=0.0, pitch=-4.0, yaw=-60.0),
        ),
    )


class WebViewerTests(unittest.TestCase):
    def setUp(self):
        self.camera = sensor("camera_front", "camera", {"fov": "90.0"})
        self.radar = sensor(
            "radar_front",
            "radar",
            {"horizontal_fov": "30.0", "range": "150.0"},
        )
        self.layout = types.SimpleNamespace(
            all_specs=(self.camera, self.radar),
            vehicle_geometry={
                "length_m": 4.7,
                "width_m": 1.9,
                "height_m": 1.5,
                "half_length_m": 2.35,
                "half_width_m": 0.95,
                "half_height_m": 0.75,
                "bounding_box_center_x_m": 0.0,
                "bounding_box_center_y_m": 0.0,
                "bounding_box_center_z_m": 0.75,
            },
        )

    def test_builds_browser_data_with_active_state_and_geometry(self):
        data = build_web_view_data(
            self.layout,
            active_sensor_names={"camera_front"},
            vehicle_type_id="vehicle.test.car",
        )

        self.assertEqual(data["vehicle_type_id"], "vehicle.test.car")
        self.assertEqual(data["vehicle"]["length_m"], 4.7)
        self.assertTrue(data["sensors"][0]["active"])
        self.assertFalse(data["sensors"][1]["active"])
        self.assertEqual(data["sensors"][0]["horizontal_fov_deg"], 90.0)
        self.assertEqual(data["sensors"][1]["range_m"], 150.0)
        self.assertEqual(data["sensors"][0]["position"]["y"], -0.5)

    def test_renders_one_self_contained_html_file(self):
        data = build_web_view_data(
            self.layout,
            active_sensor_names={"camera_front"},
            vehicle_type_id="vehicle.test.car",
        )
        html = render_web_view(data)

        self.assertIn("CARLA Sensör Yerleşimi", html)
        self.assertIn("vehicle.test.car", html)
        self.assertIn("camera_front", html)
        self.assertNotIn("__SENSOR_LAYOUT_DATA__", html)
        self.assertNotIn("<script src=", html)
        self.assertNotIn("<link rel=", html)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "layout.html"
            output.write_text(html, encoding="utf-8")
            self.assertGreater(output.stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()

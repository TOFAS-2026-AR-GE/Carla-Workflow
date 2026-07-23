import os
import sys
import types
import unittest
from unittest.mock import patch

if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *arguments, **keywords: None
    sys.modules["dotenv"] = dotenv

from carla_app.config import Settings


class SensorModeTests(unittest.TestCase):
    def test_control_mode_uses_low_latency_front_camera_without_bev_panel(self):
        environment = {
            "SENSOR_MODE": "control",
            "ENABLE_DATA_RECORDING": "false",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertFalse(settings.show_bev_panel)
        self.assertFalse(settings.enable_multicamera_perception)
        self.assertEqual(
            (settings.camera_width, settings.camera_height, settings.camera_fov),
            (1640, 590, 150.0),
        )
        self.assertEqual(
            (settings.dashboard_width, settings.dashboard_height),
            (640, 640),
        )
        self.assertFalse(settings.show_lane_overlay)
        self.assertEqual(settings.perception_latency_budget_ms, 80.0)

    def test_dashboard_size_is_always_square(self):
        environment = {
            "DASHBOARD_SIZE": "900",
            "DASHBOARD_WIDTH": "1500",
            "DASHBOARD_HEIGHT": "600",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = Settings()

        self.assertEqual(settings.dashboard_size, 900)
        self.assertEqual(
            (settings.dashboard_width, settings.dashboard_height),
            (900, 900),
        )

    def test_bev_mode_does_not_enable_recording(self):
        environment = {
            "SENSOR_MODE": "bev",
            "ENABLE_DATA_RECORDING": "false",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = Settings()

        self.assertEqual(settings.sensor_mode, "bev")
        self.assertTrue(settings.enable_bev)
        self.assertFalse(settings.enable_data_recording)
        self.assertTrue(settings.show_bev_panel)
        self.assertTrue(settings.enable_multicamera_perception)

    def test_bev_validation_is_enabled_in_every_sensor_mode(self):
        for mode in ("control", "bev", "record"):
            with self.subTest(mode=mode):
                environment = {
                    "SENSOR_MODE": mode,
                    "ENABLE_DATA_RECORDING": "false",
                }
                with patch.dict(os.environ, environment, clear=False):
                    settings = Settings()

                self.assertTrue(settings.enable_bev)
                self.assertEqual(
                    settings.enable_data_recording,
                    mode == "record",
                )

    def test_old_recording_flag_still_selects_record_mode(self):
        environment = {
            "SENSOR_MODE": "control",
            "ENABLE_DATA_RECORDING": "true",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = Settings()

        self.assertEqual(settings.sensor_mode, "record")
        self.assertTrue(settings.enable_data_recording)
        self.assertTrue(settings.enable_bev)

    def test_unknown_sensor_mode_is_rejected(self):
        environment = {
            "SENSOR_MODE": "unknown",
            "ENABLE_DATA_RECORDING": "false",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaises(ValueError):
                Settings()

    def test_bev_update_interval_cannot_be_zero(self):
        environment = {
            "SENSOR_MODE": "bev",
            "ENABLE_DATA_RECORDING": "false",
            "BEV_UPDATE_EVERY_N_FRAMES": "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = Settings()

        self.assertEqual(settings.bev_update_every_n_frames, 1)


if __name__ == "__main__":
    unittest.main()

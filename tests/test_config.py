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

    def test_old_recording_flag_still_selects_record_mode(self):
        environment = {
            "SENSOR_MODE": "control",
            "ENABLE_DATA_RECORDING": "true",
        }
        with patch.dict(os.environ, environment, clear=False):
            settings = Settings()

        self.assertEqual(settings.sensor_mode, "record")
        self.assertTrue(settings.enable_data_recording)
        self.assertFalse(settings.enable_bev)

    def test_unknown_sensor_mode_is_rejected(self):
        environment = {
            "SENSOR_MODE": "unknown",
            "ENABLE_DATA_RECORDING": "false",
        }
        with patch.dict(os.environ, environment, clear=False):
            with self.assertRaises(ValueError):
                Settings()


if __name__ == "__main__":
    unittest.main()

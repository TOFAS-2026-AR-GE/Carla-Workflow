"""Oracle doğrulama çıktısının kontrol akışına bağlanmadığını doğrular."""

import unittest
from pathlib import Path


class OracleIsolationTests(unittest.TestCase):
    def test_oracle_module_declares_control_disconnection(self):
        source = Path("carla_app/validation/oracle.py").read_text(encoding="utf-8")
        self.assertIn('"control_connected": False', source)

    def test_localized_state_does_not_add_simulator_light(self):
        source = Path("carla_app/localization/system.py").read_text(encoding="utf-8")
        self.assertNotIn('"simulator_traffic_light":', source)


if __name__ == "__main__":
    unittest.main()

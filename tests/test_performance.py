import unittest

from carla_app.core.performance import PerformanceMonitor
from carla_app.perception.device import is_cuda_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_device_writings_are_recognized(self):
        self.assertTrue(is_cuda_device("0"))
        self.assertTrue(is_cuda_device("cuda"))
        self.assertTrue(is_cuda_device("cuda:1"))
        self.assertFalse(is_cuda_device("cpu"))


class PerformanceMonitorTests(unittest.TestCase):
    def test_summary_reports_budget_and_queue_pressure(self):
        monitor = PerformanceMonitor(frame_budget_ms=50.0, smoothing=1.0)
        monitor.update(
            process_ms=60.0,
            viewer_ms=8.0,
            camera_wait_ms=2.0,
            perception_result={
                "elapsed_ms": 25.0,
                "queue_delay_ms": 12.0,
            },
            worker_diagnostics={"dropped": 3, "processed": 4},
        )

        summary = monitor.summary()

        self.assertIn("loop=60.0ms", summary)
        self.assertIn("infer=25.0ms", summary)
        self.assertIn("drop=3", summary)
        self.assertIn("budget_over=100%", summary)


if __name__ == "__main__":
    unittest.main()

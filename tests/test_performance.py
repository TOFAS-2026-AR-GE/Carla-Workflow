import unittest

from carla_app.core.performance import (
    AdaptivePerceptionScheduler,
    PerformanceMonitor,
)
from carla_app.perception.device import (
    is_cuda_device,
    ultralytics_precision_arguments,
)
from carla_app.perception.performance_profile import (
    choose_performance_profile,
)


class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_device_writings_are_recognized(self):
        self.assertTrue(is_cuda_device("0"))
        self.assertTrue(is_cuda_device("cuda"))
        self.assertTrue(is_cuda_device("cuda:1"))
        self.assertFalse(is_cuda_device("cpu"))

    def test_ultralytics_precision_argument_supports_old_and_new_8x_apis(self):
        self.assertEqual(
            ultralytics_precision_arguments(
                True,
                "cuda:0",
                supported_options={"half"},
            ),
            {"half": True},
        )
        self.assertEqual(
            ultralytics_precision_arguments(
                True,
                "cuda:0",
                supported_options={"quantize"},
            ),
            {"quantize": 16},
        )
        self.assertEqual(
            ultralytics_precision_arguments(
                True,
                "cpu",
                supported_options={"quantize"},
            ),
            {"quantize": None},
        )

    def test_32_gb_gpu_uses_every_frame_cuda_ultra_profile(self):
        profile = choose_performance_profile(
            True,
            total_vram_mb=32_000,
            free_vram_mb=25_000,
        )

        self.assertEqual(profile.name, "cuda-ultra")
        self.assertEqual(profile.perception_every_n_frames, 1)
        self.assertEqual(profile.vehicle_image_size, 640)
        self.assertEqual(profile.camera_inference_batch_size, 7)

    def test_4_gb_gpu_uses_low_vram_profile(self):
        profile = choose_performance_profile(
            True,
            total_vram_mb=4_000,
            free_vram_mb=1_500,
        )

        self.assertEqual(profile.name, "cuda-low-vram")
        self.assertEqual(profile.perception_every_n_frames, 2)
        self.assertEqual(profile.vehicle_image_size, 512)
        self.assertEqual(profile.camera_inference_batch_size, 1)

    def test_cpu_profile_reduces_inference_load(self):
        profile = choose_performance_profile(False)

        self.assertEqual(profile.name, "cpu")
        self.assertEqual(profile.perception_every_n_frames, 2)
        self.assertEqual(profile.vehicle_image_size, 416)


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
        self.assertIn("e2e=37.0ms", summary)
        self.assertIn("drop=3", summary)
        self.assertIn("budget_over=100%", summary)

    def test_reused_latest_result_does_not_distort_latency_average(self):
        monitor = PerformanceMonitor(
            frame_budget_ms=50.0,
            smoothing=1.0,
            latency_budget_ms=80.0,
        )
        slow_result = {
            "elapsed_ms": 90.0,
            "queue_delay_ms": 10.0,
            "worker_total_ms": 100.0,
        }
        diagnostics = {"dropped": 0, "processed": 1}
        monitor.update(20.0, 4.0, 0.0, slow_result, diagnostics)

        reused_result = {
            "elapsed_ms": 5.0,
            "queue_delay_ms": 0.0,
            "worker_total_ms": 5.0,
        }
        monitor.update(20.0, 4.0, 0.0, reused_result, diagnostics)

        self.assertEqual(monitor.values["end_to_end_ms"], 100.0)
        self.assertEqual(monitor.total_results, 1)
        self.assertEqual(monitor.over_latency_results, 1)

    def test_scheduler_reduces_load_when_inference_drops_frames(self):
        scheduler = AdaptivePerceptionScheduler(
            frame_budget_ms=50.0,
            initial_period=1,
            maximum_period=3,
            evaluation_frames=10,
        )

        change = None
        for _ in range(10):
            change = scheduler.update(
                inference_ms=70.0,
                worker_diagnostics={"submitted": 20, "dropped": 5},
            )

        self.assertEqual(change, 2)

    def test_scheduler_returns_to_every_frame_after_stable_windows(self):
        scheduler = AdaptivePerceptionScheduler(
            frame_budget_ms=50.0,
            initial_period=2,
            maximum_period=3,
            evaluation_frames=10,
        )

        change = None
        submitted = 0
        for _window in range(3):
            for _ in range(10):
                submitted += 1
                change = scheduler.update(
                    inference_ms=20.0,
                    worker_diagnostics={
                        "submitted": submitted,
                        "dropped": 0,
                    },
                )

        self.assertEqual(change, 1)


if __name__ == "__main__":
    unittest.main()

import os
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lane_ground_truth import (
    evaluate_lane_geometry,
    project_world_points,
    sample_ego_lane_boundaries,
)


def transform(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        ),
    )


class LaneGroundTruthGeometryTests(unittest.TestCase):
    def test_world_points_project_with_carla_to_opencv_axis_conversion(self):
        points = np.asarray(
            [
                [10.0, 0.0, 0.0],
                [10.0, -2.0, 0.0],
                [10.0, 2.0, 0.0],
            ]
        )

        pixels = project_world_points(
            points,
            camera_transform=transform(z=0.0),
            image_width=800,
            image_height=600,
            horizontal_fov_degrees=90.0,
        )

        self.assertEqual(pixels.shape, (3, 2))
        self.assertAlmostEqual(pixels[0, 0], 400.0)
        self.assertAlmostEqual(pixels[0, 1], 300.0)
        self.assertLess(pixels[1, 0], 400.0)
        self.assertGreater(pixels[2, 0], 400.0)

    def test_metric_matches_curves_independent_of_input_order(self):
        y_values = np.arange(120, 581, 4)
        left_truth = np.column_stack((250.0 + 0.1 * y_values, y_values))
        right_truth = np.column_stack((550.0 - 0.1 * y_values, y_values))
        predictions = [
            {
                "detected": True,
                "points": (right_truth + [3.0, 0.0]).tolist(),
            },
            {
                "detected": True,
                "points": (left_truth - [2.0, 0.0]).tolist(),
            },
        ]

        result = evaluate_lane_geometry(
            predictions,
            [left_truth, right_truth],
            image_width=800,
        )

        self.assertEqual(result["matched_count"], 2)
        self.assertAlmostEqual(result["precision"], 1.0)
        self.assertAlmostEqual(result["recall"], 1.0)
        self.assertAlmostEqual(result["f1"], 1.0)
        self.assertLess(result["mean_error_px"], 3.1)
        self.assertGreater(result["mean_coverage"], 0.99)

    def test_metric_rejects_geometrically_wrong_lane(self):
        truth = np.column_stack(
            (
                np.full(100, 250.0),
                np.linspace(150.0, 580.0, 100),
            )
        )
        prediction = {
            "detected": True,
            "points": (truth + [90.0, 0.0]).tolist(),
        }

        result = evaluate_lane_geometry(
            [prediction],
            [truth],
            image_width=800,
        )

        self.assertEqual(result["matched_count"], 0)
        self.assertEqual(result["f1"], 0.0)
        self.assertIsNone(result["mean_error_px"])


@unittest.skipUnless(
    os.getenv("RUN_CARLA_LANE_GT") == "1",
    "Canli CARLA lane ground-truth testi RUN_CARLA_LANE_GT=1 ister.",
)
class LiveCarlaLaneGroundTruthTests(unittest.TestCase):
    """Gerçek model çıktısını CARLA harita sınırlarıyla yalnız testte ölçer."""

    def test_model_matches_projected_carla_lane_boundaries(self):
        import carla

        from carla_app.perception.lane_detector import LaneDetector
        from carla_app.sensors.processors import image_to_rgb

        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "2000"))
        image_width = int(os.getenv("CAMERA_WIDTH", "800"))
        image_height = int(os.getenv("CAMERA_HEIGHT", "600"))
        camera_fov = float(os.getenv("CAMERA_FOV", "90"))
        sample_count = int(os.getenv("LANE_GT_SAMPLE_COUNT", "30"))
        model_path = Path(
            os.getenv(
                "LANE_MODEL",
                "models/lane/ufld_carla_best.pth",
            )
        )
        self.assertTrue(model_path.is_file(), f"Lane modeli yok: {model_path}")

        client = carla.Client(host, port)
        client.set_timeout(20.0)
        world = client.get_world()
        ego = next(
            (
                actor
                for actor in world.get_actors().filter("vehicle.*")
                if actor.attributes.get("role_name") == "ego_vehicle"
            ),
            None,
        )
        self.assertIsNotNone(ego, "role_name=ego_vehicle aktörü bulunamadı.")

        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(image_width))
        blueprint.set_attribute("image_size_y", str(image_height))
        blueprint.set_attribute("fov", str(camera_fov))
        blueprint.set_attribute("sensor_tick", "0.0")
        blueprint.set_attribute("motion_blur_intensity", "0.0")
        camera = world.spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(x=0.8, z=1.55),
                carla.Rotation(pitch=-4.0),
            ),
            attach_to=ego,
            attachment_type=carla.AttachmentType.Rigid,
        )
        frames = queue.Queue(maxsize=2)

        def callback(image):
            while frames.full():
                try:
                    frames.get_nowait()
                except queue.Empty:
                    break
            frames.put_nowait((int(image.frame), image_to_rgb(image)))

        camera.listen(callback)
        detector = LaneDetector(
            model_path,
            device=os.getenv("LANE_DEVICE", "auto"),
            minimum_points=int(os.getenv("LANE_MINIMUM_POINTS", "3")),
            minimum_confidence=float(os.getenv("LANE_CONFIDENCE", "0.30")),
        )
        metrics = []
        try:
            for _sample_index in range(sample_count):
                settings = world.get_settings()
                if settings.synchronous_mode:
                    world.tick()
                else:
                    world.wait_for_tick(10.0)
                _frame_id, rgb_image = frames.get(timeout=10.0)
                result = detector.detect(rgb_image)
                boundaries = sample_ego_lane_boundaries(
                    world.get_map(),
                    ego.get_location(),
                )
                truth = [
                    project_world_points(
                        boundaries["left"],
                        camera.get_transform(),
                        image_width,
                        image_height,
                        camera_fov,
                    ),
                    project_world_points(
                        boundaries["right"],
                        camera.get_transform(),
                        image_width,
                        image_height,
                        camera_fov,
                    ),
                ]
                truth = [curve for curve in truth if len(curve) >= 4]
                if len(truth) != 2:
                    continue
                ego_lanes = [
                    lane
                    for lane in result["lanes"]
                    if int(lane["lane_index"]) in {1, 2}
                ]
                metrics.append(
                    evaluate_lane_geometry(
                        ego_lanes,
                        truth,
                        image_width=image_width,
                    )
                )
        finally:
            camera.stop()
            camera.destroy()

        self.assertGreaterEqual(
            len(metrics),
            max(5, sample_count // 2),
            "Ölçülebilir iki şerit sınırı içeren yeterli CARLA karesi yok.",
        )
        f1 = float(np.mean([item["f1"] for item in metrics]))
        coverage = float(np.mean([item["mean_coverage"] for item in metrics]))
        errors = [
            item["mean_error_px"]
            for item in metrics
            if item["mean_error_px"] is not None
        ]
        p95_errors = [
            item["p95_error_px"]
            for item in metrics
            if item["p95_error_px"] is not None
        ]
        self.assertGreaterEqual(f1, float(os.getenv("LANE_GT_MIN_F1", "0.90")))
        self.assertGreaterEqual(
            coverage,
            float(os.getenv("LANE_GT_MIN_COVERAGE", "0.70")),
        )
        self.assertLessEqual(
            float(np.mean(errors)),
            float(os.getenv("LANE_GT_MAX_MEAN_ERROR_PX", "20")),
        )
        self.assertLessEqual(
            float(np.mean(p95_errors)),
            float(os.getenv("LANE_GT_MAX_P95_ERROR_PX", "40")),
        )

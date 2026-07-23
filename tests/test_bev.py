import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import cv2
except ModuleNotFoundError:
    cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = cv2

if not hasattr(cv2, "resize"):
        cv2.LINE_AA = 16
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.INTER_AREA = 3
        cv2.INTER_LINEAR = 1
        cv2.INTER_NEAREST = 0
        cv2.BORDER_CONSTANT = 0
        cv2.WINDOW_NORMAL = 0
        cv2.WND_PROP_VISIBLE = 0
        cv2.EVENT_LBUTTONUP = 4
        cv2.line = lambda *arguments, **keywords: None
        cv2.arrowedLine = lambda *arguments, **keywords: None
        cv2.circle = lambda *arguments, **keywords: None
        cv2.rectangle = lambda *arguments, **keywords: None
        cv2.fillPoly = lambda *arguments, **keywords: None
        cv2.polylines = lambda *arguments, **keywords: None
        cv2.putText = lambda *arguments, **keywords: None
        cv2.namedWindow = lambda *arguments, **keywords: None
        cv2.setMouseCallback = lambda *arguments, **keywords: None
        cv2.imshow = lambda *arguments, **keywords: None
        cv2.waitKey = lambda *arguments, **keywords: -1
        cv2.getWindowProperty = lambda *arguments, **keywords: 1
        cv2.destroyWindow = lambda *arguments, **keywords: None
        cv2.error = RuntimeError

        def resize(image, size, interpolation=None):
            width, height = size
            source_height, source_width = image.shape[:2]
            y_index = np.linspace(0, source_height - 1, height).astype(np.int64)
            x_index = np.linspace(0, source_width - 1, width).astype(np.int64)
            if image.ndim == 2:
                return image[y_index[:, None], x_index]
            return image[y_index[:, None], x_index, :]

        cv2.resize = resize

        def remap(image, map_x, map_y, **_options):
            height, width = map_x.shape
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        cv2.remap = remap

        def warp_perspective(image, _matrix, size, **_options):
            width, height = size
            if image.shape[:2] == (height, width):
                return image.copy()
            if image.ndim == 2:
                return np.zeros((height, width), dtype=image.dtype)
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        cv2.warpPerspective = warp_perspective

        def add_weighted(first, alpha, second, beta, gamma, dst=None):
            result = first.astype(np.float32) * alpha
            result += second.astype(np.float32) * beta + gamma
            result = np.clip(result, 0, 255).astype(first.dtype)
            if dst is not None:
                dst[:] = result
                return dst
            return result

        cv2.addWeighted = add_weighted

from carla_app.bev.association import group_associated_measurements
from carla_app.bev.calibration import (
    CalibrationSet,
    CameraCalibration,
    build_camera_matrix,
)
from carla_app.bev.camera_ipm import CameraIpm
from carla_app.bev.coordinate import EgoMotionCompensator, MetricGrid
from carla_app.bev.fusion import SensorFusion
from carla_app.bev.localization import LocalizationHealth
from carla_app.bev.module import BevModule
from carla_app.bev.occupancy import OccupancyGrid
from carla_app.bev.projector import BevProjector
from carla_app.bev.renderer import BevRenderer
from carla_app.bev.tracking import BevTracker
from carla_app.visualization.navigation_panel import NavigationPanel
from carla_app.visualization.viewer import PerceptionViewer


def transform(x=0.0, y=0.0, z=0.0, yaw=0.0, pitch=0.0):
    return SimpleNamespace(
        location=SimpleNamespace(x=x, y=y, z=z),
        rotation=SimpleNamespace(yaw=yaw, pitch=pitch, roll=0.0),
    )


def make_layout():
    camera = SimpleNamespace(
        name="camera_front_wide",
        kind="camera",
        transform=transform(x=1.0, z=1.4, pitch=-10.0),
        attributes={
            "fov": "90.0",
            "image_size_x": "100",
            "image_size_y": "100",
        },
    )
    radar = SimpleNamespace(
        name="radar_front_right",
        kind="radar",
        transform=transform(yaw=90.0),
        attributes={"horizontal_fov": "30.0", "range": "90.0"},
    )
    lidar = SimpleNamespace(
        name="lidar_roof",
        kind="lidar",
        transform=transform(z=1.8),
        attributes={"range": "80.0"},
    )
    gnss = SimpleNamespace(
        name="gnss_roof",
        kind="gnss",
        transform=transform(z=1.9),
        attributes={},
    )
    imu = SimpleNamespace(
        name="imu_cg",
        kind="imu",
        transform=transform(z=0.5),
        attributes={},
    )
    geometry = {
        "bounding_box_center_z_m": 0.75,
        "half_height_m": 0.75,
        "half_length_m": 2.35,
        "half_width_m": 0.95,
    }
    return SimpleNamespace(
        cameras=(camera,),
        radars=(radar,),
        lidar=lidar,
        gnss=gnss,
        imu=imu,
        all_specs=(camera, radar, lidar, gnss, imu),
        vehicle_geometry=geometry,
    )


def camera_spec(name, x, y, z, yaw, pitch, fov):
    return SimpleNamespace(
        name=name,
        kind="camera",
        transform=transform(x=x, y=y, z=z, yaw=yaw, pitch=pitch),
        attributes={
            "fov": str(fov),
            "image_size_x": "800",
            "image_size_y": "600",
        },
    )


def make_surround_layout():
    base = make_layout()
    cameras = (
        camera_spec("camera_front_wide", 1.0, 0.0, 1.4, 0.0, -4.0, 90.0),
        camera_spec("camera_front_narrow", 1.0, 0.0, 1.4, 0.0, -2.0, 50.0),
        camera_spec("camera_front_left", 0.5, -1.0, 1.1, -60.0, -4.0, 100.0),
        camera_spec("camera_front_right", 0.5, 1.0, 1.1, 60.0, -4.0, 100.0),
        camera_spec("camera_rear_left", -0.5, -1.0, 1.1, -120.0, -4.0, 100.0),
        camera_spec("camera_rear_right", -0.5, 1.0, 1.1, 120.0, -4.0, 100.0),
        camera_spec("camera_rear_center", -2.4, 0.0, 1.0, 180.0, -5.0, 110.0),
    )
    all_specs = cameras + (base.lidar, base.gnss, base.imu) + base.radars
    return SimpleNamespace(
        cameras=cameras,
        radars=base.radars,
        lidar=base.lidar,
        gnss=base.gnss,
        imu=base.imu,
        all_specs=all_specs,
        vehicle_geometry=base.vehicle_geometry,
    )


class BevProjectorTests(unittest.TestCase):
    def test_radar_yaw_is_applied_in_ego_coordinates(self):
        layout = make_layout()
        projector = BevProjector(layout)
        snapshot = {
            "radar_front_right": {
                "frame_id": 10,
                "age_frames": 0,
                "data": [
                    {
                        "depth_m": 10.0,
                        "azimuth_deg": 0.0,
                        "altitude_deg": 0.0,
                        "relative_velocity_mps": -1.0,
                    }
                ],
            }
        }

        points = projector.project_radars(snapshot)

        self.assertAlmostEqual(points[0]["x_m"], 0.0, places=6)
        self.assertAlmostEqual(points[0]["y_m"], 10.0, places=6)

    def test_camera_bbox_bottom_projects_to_ground_in_front(self):
        layout = make_layout()
        projector = BevProjector(layout)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        perception = {
            "camera_results": {
                "camera_front_wide": {
                    "frame_id": 10,
                    "image": image,
                    "vehicles": [
                        {
                            "bbox": (40, 30, 60, 90),
                            "class_name": "vehicle",
                            "confidence": 0.9,
                        }
                    ],
                }
            }
        }

        detections = projector.project_camera_detections(perception)

        self.assertEqual(len(detections), 1)
        self.assertGreater(detections[0]["x_m"], 1.0)
        self.assertAlmostEqual(detections[0]["y_m"], 0.0, places=6)

    def test_stale_camera_detection_is_not_projected(self):
        layout = make_layout()
        projector = BevProjector(layout, maximum_measurement_age_frames=5)
        perception = {
            "camera_results": {
                "camera_front_wide": {
                    "frame_id": 10,
                    "image": np.zeros((100, 100, 3), dtype=np.uint8),
                    "vehicles": [
                        {
                            "bbox": (40, 30, 60, 90),
                            "class_name": "vehicle",
                            "confidence": 0.9,
                        }
                    ],
                }
            }
        }

        detections = projector.project_camera_detections(
            perception,
            current_frame_id=20,
        )

        self.assertEqual(detections, [])


class LocalizationHealthTests(unittest.TestCase):
    def test_fresh_gnss_and_imu_are_healthy(self):
        health = LocalizationHealth(make_layout(), maximum_age_frames=2)
        snapshot = {
            "gnss_roof": {
                "frame_id": 20,
                "data": {
                    "latitude": 41.015,
                    "longitude": 28.979,
                    "altitude": 42.0,
                },
            },
            "imu_cg": {
                "frame_id": 20,
                "data": {
                    "accelerometer": {"x": 0.1, "y": 1.2, "z": 9.81},
                    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.08},
                    "compass": 1.5,
                },
            },
        }

        result = health.evaluate(snapshot, current_frame_id=21)

        self.assertEqual(result["status"], "HEALTHY")
        self.assertEqual(result["available_sources"], 2)
        self.assertAlmostEqual(
            result["imu"]["lateral_acceleration_mps2"],
            1.2,
        )
        self.assertAlmostEqual(result["imu"]["yaw_rate_radps"], 0.08)

    def test_stale_gnss_and_invalid_imu_are_unavailable(self):
        health = LocalizationHealth(make_layout(), maximum_age_frames=2)
        snapshot = {
            "gnss_roof": {
                "frame_id": 10,
                "data": {
                    "latitude": 41.015,
                    "longitude": 28.979,
                    "altitude": 42.0,
                },
            },
            "imu_cg": {
                "frame_id": 20,
                "data": {
                    "accelerometer": {"x": 0.1, "y": float("nan"), "z": 9.81},
                    "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.08},
                    "compass": 1.5,
                },
            },
        }

        result = health.evaluate(snapshot, current_frame_id=21)

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertEqual(result["gnss"]["reason"], "stale")
        self.assertEqual(result["imu"]["reason"], "invalid_values")


class CalibrationTests(unittest.TestCase):
    def test_camera_matrix_uses_horizontal_fov(self):
        camera_matrix = build_camera_matrix(800, 600, 90.0)

        self.assertAlmostEqual(camera_matrix[0, 0], 400.0)
        self.assertAlmostEqual(camera_matrix[1, 1], 400.0)
        self.assertAlmostEqual(camera_matrix[0, 2], 400.0)
        self.assertAlmostEqual(camera_matrix[1, 2], 300.0)

    def test_ground_homography_round_trip_returns_same_point(self):
        camera = make_layout().cameras[0]
        calibration = CameraCalibration(camera, ground_z_m=0.0)
        ground = np.array([[12.0, 2.0]], dtype=np.float64)

        pixels, depth = calibration.project_ground_points(ground)
        returned = calibration.pixel_to_ground_point(
            pixels[0, 0],
            pixels[0, 1],
        )

        self.assertGreater(depth[0], 0.0)
        self.assertAlmostEqual(returned[0], 12.0, places=5)
        self.assertAlmostEqual(returned[1], 2.0, places=5)

    def test_seven_camera_ipm_covers_front_sides_and_rear(self):
        layout = make_surround_layout()
        grid = MetricGrid(width=200, height=160)
        ipm = CameraIpm(
            layout,
            calibrations=CalibrationSet(layout),
            grid=grid,
        )
        check_points = [
            (10.0, 0.0),
            (0.0, 10.0),
            (0.0, -10.0),
            (-10.0, 0.0),
        ]

        for forward_m, right_m in check_points:
            pixel_x, pixel_y = grid.to_pixel(forward_m, right_m)
            camera_count = 0
            for camera_map in ipm.camera_maps.values():
                if camera_map["weight"][pixel_y, pixel_x] > 0.0:
                    camera_count += 1
            self.assertGreater(camera_count, 0)

    def test_seven_camera_images_are_combined_in_one_mosaic(self):
        layout = make_surround_layout()
        grid = MetricGrid(width=100, height=80)
        ipm = CameraIpm(layout, CalibrationSet(layout), grid)
        camera_results = {}
        for camera in layout.cameras:
            camera_results[camera.name] = {
                "frame_id": 10,
                "image": np.zeros((600, 800, 3), dtype=np.uint8),
            }

        mosaic, coverage, used_cameras = ipm.build_mosaic(camera_results)

        self.assertEqual(mosaic.shape, (80, 100, 3))
        self.assertEqual(len(used_cameras), 7)
        self.assertTrue(np.any(coverage > 0.0))


class AssociationTests(unittest.TestCase):
    def test_two_boxes_from_same_camera_are_not_merged(self):
        measurements = [
            {
                "x_m": 10.0,
                "y_m": 0.0,
                "uncertainty_m": 1.0,
                "sensor_names": ["camera_front_wide"],
            },
            {
                "x_m": 10.8,
                "y_m": 0.2,
                "uncertainty_m": 1.0,
                "sensor_names": ["camera_front_wide"],
            },
        ]

        groups = group_associated_measurements(measurements)

        self.assertEqual(len(groups), 2)

    def test_radar_cannot_indirectly_merge_two_boxes_from_same_camera(self):
        measurements = [
            {
                "x_m": 10.0,
                "y_m": 0.0,
                "uncertainty_m": 1.0,
                "sensor_names": ["camera_front_wide"],
            },
            {
                "x_m": 13.0,
                "y_m": 0.0,
                "uncertainty_m": 1.0,
                "sensor_names": ["camera_front_wide"],
            },
            {
                "x_m": 11.5,
                "y_m": 0.0,
                "uncertainty_m": 1.0,
                "sensor_names": ["radar_front_long"],
            },
        ]

        groups = group_associated_measurements(measurements)

        self.assertEqual(len(groups), 2)
        for group in groups:
            camera_count = 0
            for measurement in group:
                if "camera_front_wide" in measurement["sensor_names"]:
                    camera_count += 1
            self.assertLessEqual(camera_count, 1)


class EgoMotionTests(unittest.TestCase):
    def test_old_point_is_compensated_after_ego_moves_forward(self):
        motion = EgoMotionCompensator()
        motion.remember(
            10,
            {
                "location": SimpleNamespace(x=0.0, y=0.0),
                "yaw": 0.0,
            },
        )
        motion.remember(
            11,
            {
                "location": SimpleNamespace(x=1.0, y=0.0),
                "yaw": 0.0,
            },
        )

        result = motion.compensate_points(
            np.array([[10.0, 0.0, 0.0]]),
            old_frame_id=10,
            current_frame_id=11,
        )

        self.assertAlmostEqual(result[0, 0], 9.0)
        self.assertAlmostEqual(result[0, 1], 0.0)


class FusionTests(unittest.TestCase):
    def test_overlapping_cameras_radar_and_lidar_become_one_object(self):
        fusion = SensorFusion()
        camera_objects = [
            {
                "x_m": 20.0,
                "y_m": 0.0,
                "source": "camera",
                "sensor_names": ["camera_front_wide"],
                "frame_ids": [10],
                "class_name": "vehicle",
                "confidence": 0.8,
                "uncertainty_m": 1.2,
            },
            {
                "x_m": 20.5,
                "y_m": 0.2,
                "source": "camera",
                "sensor_names": ["camera_front_narrow"],
                "frame_ids": [10],
                "class_name": "vehicle",
                "confidence": 0.9,
                "uncertainty_m": 0.9,
            },
        ]
        radar_points = []
        for offset in (-0.2, 0.2):
            radar_points.append(
                {
                    "x_m": 19.8,
                    "y_m": offset,
                    "z_m": 0.8,
                    "sensor_name": "radar_front_long",
                    "frame_id": 10,
                    "relative_velocity_mps": -1.0,
                    "ground_return": False,
                }
            )
        lidar_points = np.array(
            [
                [19.0, -0.8, 0.5],
                [19.2, -0.4, 0.6],
                [19.5, 0.0, 0.7],
                [19.8, 0.4, 0.8],
                [20.0, 0.8, 0.9],
                [20.5, -0.8, 1.0],
                [21.0, -0.4, 1.1],
                [21.5, 0.0, 1.2],
                [22.0, 0.4, 1.3],
                [22.5, 0.8, 1.4],
            ]
        )

        fused = fusion.build_fused_objects(
            camera_objects,
            radar_points,
            lidar_points,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["class_name"], "vehicle")
        self.assertEqual(fused[0]["sources"], ["camera", "lidar", "radar"])
        self.assertLess(fused[0]["uncertainty_m"], 0.9)
        self.assertEqual(np.asarray(fused[0]["covariance_xy"]).shape, (2, 2))

    def test_lidar_only_obstacle_is_available_to_validation(self):
        fusion = SensorFusion()
        lidar_points = np.array(
            [
                [10.0 + 0.2 * row, -0.6 + 0.2 * column, 0.8]
                for row in range(5)
                for column in range(7)
            ],
            dtype=np.float64,
        )

        fused = fusion.build_fused_objects(
            camera_objects=[],
            radar_points=[],
            lidar_points=lidar_points,
            lidar_frame_id=12,
        )

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["sources"], ["lidar"])
        self.assertEqual(fused[0]["frame_ids"], [12])


class TrackingTests(unittest.TestCase):
    def test_world_track_stays_fixed_when_ego_moves(self):
        tracker = BevTracker(fixed_delta_seconds=0.05)
        first_pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        second_pose = {"x": 1.0, "y": 0.0, "yaw_deg": 0.0}
        measurement = {
            "x_m": 10.0,
            "y_m": 0.0,
            "uncertainty_m": 0.3,
            "confidence": 0.9,
            "class_name": "vehicle",
            "sources": ["camera", "radar"],
            "sensor_names": ["camera_front_wide", "radar_front_long"],
            "length_m": 4.5,
            "width_m": 1.9,
            "measurement_key": "first",
        }
        tracker.update([measurement], 10, first_pose)

        result = tracker.update([], 11, second_pose)

        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["x_m"], 9.0, places=4)

    def test_same_sensor_measurement_is_not_counted_twice(self):
        tracker = BevTracker(fixed_delta_seconds=0.05)
        pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        measurement = {
            "x_m": 10.0,
            "y_m": 0.0,
            "uncertainty_m": 0.3,
            "confidence": 0.9,
            "class_name": "vehicle",
            "sources": ["camera"],
            "sensor_names": ["camera_front_wide"],
            "length_m": 4.5,
            "width_m": 1.9,
            "measurement_key": "camera:10:box",
        }

        tracker.update([measurement], 10, pose)
        result = tracker.update([measurement], 11, pose)

        self.assertFalse(result[0]["confirmed"])

    def test_reused_sensor_epoch_does_not_age_track(self):
        tracker = BevTracker(fixed_delta_seconds=0.05)
        pose = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
        measurement = {
            "x_m": 10.0,
            "y_m": 0.0,
            "uncertainty_m": 0.3,
            "confidence": 0.9,
            "class_name": "vehicle",
            "sources": ["camera"],
            "sensor_names": ["camera_front_wide"],
            "length_m": 4.5,
            "width_m": 1.9,
            "measurement_key": "camera:10:box",
        }

        tracker.update(
            [measurement],
            10,
            pose,
            evidence_frame_id=10,
        )
        result = tracker.update(
            [measurement],
            11,
            pose,
            evidence_frame_id=10,
        )

        self.assertEqual(result[0]["misses"], 0)

    def test_global_assignment_avoids_greedy_identity_swap(self):
        tracker = BevTracker(fixed_delta_seconds=0.05)
        tracker.tracks = {
            1: {"state": np.array([0.0, 0.0, 0.0, 0.0])},
            2: {"state": np.array([2.0, 0.0, 0.0, 0.0])},
        }
        measurements = [
            {"world_x": 0.9, "world_y": 0.0, "uncertainty_m": 0.3},
            {"world_x": -1.0, "world_y": 0.0, "uncertainty_m": 0.3},
        ]

        matches, unmatched = tracker.match_measurements(measurements)

        self.assertEqual(sorted(matches), [(1, 1), (2, 0)])
        self.assertEqual(unmatched, [])


class OccupancyTests(unittest.TestCase):
    def test_lidar_ray_marks_free_path_and_occupied_endpoint(self):
        occupancy = OccupancyGrid(
            forward_range_m=10.0,
            rear_range_m=2.0,
            side_range_m=5.0,
            cell_size_m=1.0,
        )
        motion = EgoMotionCompensator()
        motion.remember(
            1,
            {
                "location": SimpleNamespace(x=0.0, y=0.0),
                "yaw": 0.0,
            },
        )

        state = occupancy.update(
            lidar_points=np.array([[5.0, 0.0, 1.0]]),
            lidar_origin=(0.0, 0.0),
            radar_points=[],
            tracked_objects=[],
            ground_z_m=0.0,
            motion_compensator=motion,
            current_frame_id=1,
        )

        end_x, end_y = occupancy.grid.to_pixel(5.0, 0.0)
        free_x, free_y = occupancy.grid.to_pixel(2.0, 0.0)
        self.assertTrue(state["occupied"][end_y, end_x])
        self.assertTrue(state["free"][free_y, free_x])

    def test_ground_lidar_endpoint_is_free_not_occupied(self):
        occupancy = OccupancyGrid(
            forward_range_m=10.0,
            rear_range_m=2.0,
            side_range_m=5.0,
            cell_size_m=1.0,
        )
        motion = EgoMotionCompensator()
        motion.remember(
            1,
            {
                "location": SimpleNamespace(x=0.0, y=0.0),
                "yaw": 0.0,
            },
        )

        state = occupancy.update(
            lidar_points=np.array([[5.0, 0.0, 0.10]]),
            lidar_origin=(0.0, 0.0),
            radar_points=[],
            tracked_objects=[],
            ground_z_m=0.0,
            motion_compensator=motion,
            current_frame_id=1,
        )

        end_x, end_y = occupancy.grid.to_pixel(5.0, 0.0)
        self.assertTrue(state["free"][end_y, end_x])
        self.assertFalse(state["occupied"][end_y, end_x])

    def test_same_lidar_frame_is_not_integrated_twice(self):
        occupancy = OccupancyGrid(
            forward_range_m=10.0,
            rear_range_m=2.0,
            side_range_m=5.0,
            cell_size_m=1.0,
        )
        motion = EgoMotionCompensator()
        for frame_id in (1, 2):
            motion.remember(
                frame_id,
                {
                    "location": SimpleNamespace(x=0.0, y=0.0),
                    "yaw": 0.0,
                },
            )
        point = np.array([[5.0, 0.0, 1.0]])
        occupancy.update(
            point,
            (0.0, 0.0),
            [],
            [],
            0.0,
            motion,
            1,
            measurement_frames={"lidar_roof": 1},
        )
        end_x, end_y = occupancy.grid.to_pixel(5.0, 0.0)
        first_log_odds = float(occupancy.log_odds[end_y, end_x])

        occupancy.update(
            point,
            (0.0, 0.0),
            [],
            [],
            0.0,
            motion,
            2,
            measurement_frames={"lidar_roof": 1},
        )
        second_log_odds = float(occupancy.log_odds[end_y, end_x])

        self.assertLess(second_log_odds, first_log_odds)

    def test_track_overlay_is_not_reported_as_independent_sensor_evidence(self):
        occupancy = OccupancyGrid(
            forward_range_m=10.0,
            rear_range_m=2.0,
            side_range_m=5.0,
            cell_size_m=1.0,
        )
        motion = EgoMotionCompensator()
        motion.remember(
            1,
            {
                "location": SimpleNamespace(x=0.0, y=0.0),
                "yaw": 0.0,
            },
        )
        track = {
            "track_id": 3,
            "x_m": 5.0,
            "y_m": 0.0,
            "length_m": 4.0,
            "width_m": 2.0,
            "confirmed": True,
            "misses": 0,
        }

        state = occupancy.update(
            np.empty((0, 3)),
            (0.0, 0.0),
            [],
            [track],
            0.0,
            motion,
            1,
            measurement_frames={},
        )

        pixel_x, pixel_y = occupancy.grid.to_pixel(5.0, 0.0)
        self.assertGreater(state["probability"][pixel_y, pixel_x], 0.62)
        self.assertAlmostEqual(
            float(state["sensor_probability"][pixel_y, pixel_x]),
            0.5,
            places=5,
        )


class ValidationTests(unittest.TestCase):
    def contribution_scene(self):
        probability = np.full((24, 20), 0.5, dtype=np.float32)
        occupancy = {
            "probability": probability.copy(),
            "sensor_probability": probability,
            "occupied": np.zeros_like(probability, dtype=bool),
            "free": np.zeros_like(probability, dtype=bool),
            "cell_size_m": 1.0,
            "forward_range_m": 20.0,
            "rear_range_m": 4.0,
            "side_range_m": 10.0,
        }
        # Yaklaşık (15 m ileri, 0 m yanal) hücresinde bağımsız ham sensör kanıtı.
        occupancy["sensor_probability"][5, 10] = 0.85
        return {
            "frame_id": 20,
            "sensor_status": {
                "camera_front_narrow": {"kind": "camera", "age_frames": 0},
                "lidar_roof": {"kind": "lidar", "age_frames": 0},
            },
            "tracks": [
                {
                    "track_id": 12,
                    "x_m": 15.0,
                    "y_m": 0.0,
                    "velocity_x_mps": 4.0,
                    "velocity_y_mps": 0.0,
                    "length_m": 4.0,
                    "width_m": 1.8,
                    "confirmed": True,
                    "confidence": 0.93,
                    "uncertainty_m": 0.35,
                    "sources": ["camera", "lidar"],
                    "sensor_names": ["camera_front_narrow", "lidar_roof"],
                    "source_frames": {"camera": 20, "lidar": 20},
                    "last_measurement_frame_id": 20,
                    "misses": 0,
                }
            ],
            "occupancy": occupancy,
            "route_points": [(0.0, 0.0), (30.0, 0.0)],
            "lane_width_m": 3.5,
            "vehicle_geometry": {"half_length_m": 2.0},
        }

    def test_independent_fresh_track_confirms_controller_lead(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        scene = {
            "frame_id": 20,
            "sensor_status": {
                "camera_front_wide": {"kind": "camera", "age_frames": 1},
                "radar_front_long": {"kind": "radar", "age_frames": 0},
                "lidar_roof": {"kind": "lidar", "age_frames": 0},
            },
            "tracks": [
                {
                    "track_id": 4,
                    "x_m": 19.5,
                    "y_m": 0.2,
                    "confirmed": True,
                    "confidence": 0.92,
                    "uncertainty_m": 0.4,
                    "sources": ["camera", "radar"],
                    "misses": 0,
                }
            ],
            "occupancy": None,
        }
        lead = {"distance_m": 20.0, "lateral_m": 0.0}

        result = validator.evaluate(scene, 21, lead_vehicle=lead)

        self.assertEqual(result["status"], "CONFIRMED")
        self.assertEqual(result["lead"]["track_id"], 4)
        self.assertTrue(result["safe_to_use"])

    def test_controller_sensors_alone_support_but_do_not_independently_confirm(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        scene = {
            "frame_id": 20,
            "sensor_status": {
                "camera_front_wide": {"kind": "camera", "age_frames": 0},
                "radar_front_long": {"kind": "radar", "age_frames": 0},
            },
            "tracks": [
                {
                    "track_id": 8,
                    "x_m": 15.0,
                    "y_m": 0.0,
                    "confirmed": True,
                    "confidence": 0.95,
                    "uncertainty_m": 0.3,
                    "sources": ["camera", "radar"],
                    "sensor_names": [
                        "camera_front_wide",
                        "radar_front_long",
                    ],
                    "source_frames": {"camera": 20, "radar": 20},
                    "misses": 0,
                }
            ],
            "occupancy": None,
        }
        lead = {
            "distance_m": 15.0,
            "lateral_m": 0.0,
            "source": "camera_radar_track",
        }

        result = validator.evaluate(scene, 20, lead_vehicle=lead)

        self.assertEqual(result["lead"]["status"], "SUPPORTED")
        self.assertEqual(result["status"], "UNKNOWN")

    def test_missing_or_stale_scene_never_rejects_a_hazard(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer(maximum_scene_age_frames=3)
        lead = {"distance_m": 8.0, "lateral_m": 0.0}

        result = validator.evaluate(
            {"frame_id": 10, "sensor_status": {}, "tracks": []},
            20,
            lead_vehicle=lead,
        )

        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertFalse(result["safe_to_use"])
        self.assertNotEqual(result["lead"]["status"], "REJECTED")

    def test_worker_lag_can_validate_but_cannot_drive_from_stale_scene(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        scene = self.contribution_scene()

        validation = validator.evaluate(scene, current_frame_id=26)
        contribution = validator.contribute(
            scene,
            current_frame_id=26,
            state={"speed_mps": 8.0},
            lead_vehicle=None,
        )

        self.assertTrue(validation["safe_to_use"])
        self.assertNotEqual(validation["status"], "UNAVAILABLE")
        self.assertFalse(contribution["applied"])
        self.assertEqual(contribution["reason"], "bev_scene_not_control_fresh")

    def test_control_freshness_boundary_is_inclusive_then_expires(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        scene = self.contribution_scene()

        fresh = validator.contribute(
            scene,
            current_frame_id=24,
            state={"speed_mps": 8.0},
            lead_vehicle=None,
        )
        stale = validator.contribute(
            scene,
            current_frame_id=25,
            state={"speed_mps": 8.0},
            lead_vehicle=None,
        )

        self.assertTrue(fresh["applied"])
        self.assertFalse(stale["applied"])
        self.assertEqual(stale["reason"], "bev_scene_not_control_fresh")

    def test_existing_controller_lead_is_never_replaced(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        existing = {
            "distance_m": 18.0,
            "relative_speed_mps": -1.0,
            "source": "camera_radar_track",
        }

        result = validator.contribute(
            self.contribution_scene(),
            current_frame_id=20,
            state={"speed_mps": 8.0},
            lead_vehicle=existing,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "primary_lead_preserved")
        self.assertIs(result["lead_vehicle"], existing)

    def test_multisensor_route_track_recovers_missing_lead(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()

        result = validator.contribute(
            self.contribution_scene(),
            current_frame_id=20,
            state={"speed_mps": 8.0},
            lead_vehicle=None,
        )

        self.assertTrue(result["applied"])
        self.assertEqual(result["reason"], "confirmed_bev_lead_recovery")
        self.assertEqual(result["lead_vehicle"]["source"], "bev_multisensor_recovery")
        self.assertEqual(result["lead_vehicle"]["track_id"], 12)
        self.assertAlmostEqual(result["lead_vehicle"]["distance_m"], 11.0)
        self.assertAlmostEqual(result["lead_vehicle"]["relative_speed_mps"], -4.0)

    def test_track_overlay_without_raw_sensor_support_cannot_control(self):
        from carla_app.bev.validation import BevValidationLayer

        validator = BevValidationLayer()
        scene = self.contribution_scene()
        scene["occupancy"]["sensor_probability"][:] = 0.5
        scene["occupancy"]["probability"][5, 10] = 0.9

        result = validator.contribute(
            scene,
            current_frame_id=20,
            state={"speed_mps": 8.0},
            lead_vehicle=None,
        )

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason"], "no_control_grade_bev_track")


class BevRendererTests(unittest.TestCase):
    def test_opencv_overlay_draws_objects_signs_context_and_lanes(self):
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        vehicles = [
            {
                "bbox": [10, 20, 80, 100],
                "class_name": "vehicle",
                "confidence": 0.93,
            }
        ]
        signs = [
            {
                "bbox": [100, 20, 130, 70],
                "class_name": "traffic_sign_30",
                "confidence": 0.88,
            }
        ]
        lanes = {
            "lanes": [
                {
                    "detected": True,
                    "lane_index": 1,
                    "points": [[120, 230], [140, 160], [150, 100]],
                    "raw_points": [[121, 229], [149, 101]],
                }
            ]
        }
        road_context = {
            "detections": [
                {
                    "bbox": [200, 25, 230, 75],
                    "class_name": "traffic_light_red",
                    "category": "traffic_light",
                    "confidence": 0.91,
                },
                {
                    "bbox": [10, 20, 80, 100],
                    "class_name": "vehicle",
                    "category": "vehicle",
                    "confidence": 0.93,
                },
            ]
        }

        with (
            patch("carla_app.visualization.viewer.cv2.rectangle") as rectangle,
            patch("carla_app.visualization.viewer.cv2.putText") as put_text,
            patch("carla_app.visualization.viewer.cv2.polylines") as polylines,
            patch("carla_app.visualization.viewer.cv2.circle") as circle,
        ):
            counts = viewer.draw_perception_overlay(
                frame,
                vehicles,
                signs,
                lanes,
                road_context,
            )

        self.assertEqual(counts, {"boxes": 3, "lanes": 1})
        self.assertEqual(rectangle.call_count, 3)
        self.assertEqual(put_text.call_count, 4)
        self.assertEqual(polylines.call_count, 2)
        self.assertEqual(circle.call_count, 4)

    def test_opencv_overlay_fills_only_valid_ego_lane_corridor(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        y_values = list(range(80, 231, 4))
        lanes = [
            {
                "lane_index": 1,
                "detected": True,
                "points": [[100 + y // 10, y] for y in y_values],
            },
            {
                "lane_index": 2,
                "detected": True,
                "points": [[220 - y // 10, y] for y in y_values],
            },
        ]

        with (
            patch("carla_app.visualization.viewer.cv2.fillPoly") as fill_poly,
            patch(
                "carla_app.visualization.viewer.cv2.addWeighted"
            ) as add_weighted,
        ):
            drawn = PerceptionViewer._draw_ego_lane_corridor(frame, lanes)

        self.assertTrue(drawn)
        fill_poly.assert_called_once()
        add_weighted.assert_called_once()

    def test_crossing_lane_curves_do_not_create_corridor_fill(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        y_values = list(range(80, 231, 4))
        lanes = [
            {
                "lane_index": 1,
                "detected": True,
                "points": [[100 + y, y] for y in y_values],
            },
            {
                "lane_index": 2,
                "detected": True,
                "points": [[220 - y, y] for y in y_values],
            },
        ]

        with patch(
            "carla_app.visualization.viewer.cv2.fillPoly"
        ) as fill_poly:
            drawn = PerceptionViewer._draw_ego_lane_corridor(frame, lanes)

        self.assertFalse(drawn)
        fill_poly.assert_not_called()

    def test_module_returns_requested_bgr_canvas(self):
        layout = make_layout()
        module = BevModule(layout, width=320, height=240)
        state = {
            "location": SimpleNamespace(x=0.0, y=0.0),
            "yaw": 0.0,
            "reference_path": [SimpleNamespace(x=10.0, y=0.0)],
        }

        image = module.render({}, None, state, current_frame_id=12)

        self.assertEqual(image.shape, (240, 320, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_validation_only_module_skips_ipm_and_renderer(self):
        layout = make_layout()
        module = BevModule(
            layout,
            width=320,
            height=240,
            render_output=False,
        )
        state = {
            "location": SimpleNamespace(x=0.0, y=0.0),
            "yaw": 0.0,
            "reference_path": [SimpleNamespace(x=10.0, y=0.0)],
        }

        with (
            patch.object(module.camera_ipm, "build_mosaic") as build_mosaic,
            patch.object(module.renderer, "render") as render,
        ):
            image = module.render({}, None, state, current_frame_id=12)

        self.assertIsNone(image)
        build_mosaic.assert_not_called()
        render.assert_not_called()
        self.assertIsNotNone(module.get_latest_scene())
        self.assertIsNone(module.get_latest_scene()["ipm_image"])

    def test_camera_letterbox_preserves_model_aspect_ratio(self):
        image = np.full((590, 1640, 3), 255, dtype=np.uint8)

        panel = PerceptionViewer.resize_with_letterbox(image, 1000, 600)

        non_background = np.any(panel != np.array([8, 11, 15]), axis=2)
        rows = np.flatnonzero(np.any(non_background, axis=1))
        columns = np.flatnonzero(np.any(non_background, axis=0))
        self.assertEqual(panel.shape, (600, 1000, 3))
        self.assertEqual((columns[0], columns[-1]), (0, 999))
        self.assertAlmostEqual(len(rows) / 1000.0, 590.0 / 1640.0, delta=0.002)

    def test_viewer_combines_equal_camera_and_bev_panels(self):
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        viewer.bev_mode = "driving"
        camera = np.zeros((100, 160, 3), dtype=np.uint8)
        bev = np.zeros((50, 80, 3), dtype=np.uint8)

        combined = viewer.combine_panels(camera, bev)

        self.assertEqual(combined.shape, (100, 323, 3))

    def test_bev_switch_selects_debug_and_driving_modes(self):
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        viewer.bev_mode = "driving"
        camera = np.zeros((100, 200, 3), dtype=np.uint8)
        bev = np.zeros((100, 200, 3), dtype=np.uint8)
        viewer.combine_panels(camera, bev)
        x1, y1, x2, y2 = viewer.bev_button_rect

        viewer._mouse_callback(
            cv2.EVENT_LBUTTONUP,
            x2 - 3,
            (y1 + y2) // 2,
            0,
            None,
        )
        self.assertEqual(viewer.bev_mode, "debug")

        viewer._mouse_callback(
            cv2.EVENT_LBUTTONUP,
            x1 + 3,
            (y1 + y2) // 2,
            0,
            None,
        )
        self.assertEqual(viewer.bev_mode, "driving")

    def test_left_click_selects_target_then_confirm_starts_route(self):
        class FakeNavigation:
            def __init__(self):
                self.selected = []
                self.confirmed_from = []

            def select_destination(self, world_x, world_y):
                self.selected.append((world_x, world_y))
                return True

            def confirm_destination(self, vehicle_location):
                self.confirmed_from.append(vehicle_location)
                return True

            def cancel_pending(self):
                raise AssertionError("İptal çağrılmamalı")

        class FakeMapRenderer:
            @staticmethod
            def is_confirm_button(x, y):
                return x == 200 and y == 700

            @staticmethod
            def is_cancel_button(x, y):
                return x == 50 and y == 700

            @staticmethod
            def screen_to_world(x, y):
                if x == 100 and y == 300:
                    return 12.5, -8.0
                return None

        navigation = FakeNavigation()
        panel = NavigationPanel(
            None,
            navigation,
            width=500,
            height=780,
            renderer=FakeMapRenderer(),
        )
        vehicle_location = SimpleNamespace(x=1.0, y=2.0, z=0.0)
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        viewer.bev_button_rect = None
        viewer.navigation_panel = panel
        viewer.map_offset_x = 500
        viewer.last_vehicle_location = vehicle_location

        viewer._mouse_callback(
            cv2.EVENT_LBUTTONUP,
            600,
            300,
            0,
            None,
        )
        self.assertEqual(navigation.selected, [(12.5, -8.0)])
        self.assertEqual(navigation.confirmed_from, [])

        viewer._mouse_callback(
            cv2.EVENT_LBUTTONUP,
            700,
            700,
            0,
            None,
        )
        self.assertEqual(navigation.confirmed_from, [vehicle_location])

    def test_right_click_does_not_select_navigation_target(self):
        class FakePanel:
            def __init__(self):
                self.clicks = []

            def handle_left_click(self, x, y, vehicle_location):
                self.clicks.append((x, y, vehicle_location))

        panel = FakePanel()
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        viewer.bev_button_rect = None
        viewer.navigation_panel = panel
        viewer.map_offset_x = 500
        viewer.last_vehicle_location = SimpleNamespace(x=1.0, y=2.0)

        viewer._mouse_callback(
            getattr(cv2, "EVENT_RBUTTONUP", 5),
            600,
            300,
            0,
            None,
        )

        self.assertEqual(panel.clicks, [])

    def test_navigation_cancel_button_does_not_select_map_point(self):
        class FakeNavigation:
            def __init__(self):
                self.cancel_count = 0
                self.select_count = 0

            def cancel_pending(self):
                self.cancel_count += 1

            def select_destination(self, *_point):
                self.select_count += 1
                return True

        class FakeMapRenderer:
            @staticmethod
            def is_confirm_button(_x, _y):
                return False

            @staticmethod
            def is_cancel_button(x, y):
                return x == 50 and y == 700

            @staticmethod
            def screen_to_world(_x, _y):
                return 12.5, -8.0

        navigation = FakeNavigation()
        panel = NavigationPanel(
            None,
            navigation,
            width=500,
            height=780,
            renderer=FakeMapRenderer(),
        )

        action = panel.handle_left_click(50, 700, None)

        self.assertEqual(action, "cancelled")
        self.assertEqual(navigation.cancel_count, 1)
        self.assertEqual(navigation.select_count, 0)

    def test_renderer_keeps_debug_and_driving_views(self):
        renderer = BevRenderer(width=320, height=240)
        occupancy_shape = (80, 60)
        scene = {
            "ipm_image": np.full((240, 320, 3), 40, dtype=np.uint8),
            "occupancy": {
                "occupied": np.zeros(occupancy_shape, dtype=bool),
                "free": np.zeros(occupancy_shape, dtype=bool),
            },
            "route_points": [(0.0, 0.0), (15.0, 0.0), (30.0, 3.0)],
            "lidar_points": np.empty((0, 3), dtype=np.float32),
            "radar_points": [],
            "tracks": [],
            "vehicle_geometry": make_layout().vehicle_geometry,
            "active_sensor_count": 15,
            "total_sensor_count": 15,
            "ipm_cameras": ["camera"] * 7,
            "lane_width_m": 3.5,
            "ego_speed_mps": 10.0,
            "driving_state": {
                "target_speed_mps": 12.0,
                "mode": "CRUISE",
            },
        }

        driving = renderer.render(scene, current_frame_id=20, display_mode="driving")
        debug = renderer.render(scene, current_frame_id=20, display_mode="debug")

        self.assertEqual(driving.shape, (240, 320, 3))
        self.assertEqual(debug.shape, (240, 320, 3))
        self.assertFalse(np.array_equal(driving, debug))

    def test_metric_origin_is_near_lower_center(self):
        renderer = BevRenderer(width=800, height=600)

        pixel_x, pixel_y = renderer.to_pixel(0.0, 0.0)

        self.assertAlmostEqual(pixel_x, 400, delta=1)
        self.assertAlmostEqual(pixel_y, 450, delta=1)


if __name__ == "__main__":
    unittest.main()

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

if "cv2" not in sys.modules:
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        cv2 = types.ModuleType("cv2")
        cv2.LINE_AA = 16
        cv2.FONT_HERSHEY_SIMPLEX = 0
        cv2.INTER_AREA = 3
        cv2.INTER_LINEAR = 1
        cv2.INTER_NEAREST = 0
        cv2.BORDER_CONSTANT = 0
        cv2.WINDOW_NORMAL = 0
        cv2.WND_PROP_VISIBLE = 0
        cv2.line = lambda *arguments, **keywords: None
        cv2.arrowedLine = lambda *arguments, **keywords: None
        cv2.circle = lambda *arguments, **keywords: None
        cv2.rectangle = lambda *arguments, **keywords: None
        cv2.putText = lambda *arguments, **keywords: None
        cv2.namedWindow = lambda *arguments, **keywords: None
        cv2.imshow = lambda *arguments, **keywords: None
        cv2.waitKey = lambda *arguments, **keywords: -1
        cv2.getWindowProperty = lambda *arguments, **keywords: 1
        cv2.destroyWindow = lambda *arguments, **keywords: None
        cv2.error = RuntimeError

        def resize(image, size, interpolation=None):
            width, height = size
            if image.ndim == 2:
                return np.zeros((height, width), dtype=image.dtype)
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

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
        sys.modules["cv2"] = cv2

from carla_app.bev.calibration import (
    CalibrationSet,
    CameraCalibration,
    build_camera_matrix,
)
from carla_app.bev.association import group_associated_measurements
from carla_app.bev.camera_ipm import CameraIpm
from carla_app.bev.coordinate import EgoMotionCompensator, MetricGrid
from carla_app.bev.fusion import SensorFusion
from carla_app.bev.module import BevModule
from carla_app.bev.occupancy import OccupancyGrid
from carla_app.bev.projector import BevProjector
from carla_app.bev.renderer import BevRenderer
from carla_app.bev.tracking import BevTracker
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
        all_specs=(camera, radar, lidar),
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
    all_specs = cameras + (base.lidar,) + base.radars
    return SimpleNamespace(
        cameras=cameras,
        radars=base.radars,
        lidar=base.lidar,
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


class BevRendererTests(unittest.TestCase):
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

    def test_viewer_combines_equal_camera_and_bev_panels(self):
        viewer = PerceptionViewer.__new__(PerceptionViewer)
        camera = np.zeros((100, 160, 3), dtype=np.uint8)
        bev = np.zeros((50, 80, 3), dtype=np.uint8)

        combined = viewer.combine_panels(camera, bev)

        self.assertEqual(combined.shape, (100, 323, 3))

    def test_metric_origin_is_near_lower_center(self):
        renderer = BevRenderer(width=800, height=600)

        pixel_x, pixel_y = renderer.to_pixel(0.0, 0.0)

        self.assertAlmostEqual(pixel_x, 400, delta=1)
        self.assertAlmostEqual(pixel_y, 450, delta=1)


if __name__ == "__main__":
    unittest.main()

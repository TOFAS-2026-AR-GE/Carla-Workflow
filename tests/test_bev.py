import unittest
import sys
import types
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
            return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

        cv2.resize = resize
        sys.modules["cv2"] = cv2

from carla_app.bev.module import BevModule
from carla_app.bev.projector import BevProjector
from carla_app.bev.renderer import BevRenderer
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
        attributes={"fov": "90.0"},
    )
    radar = SimpleNamespace(
        name="radar_front_right",
        kind="radar",
        transform=transform(yaw=90.0),
        attributes={"horizontal_fov": "30.0"},
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

import io
import struct
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

if "carla" not in sys.modules:
    carla = types.ModuleType("carla")
    sys.modules["carla"] = carla

carla = sys.modules["carla"]
carla.Transform = getattr(carla, "Transform", object)
carla.Location = getattr(carla, "Location", object)
carla.Rotation = getattr(carla, "Rotation", object)
carla.AttachmentType = getattr(
    carla,
    "AttachmentType",
    types.SimpleNamespace(Rigid="Rigid"),
)

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from carla_app.sensors import layout as layout_module  # noqa: E402
from carla_app.sensors.factory import start_sensor_listener  # noqa: E402
from carla_app.sensors.manager import SensorManager  # noqa: E402
from carla_app.sensors.processors import process_packet  # noqa: E402
from carla_app.sensors.stream import CameraStream, LatestSensorStream  # noqa: E402


class CameraStreamTests(unittest.TestCase):
    def test_returns_delayed_camera_frame_instead_of_discarding_it(self):
        stream = CameraStream(max_frames=4)
        image = object()
        stream.push(98, image)

        camera_frame_id, returned_image = stream.wait_latest(100, timeout=0.0)

        self.assertEqual(camera_frame_id, 98)
        self.assertIs(returned_image, image)

    def test_returns_each_camera_frame_only_once(self):
        stream = CameraStream(max_frames=4)
        stream.push(12, object())

        self.assertEqual(stream.wait_latest(14, timeout=0.0)[0], 12)
        self.assertEqual(stream.wait_latest(15, timeout=0.0), (None, None))

    def test_future_frame_is_not_labeled_as_current(self):
        stream = CameraStream(max_frames=4)
        stream.push(21, object())

        self.assertEqual(stream.wait_latest(20, timeout=0.0), (None, None))
        self.assertEqual(stream.wait_latest(21, timeout=0.0)[0], 21)

    def test_bev_snapshot_drops_stale_sensor_without_waiting(self):
        stream = LatestSensorStream()
        stream.push("camera_front_wide", 100, "new")
        stream.push("camera_rear_center", 80, "old")

        snapshot = stream.get_snapshot(105, max_age_frames=10)

        self.assertIn("camera_front_wide", snapshot)
        self.assertNotIn("camera_rear_center", snapshot)
        self.assertEqual(snapshot["camera_front_wide"]["age_frames"], 5)


class SensorManagerTests(unittest.TestCase):
    def test_layout_rejects_zero_simulation_step(self):
        with self.assertRaises(ValueError):
            layout_module.build_sensor_layout(
                vehicle=object(),
                camera_width=800,
                camera_height=600,
                front_wide_fov=90.0,
                fixed_delta_seconds=0.0,
            )

    def test_layout_has_only_real_sensors_and_correct_front_radar(self):
        vehicle = types.SimpleNamespace(
            bounding_box=types.SimpleNamespace(
                location=types.SimpleNamespace(x=0.0, y=0.0, z=0.75),
                extent=types.SimpleNamespace(x=2.35, y=0.95, z=0.75),
            )
        )

        def fake_location(x=0.0, y=0.0, z=0.0):
            return types.SimpleNamespace(x=x, y=y, z=z)

        def fake_rotation(roll=0.0, pitch=0.0, yaw=0.0):
            return types.SimpleNamespace(roll=roll, pitch=pitch, yaw=yaw)

        def fake_transform(location, rotation):
            return types.SimpleNamespace(location=location, rotation=rotation)

        with (
            patch.object(layout_module.carla, "Location", side_effect=fake_location),
            patch.object(layout_module.carla, "Rotation", side_effect=fake_rotation),
            patch.object(layout_module.carla, "Transform", side_effect=fake_transform),
        ):
            sensor_layout = layout_module.build_sensor_layout(
                vehicle=vehicle,
                camera_width=1640,
                camera_height=590,
                front_wide_fov=150.0,
                fixed_delta_seconds=0.05,
                surround_camera_width=640,
                surround_camera_height=360,
            )

        geometry = sensor_layout.front_radar_geometry
        primary_camera = sensor_layout.cameras[0]
        surround_camera = sensor_layout.cameras[1]
        self.assertEqual(primary_camera.attributes["image_size_x"], "1640")
        self.assertEqual(primary_camera.attributes["image_size_y"], "590")
        self.assertEqual(primary_camera.attributes["fov"], "150.0")
        self.assertAlmostEqual(primary_camera.transform.location.x, 1.5)
        self.assertAlmostEqual(primary_camera.transform.location.z, 2.4)
        self.assertAlmostEqual(primary_camera.transform.rotation.pitch, 0.0)
        self.assertEqual(surround_camera.attributes["image_size_x"], "640")
        self.assertEqual(surround_camera.attributes["image_size_y"], "360")
        self.assertGreaterEqual(geometry["height_above_ground_m"], 0.85)
        self.assertAlmostEqual(geometry["pitch_deg"], 2.0)
        self.assertEqual(sensor_layout.front_radar.attributes["vertical_fov"], "6.0")
        self.assertEqual(len(sensor_layout.all_specs), 15)
        self.assertEqual(len(sensor_layout.cameras), 7)
        self.assertEqual(len(sensor_layout.radars), 5)
        self.assertEqual(
            {sensor.kind for sensor in sensor_layout.all_specs},
            {"camera", "radar", "lidar", "gnss", "imu"},
        )

        manifest = sensor_layout.to_manifest("vehicle.test")
        self.assertEqual(manifest["sensor_count"]["total"], 15)
        self.assertEqual(
            set(manifest["sensor_count"]),
            {"cameras", "lidars", "automotive_radars", "gnss", "imu", "total"},
        )

    def test_live_control_spawns_all_sensors_for_bev_validation(self):
        settings = types.SimpleNamespace(
            enable_data_recording=False,
            output_folder=None,
            camera_width=800,
            camera_height=600,
            camera_fov=90.0,
            fixed_delta_seconds=0.05,
        )
        camera = types.SimpleNamespace(name="camera_front_wide")
        radar = types.SimpleNamespace(name="radar_front_long")
        unused = types.SimpleNamespace(name="lidar_roof")
        layout = types.SimpleNamespace(
            control_specs=(camera, radar),
            all_specs=(camera, radar, unused),
            sensor_names=[camera.name, radar.name, unused.name],
        )
        captured = {}

        def fake_spawn_layout(**arguments):
            captured.update(arguments)
            return []

        manager = SensorManager(settings)
        with (
            patch(
                "carla_app.sensors.manager.build_sensor_layout",
                return_value=layout,
            ),
            patch(
                "carla_app.sensors.manager.spawn_layout",
                side_effect=fake_spawn_layout,
            ),
            redirect_stdout(io.StringIO()),
        ):
            manager.start(
                world=object(),
                vehicle=object(),
                fixed_delta_seconds=0.05,
            )

        self.assertEqual(captured["specs"], layout.all_specs)
        self.assertIsNone(captured["sync"])
        self.assertIs(captured["live_stream"], manager.live_stream)
        self.assertEqual(
            captured["live_camera_names"],
            {"camera_front_wide"},
        )

    def test_full_sensor_mode_does_not_duplicate_lidar(self):
        settings = types.SimpleNamespace(
            enable_data_recording=False,
            enable_lidar_fusion=True,
            output_folder=None,
            camera_width=800,
            camera_height=600,
            camera_fov=90.0,
        )
        camera = types.SimpleNamespace(name="camera_front_wide")
        radar = types.SimpleNamespace(name="radar_front_long")
        lidar = types.SimpleNamespace(name="lidar_roof")
        layout = types.SimpleNamespace(
            control_specs=(camera, radar),
            lidar=lidar,
            all_specs=(camera, radar, lidar),
            sensor_names=[camera.name, radar.name, lidar.name],
        )
        captured = {}

        def fake_spawn_layout(**arguments):
            captured.update(arguments)
            return []

        manager = SensorManager(settings)
        with (
            patch(
                "carla_app.sensors.manager.build_sensor_layout",
                return_value=layout,
            ),
            patch(
                "carla_app.sensors.manager.spawn_layout",
                side_effect=fake_spawn_layout,
            ),
            redirect_stdout(io.StringIO()),
        ):
            manager.start(object(), object(), 0.05)

        self.assertEqual(captured["specs"], layout.all_specs)
        self.assertEqual(captured["specs"].count(lidar), 1)
        self.assertIsNotNone(captured["live_stream"])
        self.assertEqual(
            captured["live_camera_names"],
            {"camera_front_wide"},
        )

    def test_fresh_imu_and_gnss_are_added_to_vehicle_state(self):
        settings = types.SimpleNamespace(
            enable_data_recording=False,
            enable_bev=True,
            sensor_mode="control",
            camera_wait_timeout_ms=0.0,
            output_folder=None,
        )
        manager = SensorManager(settings)
        manager.layout = types.SimpleNamespace(
            imu=types.SimpleNamespace(name="imu_cg"),
            gnss=types.SimpleNamespace(name="gnss_roof"),
        )
        manager.live_stream.push(
            "imu_cg",
            20,
            {
                "accelerometer": {"x": 0.0, "y": 1.25, "z": 9.81},
                "gyroscope": {"x": 0.0, "y": 0.0, "z": 0.18},
                "compass": 0.5,
            },
        )
        manager.live_stream.push(
            "gnss_roof",
            20,
            {"latitude": 41.0, "longitude": 29.0, "altitude": 10.0},
        )

        state = manager.enrich_vehicle_state(
            {"speed_mps": 8.0},
            frame_id=21,
        )

        self.assertAlmostEqual(
            state["imu_lateral_acceleration_mps2"],
            1.25,
        )
        self.assertAlmostEqual(state["imu_yaw_rate_radps"], 0.18)
        self.assertEqual(state["imu_frame_id"], 20)
        self.assertEqual(state["gnss_frame_id"], 20)

    def test_bev_mode_spawns_all_sensors_without_recording(self):
        settings = types.SimpleNamespace(
            enable_data_recording=False,
            enable_bev=True,
            sensor_mode="bev",
            output_folder=None,
            camera_width=800,
            camera_height=600,
            camera_fov=90.0,
        )
        camera = types.SimpleNamespace(name="camera_front_wide")
        radar = types.SimpleNamespace(name="radar_front_long")
        lidar = types.SimpleNamespace(name="lidar_roof")
        layout = types.SimpleNamespace(
            control_specs=(camera, radar),
            all_specs=(camera, radar, lidar),
            sensor_names=[camera.name, radar.name, lidar.name],
        )
        captured = {}

        def fake_spawn_layout(**arguments):
            captured.update(arguments)
            return []

        manager = SensorManager(settings)
        with (
            patch(
                "carla_app.sensors.manager.build_sensor_layout",
                return_value=layout,
            ),
            patch(
                "carla_app.sensors.manager.spawn_layout",
                side_effect=fake_spawn_layout,
            ),
            redirect_stdout(io.StringIO()),
        ):
            manager.start(
                world=object(),
                vehicle=object(),
                fixed_delta_seconds=0.05,
            )

        self.assertEqual(captured["specs"], layout.all_specs)
        self.assertIsNone(captured["sync"])
        self.assertIs(captured["live_stream"], manager.live_stream)
        self.assertIsNotNone(manager.live_stream)
        self.assertIsNone(captured["live_camera_names"])

    def test_record_mode_spawns_all_sensors_and_keeps_live_bev_stream(self):
        settings = types.SimpleNamespace(
            enable_data_recording=True,
            enable_bev=True,
            sensor_mode="record",
            output_folder="unused",
            camera_width=800,
            camera_height=600,
            camera_fov=90.0,
        )
        sensors = tuple(
            types.SimpleNamespace(name=name)
            for name in (
                "camera_front_wide",
                "radar_front_long",
                "lidar_roof",
                "gnss_roof",
                "imu_cg",
            )
        )
        layout = types.SimpleNamespace(
            all_specs=sensors,
            sensor_names=[sensor.name for sensor in sensors],
            to_manifest=lambda _vehicle_type: {"sensors": len(sensors)},
        )
        captured = {}
        writer = types.SimpleNamespace(write_manifest=lambda _manifest: None)

        def fake_spawn_layout(**arguments):
            captured.update(arguments)
            return []

        with patch(
            "carla_app.sensors.manager.DatasetWriter",
            return_value=writer,
        ):
            manager = SensorManager(settings)
        with (
            patch(
                "carla_app.sensors.manager.build_sensor_layout",
                return_value=layout,
            ),
            patch(
                "carla_app.sensors.manager.spawn_layout",
                side_effect=fake_spawn_layout,
            ),
            redirect_stdout(io.StringIO()),
        ):
            manager.start(
                world=object(),
                vehicle=types.SimpleNamespace(type_id="vehicle.test"),
                fixed_delta_seconds=0.05,
            )

        self.assertEqual(captured["specs"], layout.all_specs)
        self.assertIs(captured["live_stream"], manager.live_stream)
        self.assertIsNotNone(captured["sync"])
        self.assertEqual(
            captured["live_camera_names"],
            {"camera_front_wide"},
        )
        self.assertEqual(
            set(captured["sync"].sensor_names),
            set(layout.sensor_names),
        )

    def test_recording_packet_contains_only_real_sensor_groups(self):
        camera = types.SimpleNamespace(name="camera_front_wide")
        radar = types.SimpleNamespace(name="radar_front_long")
        layout = types.SimpleNamespace(
            cameras=(camera,),
            radars=(radar,),
            lidar=types.SimpleNamespace(name="lidar_roof"),
            gnss=types.SimpleNamespace(name="gnss_roof"),
            imu=types.SimpleNamespace(name="imu_cg"),
            primary_camera_name="camera_front_wide",
        )

        packet = {
            "camera_front_wide": types.SimpleNamespace(
                raw_data=bytes([10, 20, 30, 255]),
                height=1,
                width=1,
            ),
            "radar_front_long": [
                types.SimpleNamespace(
                    depth=12.0,
                    velocity=-1.5,
                    azimuth=0.0,
                    altitude=0.0,
                )
            ],
            "lidar_roof": types.SimpleNamespace(
                raw_data=struct.pack("ffff", 1.0, 2.0, 3.0, 0.5),
            ),
            "gnss_roof": types.SimpleNamespace(
                latitude=41.0,
                longitude=29.0,
                altitude=10.0,
            ),
            "imu_cg": types.SimpleNamespace(
                accelerometer=types.SimpleNamespace(x=0.1, y=0.2, z=9.8),
                gyroscope=types.SimpleNamespace(x=0.0, y=0.0, z=0.1),
                compass=0.3,
            ),
        }

        result = process_packet(packet, layout)

        self.assertEqual(
            set(result),
            {"primary_camera", "cameras", "lidar", "gnss", "imu", "radars"},
        )
        front_point = result["radars"]["radar_front_long"][0]
        self.assertEqual(front_point["depth_m"], 12.0)

    def test_each_recording_listener_keeps_its_own_sensor_name(self):
        class FakeActor:
            def listen(self, callback):
                self.callback = callback

        class FakeSync:
            def __init__(self):
                self.names = []

            def push(self, sensor_name, frame_id, data):
                self.names.append(sensor_name)

        sync = FakeSync()
        first_actor = FakeActor()
        second_actor = FakeActor()
        first_sensor = types.SimpleNamespace(kind="lidar", name="lidar_roof")
        second_sensor = types.SimpleNamespace(kind="gnss", name="gnss_roof")

        start_sensor_listener(first_actor, first_sensor, sync, None, None)
        start_sensor_listener(second_actor, second_sensor, sync, None, None)
        measurement = types.SimpleNamespace(frame=10)
        first_actor.callback(measurement)
        second_actor.callback(measurement)

        self.assertEqual(sync.names, ["lidar_roof", "gnss_roof"])

    def test_surround_camera_is_sent_to_bev_live_stream(self):
        class FakeActor:
            def listen(self, callback):
                self.callback = callback

        class FakeLiveStream:
            def __init__(self):
                self.items = []

            def push(self, sensor_name, frame_id, data):
                self.items.append((sensor_name, frame_id, data))

        actor = FakeActor()
        live_stream = FakeLiveStream()
        spec = types.SimpleNamespace(
            kind="camera",
            name="camera_rear_center",
            primary=False,
        )
        start_sensor_listener(
            actor,
            spec,
            sync=None,
            camera_stream=None,
            radar_stream=None,
            live_stream=live_stream,
        )
        image = types.SimpleNamespace(
            frame=12,
            width=1,
            height=1,
            raw_data=bytes([10, 20, 30, 255]),
        )

        actor.callback(image)

        self.assertEqual(live_stream.items[0][0], "camera_rear_center")
        self.assertEqual(live_stream.items[0][1], 12)
        self.assertEqual(live_stream.items[0][2].tolist(), [[[30, 20, 10]]])

    def test_control_mode_does_not_decode_unused_surround_camera(self):
        class FakeActor:
            def listen(self, callback):
                self.callback = callback

        class FakeLiveStream:
            def __init__(self):
                self.items = []

            def push(self, sensor_name, frame_id, data):
                self.items.append((sensor_name, frame_id, data))

        actor = FakeActor()
        live_stream = FakeLiveStream()
        spec = types.SimpleNamespace(
            kind="camera",
            name="camera_rear_center",
            primary=False,
        )
        start_sensor_listener(
            actor,
            spec,
            sync=None,
            camera_stream=None,
            radar_stream=None,
            live_stream=live_stream,
            live_camera_names={"camera_front_wide"},
        )
        image = types.SimpleNamespace(
            frame=12,
            width=1,
            height=1,
            raw_data=bytes([10, 20, 30, 255]),
        )

        with patch(
            "carla_app.sensors.factory.image_to_rgb"
        ) as image_to_rgb:
            actor.callback(image)

        image_to_rgb.assert_not_called()
        self.assertEqual(live_stream.items, [])


if __name__ == "__main__":
    unittest.main()

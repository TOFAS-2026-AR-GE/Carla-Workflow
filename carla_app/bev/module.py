"""Kalibrasyon, IPM, füzyon, takip ve occupancy katmanlarını birleştirir."""

import queue
import threading

import numpy as np

from carla_app.bev.calibration import CalibrationSet
from carla_app.bev.camera_ipm import CameraIpm
from carla_app.bev.coordinate import EgoMotionCompensator, MetricGrid
from carla_app.bev.fusion import SensorFusion
from carla_app.bev.localization import LocalizationHealth
from carla_app.bev.occupancy import OccupancyGrid
from carla_app.bev.projector import BevProjector
from carla_app.bev.renderer import BevRenderer
from carla_app.bev.tracking import BevTracker
from carla_app.bev.validation import BevValidationLayer


class BevModule:
    """Sensör snapshot'ından çıkarılabilir kuş bakışı görünüm üretir."""

    def __init__(
        self,
        layout,
        width=800,
        height=600,
        fixed_delta_seconds=0.05,
        update_every_n_frames=2,
        asynchronous=False,
        render_output=True,
    ):
        self.layout = layout
        self.update_every_n_frames = max(1, int(update_every_n_frames))
        self.grid = MetricGrid(width=width, height=height)
        self.calibrations = CalibrationSet(layout)
        self.motion = EgoMotionCompensator()
        self.projector = BevProjector(
            layout,
            calibrations=self.calibrations,
            motion_compensator=self.motion,
        )
        self.camera_ipm = CameraIpm(layout, self.calibrations, self.grid)
        self.fusion = SensorFusion()
        self.localization = LocalizationHealth(layout)
        self.tracker = BevTracker(fixed_delta_seconds)
        self.occupancy = OccupancyGrid(
            forward_range_m=self.grid.forward_range_m,
            rear_range_m=self.grid.rear_range_m,
            side_range_m=self.grid.side_range_m,
            fixed_delta_seconds=fixed_delta_seconds,
        )
        self.renderer = BevRenderer(grid=self.grid)
        self.validator = BevValidationLayer()
        self.render_output = bool(render_output)

        self.asynchronous = bool(asynchronous)
        self.work_queue = queue.Queue(maxsize=1)
        self.lock = threading.Lock()
        self.latest_image = None
        self.latest_scene = None
        self.last_error = None
        self.stop_event = threading.Event()
        self.thread = None
        if self.asynchronous:
            self.thread = threading.Thread(
                target=self.worker_loop,
                name="bev-worker",
                daemon=True,
            )
            self.thread.start()

    def render(
        self,
        sensor_snapshot,
        perception_result,
        vehicle_state,
        current_frame_id=None,
        driving_state=None,
        display_mode="driving",
    ):
        """Tek BEV karesini senkron üretir; test ve bağımsız kullanım içindir."""
        current_frame_id = 0 if current_frame_id is None else int(current_frame_id)
        self.motion.remember(current_frame_id, vehicle_state)

        scene = self.projector.build_scene(
            sensor_snapshot,
            perception_result,
            vehicle_state,
            current_frame_id,
        )
        scene["localization"] = self.localization.evaluate(
            sensor_snapshot,
            current_frame_id,
        )
        scene["driving_state"] = dict(driving_state or {})
        ipm_image = None
        coverage = None
        ipm_cameras = []
        if self.render_output:
            camera_results = self.camera_results_for_ipm(
                sensor_snapshot,
                perception_result,
                current_frame_id,
            )
            ipm_image, coverage, ipm_cameras = self.camera_ipm.build_mosaic(
                camera_results,
                motion_compensator=self.motion,
                current_frame_id=current_frame_id,
            )

        object_lidar_points = self.object_lidar_points(
            scene["lidar_points"],
            scene["ground_z_m"],
        )
        fused_objects = self.fusion.build_fused_objects(
            scene["camera_objects"],
            scene["radar_points"],
            object_lidar_points,
            lidar_frame_id=scene["lidar_frame_id"],
        )
        ego_pose = self.motion.get_pose(current_frame_id)
        tracks = self.tracker.update(
            fused_objects,
            current_frame_id,
            ego_pose,
            evidence_frame_id=scene["evidence_frame_id"],
        )
        occupancy = self.occupancy.update(
            scene["lidar_points"],
            scene["lidar_origin"],
            scene["radar_points"],
            tracks,
            scene["ground_z_m"],
            self.motion,
            current_frame_id,
            measurement_frames=scene["measurement_frames"],
        )

        scene["ipm_image"] = ipm_image
        scene["ipm_coverage"] = coverage
        scene["ipm_cameras"] = ipm_cameras
        scene["fused_objects"] = fused_objects
        scene["tracks"] = tracks
        scene["occupancy"] = occupancy
        image = None
        if self.render_output:
            image = self.renderer.render(
                scene,
                current_frame_id,
                display_mode=display_mode,
            )
        with self.lock:
            self.latest_scene = scene
        return image

    def submit(
        self,
        sensor_snapshot,
        perception_result,
        vehicle_state,
        current_frame_id,
        driving_state=None,
        display_mode="driving",
    ):
        """En yeni BEV işini ana kontrol döngüsünü bekletmeden kuyruğa koyar."""
        self.motion.remember(current_frame_id, vehicle_state)
        if int(current_frame_id) % self.update_every_n_frames != 0:
            return
        if not self.asynchronous:
            image = self.render(
                sensor_snapshot,
                perception_result,
                vehicle_state,
                current_frame_id,
                driving_state,
                display_mode,
            )
            with self.lock:
                self.latest_image = image
            return

        item = {
            "sensor_snapshot": sensor_snapshot,
            "perception_result": perception_result,
            "vehicle_state": vehicle_state,
            "current_frame_id": int(current_frame_id),
            "driving_state": driving_state,
            "display_mode": str(display_mode),
        }
        try:
            self.work_queue.put_nowait(item)
            return
        except queue.Full:
            pass

        try:
            self.work_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.work_queue.put_nowait(item)
        except queue.Full:
            pass

    def get_latest(self):
        with self.lock:
            return self.latest_image

    def get_latest_scene(self):
        with self.lock:
            return self.latest_scene

    def validate(
        self,
        current_frame_id,
        lead_vehicle=None,
        emergency_obstacle=None,
    ):
        """En son BEV sahnesiyle kontrol algısını yalnızca çapraz doğrular."""
        with self.lock:
            scene = self.latest_scene
        return self.validator.evaluate(
            scene,
            current_frame_id,
            lead_vehicle=lead_vehicle,
            emergency_obstacle=emergency_obstacle,
        )

    def contribute(self, current_frame_id, vehicle_state, lead_vehicle=None):
        """Güvenli BEV lead recovery sonucunu kontrol girişi için döndürür."""
        with self.lock:
            scene = self.latest_scene
        return self.validator.contribute(
            scene,
            current_frame_id,
            vehicle_state,
            lead_vehicle=lead_vehicle,
        )

    def worker_loop(self):
        while not self.stop_event.is_set():
            try:
                item = self.work_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return

            try:
                image = self.render(
                    item["sensor_snapshot"],
                    item["perception_result"],
                    item["vehicle_state"],
                    item["current_frame_id"],
                    item.get("driving_state"),
                    item.get("display_mode", "driving"),
                )
                with self.lock:
                    self.latest_image = image
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if message != self.last_error:
                    print(f"[ERROR] BEV modülü: {message}")
                self.last_error = message
                continue
            self.last_error = None

    def stop(self):
        self.stop_event.set()
        if self.thread is None:
            return
        try:
            self.work_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.work_queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=3.0)
        if self.thread.is_alive():
            print("[WARN] BEV işçisi 3 saniyede durmadı.")

    def object_lidar_points(self, lidar_points, ground_z_m):
        points = np.asarray(lidar_points)
        if points.ndim != 2 or points.shape[1] < 3 or len(points) == 0:
            return np.empty((0, 3), dtype=np.float32)
        minimum_height = float(ground_z_m) + 0.18
        maximum_height = float(ground_z_m) + 3.50
        mask = points[:, 2] >= minimum_height
        mask &= points[:, 2] <= maximum_height

        geometry = self.layout.vehicle_geometry
        inside_ego = np.abs(points[:, 0]) <= float(geometry["half_length_m"]) + 0.3
        inside_ego &= np.abs(points[:, 1]) <= float(geometry["half_width_m"]) + 0.3
        mask &= ~inside_ego
        return points[mask]

    def camera_results_for_ipm(
        self,
        sensor_snapshot,
        perception_result,
        current_frame_id=None,
    ):
        """IPM için algılamadan bağımsız en güncel yedi kamerayı seçer."""
        results = {}
        if perception_result:
            for camera_name, result in perception_result.get(
                "camera_results", {}
            ).items():
                if self.projector.frame_is_fresh(
                    result.get("frame_id"),
                    current_frame_id,
                ):
                    results[camera_name] = result

        for camera in self.layout.cameras:
            entry = sensor_snapshot.get(camera.name)
            if entry is None:
                continue
            if not self.projector.frame_is_fresh(
                entry.get("frame_id"),
                current_frame_id,
            ):
                continue
            results[camera.name] = {
                "camera_name": camera.name,
                "frame_id": int(entry["frame_id"]),
                "image": entry["data"],
            }
        return results

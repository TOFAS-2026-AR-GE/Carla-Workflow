"""Birleştirilmiş BEV nesnelerini Kalman filtresiyle zaman içinde izler."""

import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from carla_app.bev.association import measurement_covariance


class BevTracker:
    """Nesneleri dünya koordinatında izleyerek ego hareketini telafi eder."""

    def __init__(self, fixed_delta_seconds=0.05):
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.tracks = {}
        self.next_track_id = 1
        self.used_measurements = {}
        self.last_evidence_frame_id = None

    def update(
        self,
        measurements,
        current_frame_id,
        ego_pose,
        evidence_frame_id=None,
    ):
        current_frame_id = int(current_frame_id)
        self.predict_tracks(current_frame_id)
        evidence_delta = self.evidence_delta(evidence_frame_id)
        evidence_advanced = evidence_delta > 0

        fresh_measurements = []
        for measurement in measurements:
            key = measurement.get("measurement_key")
            if key and key in self.used_measurements:
                continue
            prepared = dict(measurement)
            world_x, world_y = self.ego_to_world(
                prepared["x_m"],
                prepared["y_m"],
                ego_pose,
            )
            prepared["world_x"] = world_x
            prepared["world_y"] = world_y
            fresh_measurements.append(prepared)
            if key:
                self.used_measurements[key] = current_frame_id

        self.remove_old_measurement_keys(current_frame_id)
        matches, unmatched_measurements = self.match_measurements(fresh_measurements)

        matched_track_ids = set()
        for track_id, measurement_index in matches:
            measurement = fresh_measurements[measurement_index]
            self.correct_track(self.tracks[track_id], measurement)
            matched_track_ids.add(track_id)

        for track_id, track in self.tracks.items():
            if evidence_advanced and track_id not in matched_track_ids:
                track["misses"] += evidence_delta
                track["confidence"] *= 0.92**evidence_delta

        for measurement_index in unmatched_measurements:
            self.create_track(fresh_measurements[measurement_index], current_frame_id)

        if evidence_advanced and evidence_frame_id is not None:
            self.last_evidence_frame_id = int(evidence_frame_id)

        expired = []
        for track_id, track in self.tracks.items():
            if track["misses"] > 12:
                expired.append(track_id)
        for track_id in expired:
            del self.tracks[track_id]

        return self.build_output(ego_pose)

    def predict_tracks(self, current_frame_id):
        for track in self.tracks.values():
            frame_difference = max(0, current_frame_id - track["frame_id"])
            dt = frame_difference * self.fixed_delta_seconds
            if dt <= 0.0:
                continue

            transition = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            acceleration_noise = 2.0
            process_noise = acceleration_noise * np.array(
                [
                    [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
                    [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
                    [dt**3 / 2.0, 0.0, dt**2, 0.0],
                    [0.0, dt**3 / 2.0, 0.0, dt**2],
                ],
                dtype=np.float64,
            )
            track["state"] = transition @ track["state"]
            track["covariance"] = (
                transition @ track["covariance"] @ transition.T
                + process_noise
            )
            track["frame_id"] = current_frame_id

    def match_measurements(self, measurements):
        track_ids = sorted(self.tracks)
        if not track_ids or not measurements:
            return [], list(range(len(measurements)))

        invalid_cost = 1e9
        costs = np.full(
            (len(track_ids), len(measurements)),
            invalid_cost,
            dtype=np.float64,
        )
        for track_index, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            track_covariance = np.asarray(
                track.get("covariance", np.diag([4.0, 4.0, 16.0, 16.0])),
                dtype=np.float64,
            )[:2, :2]
            for measurement_index, measurement in enumerate(measurements):
                distance = math.hypot(
                    float(track["state"][0]) - measurement["world_x"],
                    float(track["state"][1]) - measurement["world_y"],
                )
                gate = 3.0 + 1.5 * float(measurement["uncertainty_m"])
                if distance > min(6.0, gate):
                    continue
                innovation = np.array(
                    [
                        measurement["world_x"] - float(track["state"][0]),
                        measurement["world_y"] - float(track["state"][1]),
                    ],
                    dtype=np.float64,
                )
                innovation_covariance = (
                    track_covariance + measurement_covariance(measurement)
                )
                innovation_covariance += np.eye(2, dtype=np.float64) * 0.04
                mahalanobis = float(
                    innovation.T
                    @ np.linalg.solve(innovation_covariance, innovation)
                )
                if mahalanobis > 9.21:
                    continue
                costs[track_index, measurement_index] = mahalanobis + 1e-3 * distance

        rows, columns = linear_sum_assignment(costs)
        used_measurements = set()
        matches = []
        for row, measurement_index in zip(rows, columns):
            if costs[row, measurement_index] >= invalid_cost:
                continue
            used_measurements.add(measurement_index)
            matches.append((track_ids[row], int(measurement_index)))

        unmatched = []
        for index in range(len(measurements)):
            if index not in used_measurements:
                unmatched.append(index)
        return matches, unmatched

    def evidence_delta(self, evidence_frame_id):
        if evidence_frame_id is None:
            return 1
        if self.last_evidence_frame_id is None:
            return 1
        return max(0, int(evidence_frame_id) - self.last_evidence_frame_id)

    def correct_track(self, track, measurement):
        observation = np.array(
            [measurement["world_x"], measurement["world_y"]],
            dtype=np.float64,
        )
        observation_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        measurement_noise = measurement_covariance(measurement)

        innovation = observation - observation_matrix @ track["state"]
        innovation_covariance = (
            observation_matrix
            @ track["covariance"]
            @ observation_matrix.T
            + measurement_noise
        )
        kalman_gain = (
            track["covariance"]
            @ observation_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        track["state"] = track["state"] + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        residual_matrix = identity - kalman_gain @ observation_matrix
        track["covariance"] = (
            residual_matrix @ track["covariance"] @ residual_matrix.T
            + kalman_gain @ measurement_noise @ kalman_gain.T
        )

        track["hits"] += 1
        track["misses"] = 0
        track["confidence"] = (
            0.65 * track["confidence"]
            + 0.35 * float(measurement["confidence"])
        )
        track["class_name"] = measurement["class_name"]
        track["sources"] = measurement["sources"]
        track["sensor_names"] = measurement["sensor_names"]
        track["length_m"] = measurement["length_m"]
        track["width_m"] = measurement["width_m"]
        track["source_frames"].update(measurement.get("source_frames", {}))
        if measurement.get("frame_ids"):
            track["last_measurement_frame_id"] = max(measurement["frame_ids"])

    def create_track(self, measurement, current_frame_id):
        state = np.array(
            [measurement["world_x"], measurement["world_y"], 0.0, 0.0],
            dtype=np.float64,
        )
        position_covariance = measurement_covariance(measurement)
        covariance = np.zeros((4, 4), dtype=np.float64)
        covariance[:2, :2] = position_covariance
        covariance[2:, 2:] = np.eye(2, dtype=np.float64) * 16.0
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = {
            "track_id": track_id,
            "state": state,
            "covariance": covariance,
            "frame_id": int(current_frame_id),
            "hits": 1,
            "misses": 0,
            "confidence": float(measurement["confidence"]),
            "class_name": measurement["class_name"],
            "sources": measurement["sources"],
            "sensor_names": measurement["sensor_names"],
            "length_m": measurement["length_m"],
            "width_m": measurement["width_m"],
            "source_frames": dict(measurement.get("source_frames", {})),
            "last_measurement_frame_id": max(
                measurement.get("frame_ids", [current_frame_id])
            ),
        }

    def build_output(self, ego_pose):
        output = []
        for track_id in sorted(self.tracks):
            track = self.tracks[track_id]
            x_m, y_m = self.world_to_ego(
                track["state"][0],
                track["state"][1],
                ego_pose,
            )
            velocity_x, velocity_y = self.world_vector_to_ego(
                track["state"][2],
                track["state"][3],
                ego_pose,
            )
            uncertainty = math.sqrt(
                max(track["covariance"][0, 0], track["covariance"][1, 1])
            )
            output.append(
                {
                    "track_id": track_id,
                    "x_m": x_m,
                    "y_m": y_m,
                    "velocity_x_mps": velocity_x,
                    "velocity_y_mps": velocity_y,
                    "uncertainty_m": uncertainty,
                    "confidence": track["confidence"],
                    "class_name": track["class_name"],
                    "sources": track["sources"],
                    "sensor_names": track["sensor_names"],
                    "length_m": track["length_m"],
                    "width_m": track["width_m"],
                    "confirmed": track["hits"] >= 2,
                    "misses": track["misses"],
                    "source_frames": dict(track.get("source_frames", {})),
                    "last_measurement_frame_id": track.get(
                        "last_measurement_frame_id"
                    ),
                }
            )
        return output

    def remove_old_measurement_keys(self, current_frame_id):
        old_keys = []
        for key, frame_id in self.used_measurements.items():
            if current_frame_id - frame_id > 160:
                old_keys.append(key)
        for key in old_keys:
            del self.used_measurements[key]

    def ego_to_world(self, x_m, y_m, ego_pose):
        if ego_pose is None:
            return float(x_m), float(y_m)
        yaw = math.radians(float(ego_pose["yaw_deg"]))
        world_x = ego_pose["x"] + x_m * math.cos(yaw) - y_m * math.sin(yaw)
        world_y = ego_pose["y"] + x_m * math.sin(yaw) + y_m * math.cos(yaw)
        return float(world_x), float(world_y)

    def world_to_ego(self, world_x, world_y, ego_pose):
        if ego_pose is None:
            return float(world_x), float(world_y)
        yaw = math.radians(float(ego_pose["yaw_deg"]))
        dx = float(world_x) - ego_pose["x"]
        dy = float(world_y) - ego_pose["y"]
        forward = dx * math.cos(yaw) + dy * math.sin(yaw)
        right = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return float(forward), float(right)

    def world_vector_to_ego(self, world_x, world_y, ego_pose):
        if ego_pose is None:
            return float(world_x), float(world_y)
        yaw = math.radians(float(ego_pose["yaw_deg"]))
        forward = world_x * math.cos(yaw) + world_y * math.sin(yaw)
        right = -world_x * math.sin(yaw) + world_y * math.cos(yaw)
        return float(forward), float(right)

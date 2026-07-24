"""Covariance, Mahalanobis kapısı ve Hungarian eşleştirmeli çoklu takip.

Durum vektörü ``[x, y, vx, vy]`` biçimindedir. Ölçüm kovaryansı verilmezse
mesafeye göre güvenli bir varsayılan üretilir. Dış API eski sade takipçiyle
uyumludur; ``LeadVehicleTracker`` değişmeden bu sınıfı kullanabilir.
"""

import math

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # Küçük test ortamlarında anlaşılır bir fallback kullanılır.
    linear_sum_assignment = None


class Axis1DKalman:
    """Eski test ve yardımcı kodlar için korunan tek eksenli filtre."""

    def __init__(self, position, process_noise=1.0, measurement_noise=1.5):
        self.pos = float(position)
        self.vel = 0.0
        self.p_pp = 2.0
        self.p_pv = 0.0
        self.p_vv = 8.0
        self.q = float(process_noise)
        self.r = float(measurement_noise)

    def predict(self, dt):
        dt = max(0.0, float(dt))
        self.pos += self.vel * dt
        p_pp = self.p_pp + 2.0 * dt * self.p_pv + dt * dt * self.p_vv
        p_pv = self.p_pv + dt * self.p_vv
        p_vv = self.p_vv
        self.p_pp = p_pp + self.q * dt**4 / 4.0
        self.p_pv = p_pv + self.q * dt**3 / 2.0
        self.p_vv = p_vv + self.q * dt**2

    def update(self, measured_position):
        residual = float(measured_position) - self.pos
        innovation = self.p_pp + self.r
        k_pos = self.p_pp / innovation
        k_vel = self.p_pv / innovation
        old_p_pv = self.p_pv
        self.pos += k_pos * residual
        self.vel += k_vel * residual
        self.p_pp = max(1e-9, (1.0 - k_pos) * self.p_pp)
        self.p_pv = (1.0 - k_pos) * old_p_pv
        self.p_vv = max(1e-9, self.p_vv - k_vel * old_p_pv)


class AxisStateView:
    """Eski ``track.kx`` ve ``track.ky`` okuma alanlarını korur."""

    def __init__(self, track, position_index, velocity_index):
        self.track = track
        self.position_index = int(position_index)
        self.velocity_index = int(velocity_index)

    @property
    def pos(self):
        return float(self.track.state[self.position_index])

    @property
    def vel(self):
        return float(self.track.state[self.velocity_index])

    @property
    def p_pp(self):
        return float(
            self.track.covariance[self.position_index, self.position_index]
        )

    @property
    def p_pv(self):
        return float(
            self.track.covariance[self.position_index, self.velocity_index]
        )

    @property
    def p_vv(self):
        return float(
            self.track.covariance[self.velocity_index, self.velocity_index]
        )


class Track:
    """Tek hedef için 2B sabit hızlı Kalman durumunu tutar."""

    def __init__(
        self,
        track_id,
        x,
        y,
        class_name,
        process_noise=2.0,
        measurement_noise=1.5,
    ):
        self.id = int(track_id)
        self.class_name = str(class_name)
        self.process_noise = max(0.01, float(process_noise))
        self.default_measurement_std = max(0.10, float(measurement_noise))
        self.state = np.array([float(x), float(y), 0.0, 0.0], dtype=np.float64)
        self.covariance = np.diag([2.0, 2.0, 16.0, 16.0]).astype(np.float64)
        self.kx = AxisStateView(self, 0, 2)
        self.ky = AxisStateView(self, 1, 3)

        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.hit_count = 1
        self.miss_count = 0
        self.confirmed = False
        self.last_range_m = None
        self.last_bearing_deg = None
        self.last_relative_velocity_mps = None
        self.last_measurement_frame_id = None
        self.last_mahalanobis_sq = None
        self.last_innovation = None

    def predict(self, dt):
        dt = max(0.001, float(dt))
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.process_noise
        process_covariance = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + process_covariance
        )
        self.copy_filter_values()

    def measurement_covariance(self, measurement=None):
        measurement = measurement or {}
        covariance = measurement.get("covariance")
        if covariance is not None:
            array = np.asarray(covariance, dtype=np.float64)
            if array.shape == (2, 2) and np.isfinite(array).all():
                return array

        standard_deviation = measurement.get("position_std_m")
        try:
            standard_deviation = float(standard_deviation)
        except (TypeError, ValueError):
            standard_deviation = self.default_measurement_std
        standard_deviation = max(0.10, standard_deviation)
        return np.eye(2, dtype=np.float64) * standard_deviation**2

    def innovation(self, measurement):
        observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        measured = np.array(
            [float(measurement["x"]), float(measurement["y"])],
            dtype=np.float64,
        )
        residual = measured - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T
            + self.measurement_covariance(measurement)
        )
        return residual, innovation_covariance, observation

    def mahalanobis_sq(self, measurement):
        residual, innovation_covariance, _ = self.innovation(measurement)
        try:
            value = float(
                residual.T @ np.linalg.solve(innovation_covariance, residual)
            )
        except np.linalg.LinAlgError:
            return math.inf
        return value if math.isfinite(value) else math.inf

    def update(self, measured_x, measured_y=None, measurement=None):
        if measurement is None:
            measurement = {
                "x": float(measured_x),
                "y": float(measured_y),
            }
        residual, innovation_covariance, observation = self.innovation(measurement)
        measurement_covariance = self.measurement_covariance(measurement)
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ residual
        identity = np.eye(4, dtype=np.float64)
        correction = identity - gain @ observation
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        self.copy_filter_values()
        self.hit_count += 1
        self.miss_count = 0
        if self.hit_count >= 3:
            self.confirmed = True
        self.last_innovation = residual.copy()
        self.last_mahalanobis_sq = float(
            residual.T @ np.linalg.solve(innovation_covariance, residual)
        )

    def mark_missed(self):
        self.miss_count += 1

    def copy_filter_values(self):
        self.x = float(self.state[0])
        self.y = float(self.state[1])
        self.vx = float(self.state[2])
        self.vy = float(self.state[3])

    @property
    def position_covariance(self):
        return self.covariance[:2, :2].copy()

    @property
    def position_std_m(self):
        return math.sqrt(
            max(0.0, float(self.covariance[0, 0] + self.covariance[1, 1]))
        )


def polar_to_world(range_m, bearing_deg, ego_x, ego_y, ego_yaw_deg):
    """Araca göre mesafe-açı ölçümünü dünya koordinatına çevirir."""
    bearing_rad = math.radians(float(bearing_deg))
    yaw_rad = math.radians(float(ego_yaw_deg))
    x_local = float(range_m) * math.cos(bearing_rad)
    y_local = float(range_m) * math.sin(bearing_rad)
    world_x = float(ego_x) + x_local * math.cos(yaw_rad) - y_local * math.sin(
        yaw_rad
    )
    world_y = float(ego_y) + x_local * math.sin(yaw_rad) + y_local * math.cos(
        yaw_rad
    )
    return world_x, world_y


class Tracker:
    """Hungarian ataması ve Mahalanobis kapısıyla ölçümleri takip eder."""

    def __init__(
        self,
        gate_distance_m=5.0,
        max_misses=5,
        mahalanobis_gate=9.21,
        process_noise=2.0,
        measurement_noise=1.5,
    ):
        self.tracks = []
        self.next_id = 1
        self.gate_distance_m = max(0.5, float(gate_distance_m))
        self.max_misses = max(0, int(max_misses))
        self.mahalanobis_gate = max(1.0, float(mahalanobis_gate))
        self.process_noise = max(0.01, float(process_noise))
        self.measurement_noise = max(0.10, float(measurement_noise))

    def step(self, dt, measurements):
        measurements = list(measurements or [])
        for track in self.tracks:
            track.predict(dt)

        assignments = self.assign_measurements(measurements)
        used_tracks = set()
        used_measurements = set()

        for track_index, measurement_index, mahalanobis_sq in assignments:
            track = self.tracks[track_index]
            measurement = measurements[measurement_index]
            track.update(
                measurement["x"],
                measurement["y"],
                measurement=measurement,
            )
            track.last_mahalanobis_sq = mahalanobis_sq
            track.class_name = measurement.get("class_name", track.class_name)
            track.last_range_m = measurement.get("range_m")
            track.last_bearing_deg = measurement.get("bearing_deg")
            track.last_relative_velocity_mps = measurement.get(
                "relative_velocity_mps"
            )
            track.last_measurement_frame_id = measurement.get(
                "measurement_frame_id"
            )
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in used_tracks:
                track.mark_missed()

        for measurement_index, measurement in enumerate(measurements):
            if measurement_index in used_measurements:
                continue
            new_track = Track(
                self.next_id,
                measurement["x"],
                measurement["y"],
                measurement.get("class_name", "unknown"),
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            new_track.last_range_m = measurement.get("range_m")
            new_track.last_bearing_deg = measurement.get("bearing_deg")
            new_track.last_relative_velocity_mps = measurement.get(
                "relative_velocity_mps"
            )
            new_track.last_measurement_frame_id = measurement.get(
                "measurement_frame_id"
            )
            covariance = new_track.measurement_covariance(measurement)
            new_track.covariance[0:2, 0:2] = covariance
            self.tracks.append(new_track)
            self.next_id += 1

        self.tracks = [
            track for track in self.tracks if track.miss_count <= self.max_misses
        ]
        return self.tracks

    def assign_measurements(self, measurements):
        if not self.tracks or not measurements:
            return []

        large_cost = 1e6
        cost = np.full(
            (len(self.tracks), len(measurements)),
            large_cost,
            dtype=np.float64,
        )
        mahalanobis_values = {}

        for track_index, track in enumerate(self.tracks):
            for measurement_index, measurement in enumerate(measurements):
                euclidean_distance = math.hypot(
                    track.x - float(measurement["x"]),
                    track.y - float(measurement["y"]),
                )
                if euclidean_distance > self.gate_distance_m * 2.0:
                    continue
                mahalanobis_sq = track.mahalanobis_sq(measurement)
                if mahalanobis_sq > self.mahalanobis_gate:
                    continue
                class_penalty = 0.0
                measurement_class = str(measurement.get("class_name", "unknown"))
                if (
                    track.class_name not in {"unknown", measurement_class}
                    and measurement_class != "unknown"
                ):
                    class_penalty = 1.0
                cost[track_index, measurement_index] = mahalanobis_sq + class_penalty
                mahalanobis_values[(track_index, measurement_index)] = mahalanobis_sq

        if linear_sum_assignment is not None:
            row_indexes, column_indexes = linear_sum_assignment(cost)
            pairs = zip(row_indexes.tolist(), column_indexes.tolist())
        else:
            pairs = self.greedy_fallback(cost)

        assignments = []
        for track_index, measurement_index in pairs:
            if cost[track_index, measurement_index] >= large_cost:
                continue
            assignments.append(
                (
                    int(track_index),
                    int(measurement_index),
                    float(mahalanobis_values[(track_index, measurement_index)]),
                )
            )
        return assignments

    def greedy_fallback(self, cost):
        candidates = []
        for track_index in range(cost.shape[0]):
            for measurement_index in range(cost.shape[1]):
                candidates.append(
                    (cost[track_index, measurement_index], track_index, measurement_index)
                )
        candidates.sort()
        used_tracks = set()
        used_measurements = set()
        result = []
        for value, track_index, measurement_index in candidates:
            if value >= 1e6:
                break
            if track_index in used_tracks or measurement_index in used_measurements:
                continue
            used_tracks.add(track_index)
            used_measurements.add(measurement_index)
            result.append((track_index, measurement_index))
        return result

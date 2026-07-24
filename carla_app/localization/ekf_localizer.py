"""GNSS, IMU ve isteğe bağlı odometriyi birleştiren sade EKF.

Durum vektörü ``[x, y, hiz, yaw, ivme_bias, gyro_bias]`` biçimindedir.
Bütün açılar radyan, konumlar metre, hızlar m/s cinsindedir.
"""

import math

import numpy as np


def wrap_angle(angle):
    """Açıyı ``[-pi, pi)`` aralığına taşır."""
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class ExtendedKalmanLocalizer:
    """Düzlemsel araç hareketi için okunabilir bir EKF uygular."""

    def __init__(
        self,
        dt,
        process_position_std_m=0.20,
        process_speed_std_mps=0.60,
        process_yaw_std_rad=0.035,
        gnss_position_std_m=1.50,
        compass_std_rad=0.10,
        odometry_speed_std_mps=0.45,
    ):
        self.dt = max(0.01, float(dt))
        self.process_position_std_m = max(0.01, float(process_position_std_m))
        self.process_speed_std_mps = max(0.01, float(process_speed_std_mps))
        self.process_yaw_std_rad = max(0.001, float(process_yaw_std_rad))
        self.gnss_position_std_m = max(0.10, float(gnss_position_std_m))
        self.compass_std_rad = max(0.01, float(compass_std_rad))
        self.odometry_speed_std_mps = max(0.05, float(odometry_speed_std_mps))

        self.state = np.zeros(6, dtype=np.float64)
        self.covariance = np.eye(6, dtype=np.float64)
        self.initialized = False
        self.last_frame_id = None
        self.last_gnss_frame_id = None
        self.last_compass_frame_id = None
        self.last_odometry_frame_id = None
        self.rejected_gnss_updates = 0

    def initialize(self, x_m, y_m, yaw_rad, frame_id, speed_mps=0.0):
        """İlk GNSS konumu ve pusula yönüyle filtreyi başlatır."""
        self.state[:] = (
            float(x_m),
            float(y_m),
            max(0.0, float(speed_mps)),
            wrap_angle(yaw_rad),
            0.0,
            0.0,
        )
        self.covariance = np.diag(
            [
                self.gnss_position_std_m**2,
                self.gnss_position_std_m**2,
                4.0,
                self.compass_std_rad**2,
                0.50,
                0.05,
            ]
        ).astype(np.float64)
        self.initialized = True
        self.last_frame_id = int(frame_id)

    def predict(self, frame_id, imu=None):
        """IMU ivmesi ve yaw-rate ile durumu bir sonraki kareye taşır."""
        if not self.initialized:
            return

        frame_id = int(frame_id)
        if self.last_frame_id is None:
            self.last_frame_id = frame_id
            return

        frame_delta = max(0, frame_id - int(self.last_frame_id))
        if frame_delta == 0:
            return
        dt = min(0.50, frame_delta * self.dt)
        self.last_frame_id = frame_id

        acceleration = 0.0
        yaw_rate = 0.0
        if imu:
            accelerometer = imu.get("accelerometer", {}) or {}
            gyroscope = imu.get("gyroscope", {}) or {}
            try:
                acceleration = float(accelerometer.get("x", 0.0))
                yaw_rate = float(gyroscope.get("z", 0.0))
            except (TypeError, ValueError):
                acceleration = 0.0
                yaw_rate = 0.0

        x_m, y_m, speed_mps, yaw_rad, accel_bias, gyro_bias = self.state
        corrected_acceleration = acceleration - accel_bias
        corrected_yaw_rate = yaw_rate - gyro_bias

        yaw_mid = wrap_angle(yaw_rad + 0.5 * corrected_yaw_rate * dt)
        distance = speed_mps * dt + 0.5 * corrected_acceleration * dt * dt
        new_x = x_m + distance * math.cos(yaw_mid)
        new_y = y_m + distance * math.sin(yaw_mid)
        new_speed = max(0.0, speed_mps + corrected_acceleration * dt)
        new_yaw = wrap_angle(yaw_rad + corrected_yaw_rate * dt)

        self.state[:] = (
            new_x,
            new_y,
            new_speed,
            new_yaw,
            accel_bias,
            gyro_bias,
        )

        jacobian = np.eye(6, dtype=np.float64)
        jacobian[0, 2] = dt * math.cos(yaw_mid)
        jacobian[1, 2] = dt * math.sin(yaw_mid)
        jacobian[0, 3] = -distance * math.sin(yaw_mid)
        jacobian[1, 3] = distance * math.cos(yaw_mid)
        jacobian[0, 4] = -0.5 * dt * dt * math.cos(yaw_mid)
        jacobian[1, 4] = -0.5 * dt * dt * math.sin(yaw_mid)
        jacobian[2, 4] = -dt
        jacobian[3, 5] = -dt

        position_noise = (self.process_position_std_m * max(dt, self.dt)) ** 2
        speed_noise = (self.process_speed_std_mps * max(dt, self.dt)) ** 2
        yaw_noise = (self.process_yaw_std_rad * max(dt, self.dt)) ** 2
        process_noise = np.diag(
            [
                position_noise,
                position_noise,
                speed_noise,
                yaw_noise,
                1e-5 * dt,
                1e-6 * dt,
            ]
        )
        self.covariance = jacobian @ self.covariance @ jacobian.T + process_noise
        self._symmetrize_covariance()

    def update_gnss(self, x_m, y_m, frame_id, std_m=None):
        """GNSS konumunu 2B Mahalanobis kapısıyla filtreye ekler."""
        if not self.initialized:
            raise RuntimeError("EKF başlatılmadan GNSS güncellemesi yapılamaz.")
        if self.last_gnss_frame_id == int(frame_id):
            return False

        measurement = np.array([float(x_m), float(y_m)], dtype=np.float64)
        observation = np.zeros((2, 6), dtype=np.float64)
        observation[0, 0] = 1.0
        observation[1, 1] = 1.0
        standard_deviation = self.gnss_position_std_m if std_m is None else max(
            0.10,
            float(std_m),
        )
        measurement_noise = np.eye(2, dtype=np.float64) * standard_deviation**2

        residual = measurement - observation @ self.state
        innovation = observation @ self.covariance @ observation.T + measurement_noise
        mahalanobis_sq = float(residual.T @ np.linalg.solve(innovation, residual))

        # 2 serbestlik derecesinde yaklaşık yüzde 99 kapı.
        if mahalanobis_sq > 9.21:
            self.rejected_gnss_updates += 1
            self.last_gnss_frame_id = int(frame_id)
            return False

        self._linear_update(residual, observation, measurement_noise)
        self.last_gnss_frame_id = int(frame_id)
        return True

    def update_compass(self, yaw_rad, frame_id, std_rad=None):
        """Pusula yönünü açı sarmalamasını koruyarak ekler."""
        if not self.initialized:
            raise RuntimeError("EKF başlatılmadan pusula güncellemesi yapılamaz.")
        if self.last_compass_frame_id == int(frame_id):
            return False

        standard_deviation = self.compass_std_rad if std_rad is None else max(
            0.01,
            float(std_rad),
        )
        observation = np.zeros((1, 6), dtype=np.float64)
        observation[0, 3] = 1.0
        residual = np.array(
            [wrap_angle(float(yaw_rad) - float(self.state[3]))],
            dtype=np.float64,
        )
        measurement_noise = np.array([[standard_deviation**2]], dtype=np.float64)
        self._linear_update(residual, observation, measurement_noise)
        self.state[3] = wrap_angle(self.state[3])
        self.last_compass_frame_id = int(frame_id)
        return True

    def update_odometry_speed(self, speed_mps, frame_id, std_mps=None):
        """Teker/araç odometrisi varsa hız ölçümünü filtreye ekler."""
        if not self.initialized:
            raise RuntimeError("EKF başlatılmadan odometri güncellemesi yapılamaz.")
        if self.last_odometry_frame_id == int(frame_id):
            return False

        standard_deviation = (
            self.odometry_speed_std_mps
            if std_mps is None
            else max(0.05, float(std_mps))
        )
        observation = np.zeros((1, 6), dtype=np.float64)
        observation[0, 2] = 1.0
        residual = np.array(
            [max(0.0, float(speed_mps)) - float(self.state[2])],
            dtype=np.float64,
        )
        measurement_noise = np.array([[standard_deviation**2]], dtype=np.float64)
        self._linear_update(residual, observation, measurement_noise)
        self.state[2] = max(0.0, float(self.state[2]))
        self.last_odometry_frame_id = int(frame_id)
        return True

    def result(self):
        """Filtre sonucunu sade bir sözlük olarak döndürür."""
        if not self.initialized:
            return {"available": False, "status": "UNINITIALIZED"}
        position_std = math.sqrt(
            max(0.0, float(self.covariance[0, 0] + self.covariance[1, 1]))
        )
        speed_std = math.sqrt(max(0.0, float(self.covariance[2, 2])))
        yaw_std = math.sqrt(max(0.0, float(self.covariance[3, 3])))
        return {
            "available": True,
            "x_m": float(self.state[0]),
            "y_m": float(self.state[1]),
            "speed_mps": max(0.0, float(self.state[2])),
            "yaw_rad": wrap_angle(self.state[3]),
            "yaw_deg": math.degrees(wrap_angle(self.state[3])),
            "position_std_m": position_std,
            "speed_std_mps": speed_std,
            "yaw_std_rad": yaw_std,
            "covariance": self.covariance.copy(),
            "rejected_gnss_updates": int(self.rejected_gnss_updates),
        }

    def inflate_for_missing_measurement(self, seconds):
        """Sensör yaşlandığında belirsizliği açıkça büyütür."""
        seconds = max(0.0, float(seconds))
        self.covariance[0, 0] += (0.80 * seconds) ** 2
        self.covariance[1, 1] += (0.80 * seconds) ** 2
        self.covariance[2, 2] += (0.50 * seconds) ** 2
        self.covariance[3, 3] += (0.08 * seconds) ** 2
        self._symmetrize_covariance()

    def _linear_update(self, residual, observation, measurement_noise):
        innovation = observation @ self.covariance @ observation.T + measurement_noise
        gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
        self.state = self.state + gain @ residual

        identity = np.eye(len(self.state), dtype=np.float64)
        correction = identity - gain @ observation
        # Joseph biçimi sayısal olarak pozitif yarı-belirliliği daha iyi korur.
        self.covariance = (
            correction @ self.covariance @ correction.T
            + gain @ measurement_noise @ gain.T
        )
        self._symmetrize_covariance()

    def _symmetrize_covariance(self):
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
        diagonal = np.diag(self.covariance).copy()
        diagonal = np.maximum(diagonal, 1e-9)
        np.fill_diagonal(self.covariance, diagonal)

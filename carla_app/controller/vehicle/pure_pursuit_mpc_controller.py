"""Pure Pursuit başlangıç çözümünü MPC ile iyileştiren yanal kontrolcü."""

import math
import time

import numpy as np

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.pure_pursuit_controller import (
    PurePursuitController,
)

try:
    from scipy.optimize import LinearConstraint, minimize
except ImportError:
    LinearConstraint = None
    minimize = None


def clamp(value, minimum, maximum):
    """Bir sayıyı alt ve üst sınırlar içinde tutar."""
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    """Açıyı -pi ile +pi aralığına getirir."""
    return math.atan2(math.sin(angle), math.cos(angle))


class PurePursuitMPCController:
    """Pure Pursuit ile başlar, direksiyon dizisini MPC ile iyileştirir."""

    def __init__(self, dt=0.05, parameters=None):
        self.dt = max(0.01, float(dt))
        self.parameters = parameters or DrivingParameters(dt)
        self.pure_pursuit = PurePursuitController(dt)
        self.wheelbase_m = self.pure_pursuit.wheelbase_m
        self.maximum_wheel_angle_rad = (
            self.pure_pursuit.maximum_wheel_angle_rad
        )
        self.previous_steer = 0.0
        self.last_solution = None
        self.last_info = self.empty_info()

    def run_step(self, state):
        """Pure Pursuit warm-start ile bir MPC çevrimi çalıştırır."""
        self.pure_pursuit.previous_steer = self.previous_steer
        pure_pursuit_steer = self.pure_pursuit.run_step(state)
        pure_pursuit_info = dict(self.pure_pursuit.last_info)
        path = state.get("reference_path", [])
        speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))

        if len(path) < 5:
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "short_path",
            )
        if speed_mps < self.parameters.mpc_minimum_speed_mps:
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "low_speed",
            )
        if minimize is None or LinearConstraint is None:
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "solver_unavailable",
            )

        started_at = time.perf_counter()
        try:
            result = self.solve_mpc(
                state,
                pure_pursuit_steer,
            )
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "solver_error",
                elapsed_ms,
                {"solver_error": f"{type(error).__name__}: {error}"},
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if not result["solver_success"]:
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "solver_status",
                elapsed_ms,
                result,
            )
        if elapsed_ms > self.parameters.mpc_time_budget_ms:
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "time_budget",
                elapsed_ms,
                result,
            )
        if result["predicted_max_error_m"] > (
            self.parameters.mpc_maximum_predicted_error_m
        ):
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "prediction_error",
                elapsed_ms,
                result,
            )

        steer = self.limit_steering_change(result["steer"], speed_mps)
        if not math.isfinite(steer):
            return self.use_pure_pursuit(
                pure_pursuit_steer,
                pure_pursuit_info,
                "non_finite_command",
                elapsed_ms,
                result,
            )

        self.previous_steer = steer
        self.pure_pursuit.previous_steer = steer
        self.last_info = pure_pursuit_info
        self.last_info.update(result)
        self.last_info.update(
            {
                "controller": "pure_pursuit_mpc",
                "mpc_active": True,
                "fallback_reason": None,
                "mpc_solve_ms": float(elapsed_ms),
                "steer": float(steer),
            }
        )
        return steer

    def solve_mpc(self, state, pure_pursuit_steer):
        """Hata modelini kurar ve direksiyon dizisini optimize eder."""
        speed_mps = max(
            self.parameters.mpc_minimum_speed_mps,
            float(state.get("speed_mps", 0.0)),
        )
        horizon = self.parameters.mpc_horizon_steps
        model_dt = self.parameters.mpc_step_s
        path_error = self.current_path_error(state)
        initial_state = np.array(
            [
                path_error["cross_track_error_m"],
                path_error["heading_error_rad"],
            ],
            dtype=np.float64,
        )
        curvatures = self.preview_curvatures(
            state.get("reference_path", []),
            path_error["target_index"],
            speed_mps,
            horizon,
            model_dt,
        )
        base, control_matrix = self.build_prediction_model(
            initial_state,
            curvatures,
            speed_mps,
            model_dt,
        )
        objective_matrix, objective_vector = self.build_objective(
            base,
            control_matrix,
            horizon,
        )
        difference, lower_rate, upper_rate, wheel_limit = (
            self.build_constraints(speed_mps, horizon, model_dt)
        )
        warm_start = self.build_pure_pursuit_warm_start(
            pure_pursuit_steer,
            speed_mps,
            horizon,
            model_dt,
        )

        def objective(steering_sequence):
            return float(
                0.5 * steering_sequence @ objective_matrix @ steering_sequence
                + objective_vector @ steering_sequence
            )

        def gradient(steering_sequence):
            return objective_matrix @ steering_sequence + objective_vector

        result = minimize(
            objective,
            warm_start,
            jac=gradient,
            method="SLSQP",
            bounds=[(-wheel_limit, wheel_limit)] * horizon,
            constraints=[
                LinearConstraint(difference, lower_rate, upper_rate),
            ],
            options={
                "maxiter": self.parameters.mpc_maximum_iterations,
                "ftol": self.parameters.mpc_solver_tolerance,
                "disp": False,
            },
        )
        steering_sequence = np.asarray(result.x, dtype=np.float64)
        if not np.isfinite(steering_sequence).all():
            raise ValueError("MPC sonlu olmayan direksiyon dizisi üretti.")

        predicted_states = base + control_matrix @ steering_sequence
        predicted_errors = predicted_states[0::2]
        self.last_solution = steering_sequence.copy()
        return {
            "solver_success": bool(result.success),
            "solver_status": str(result.message),
            "mpc_iterations": int(result.nit),
            "mpc_objective": float(result.fun),
            "steer": float(
                steering_sequence[0] / self.maximum_wheel_angle_rad
            ),
            "pure_pursuit_seed_steer": float(pure_pursuit_steer),
            "warm_start_first_steer": float(
                warm_start[0] / self.maximum_wheel_angle_rad
            ),
            "predicted_max_error_m": float(
                np.max(np.abs(predicted_errors))
            ),
            "predicted_terminal_error_m": float(predicted_errors[-1]),
            "mpc_horizon_steps": int(horizon),
            "cross_track_error_m": float(initial_state[0]),
            "heading_error_rad": float(initial_state[1]),
            "target_index": int(path_error["target_index"]),
        }

    def current_path_error(self, state):
        """Aracın rota merkezine göre yanal ve yön hatasını verir."""
        location = state["location"]
        projection = self.pure_pursuit.find_nearest_projection(
            float(location.x),
            float(location.y),
            state.get("reference_path", [])[:80],
        )
        yaw = math.radians(float(state.get("yaw", 0.0)))
        return {
            "cross_track_error_m": float(
                projection["cross_track_error_m"]
            ),
            "heading_error_rad": float(
                normalize_angle(projection["path_heading_rad"] - yaw)
            ),
            "target_index": int(projection["segment_index"]),
        }

    def build_prediction_model(
        self,
        initial_state,
        curvatures,
        speed_mps,
        model_dt,
    ):
        """Basit bisiklet hata modelini MPC tahmin matrisine çevirir."""
        horizon = len(curvatures)
        distance_step = speed_mps * model_dt
        state_matrix = np.array(
            [[1.0, -distance_step], [0.0, 1.0]],
            dtype=np.float64,
        )
        control_vector = np.array(
            [0.0, -distance_step / self.wheelbase_m],
            dtype=np.float64,
        )
        base = np.zeros(2 * horizon, dtype=np.float64)
        control_matrix = np.zeros((2 * horizon, horizon), dtype=np.float64)
        current_base = initial_state.copy()
        current_control = np.zeros((2, horizon), dtype=np.float64)

        for step in range(horizon):
            curve_input = np.array(
                [0.0, distance_step * curvatures[step]],
                dtype=np.float64,
            )
            current_base = state_matrix @ current_base + curve_input
            current_control = state_matrix @ current_control
            current_control[:, step] += control_vector
            row = 2 * step
            base[row : row + 2] = current_base
            control_matrix[row : row + 2, :] = current_control
        return base, control_matrix

    def build_objective(self, base, control_matrix, horizon):
        """Rota hatası, direksiyon ve direksiyon değişimi maliyetini kurar."""
        state_weights = np.zeros(2 * horizon, dtype=np.float64)
        for step in range(horizon):
            row = 2 * step
            terminal_scale = 2.0 if step == horizon - 1 else 1.0
            state_weights[row] = (
                self.parameters.mpc_lateral_error_weight * terminal_scale
            )
            state_weights[row + 1] = (
                self.parameters.mpc_heading_error_weight * terminal_scale
            )

        weighted_control = control_matrix * state_weights[:, np.newaxis]
        difference = self.difference_matrix(horizon)
        previous_angle = self.previous_steer * self.maximum_wheel_angle_rad
        difference_target = np.zeros(horizon, dtype=np.float64)
        difference_target[0] = previous_angle
        identity = np.eye(horizon, dtype=np.float64)

        objective_matrix = 2.0 * (
            control_matrix.T @ weighted_control
            + self.parameters.mpc_steering_weight * identity
            + self.parameters.mpc_steering_rate_weight
            * (difference.T @ difference)
        )
        objective_vector = 2.0 * (
            control_matrix.T @ (state_weights * base)
            - self.parameters.mpc_steering_rate_weight
            * difference.T
            @ difference_target
        )
        objective_matrix += 1e-7 * identity
        return objective_matrix, objective_vector

    def build_constraints(self, speed_mps, horizon, model_dt):
        """Direksiyon açısı ve değişim hızının güvenli sınırlarını kurar."""
        normalized_limit = self.pure_pursuit.calculate_steering_limit(
            speed_mps
        )
        wheel_limit = normalized_limit * self.maximum_wheel_angle_rad
        difference = self.difference_matrix(horizon)

        predicted_rate = self.steering_rate_limit(speed_mps)
        predicted_change = (
            predicted_rate * model_dt * self.maximum_wheel_angle_rad
        )
        lower_rate = np.full(horizon, -predicted_change, dtype=np.float64)
        upper_rate = np.full(horizon, predicted_change, dtype=np.float64)

        previous_angle = self.previous_steer * self.maximum_wheel_angle_rad
        current_change = (
            predicted_rate * self.dt * self.maximum_wheel_angle_rad
        )
        lower_rate[0] = previous_angle - current_change
        upper_rate[0] = previous_angle + current_change
        return difference, lower_rate, upper_rate, wheel_limit

    def difference_matrix(self, horizon):
        """Her direksiyonun bir önceki adıma farkını alan matrisi üretir."""
        difference = np.zeros((horizon, horizon), dtype=np.float64)
        difference[0, 0] = 1.0
        for step in range(1, horizon):
            difference[step, step] = 1.0
            difference[step, step - 1] = -1.0
        return difference

    def build_pure_pursuit_warm_start(
        self,
        pure_pursuit_steer,
        speed_mps,
        horizon,
        model_dt,
    ):
        """Pure Pursuit komutundan sınırları sağlayan başlangıç dizisi kurar."""
        target_angle = (
            clamp(float(pure_pursuit_steer), -1.0, 1.0)
            * self.maximum_wheel_angle_rad
        )
        current_angle = self.previous_steer * self.maximum_wheel_angle_rad
        rate = self.steering_rate_limit(speed_mps)
        warm_start = np.zeros(horizon, dtype=np.float64)

        for step in range(horizon):
            step_time = self.dt if step == 0 else model_dt
            maximum_change = rate * step_time * self.maximum_wheel_angle_rad
            change = clamp(
                target_angle - current_angle,
                -maximum_change,
                maximum_change,
            )
            current_angle += change
            warm_start[step] = current_angle
        return warm_start

    def preview_curvatures(
        self,
        path,
        start_index,
        speed_mps,
        horizon,
        model_dt,
    ):
        """MPC adımlarına karşılık gelen gelecekteki rota eğriliklerini bulur."""
        if len(path) < 5:
            return [0.0] * horizon
        index = max(1, min(int(start_index), len(path) - 2))
        travelled_m = 0.0
        curvatures = []

        for step in range(horizon):
            target_distance = speed_mps * model_dt * (step + 1)
            while index + 1 < len(path) - 1 and travelled_m < target_distance:
                first = path[index]
                second = path[index + 1]
                travelled_m += math.hypot(
                    float(second.x) - float(first.x),
                    float(second.y) - float(first.y),
                )
                index += 1
            curvatures.append(self.path_curvature(path, index))
        return curvatures

    def path_curvature(self, path, center_index):
        """Üç rota noktasından işaretli eğrilik hesaplar."""
        index = max(1, min(int(center_index), len(path) - 2))
        first = path[index - 1]
        middle = path[index]
        last = path[index + 1]
        first_x = float(first.x)
        first_y = float(first.y)
        middle_x = float(middle.x)
        middle_y = float(middle.y)
        last_x = float(last.x)
        last_y = float(last.y)

        first_length = math.hypot(middle_x - first_x, middle_y - first_y)
        second_length = math.hypot(last_x - middle_x, last_y - middle_y)
        chord_length = math.hypot(last_x - first_x, last_y - first_y)
        denominator = first_length * second_length * chord_length
        if denominator <= 1e-8:
            return 0.0
        cross = (
            (middle_x - first_x) * (last_y - first_y)
            - (middle_y - first_y) * (last_x - first_x)
        )
        return 2.0 * cross / denominator

    def steering_rate_limit(self, speed_mps):
        """Hız arttıkça direksiyon değişimini daha sakin hale getirir."""
        return clamp(0.90 - 0.020 * speed_mps, 0.45, 0.90)

    def limit_steering_change(self, desired_steer, speed_mps):
        """MPC'nin ilk komutuna son bir fiziksel değişim sınırı uygular."""
        maximum_change = self.steering_rate_limit(speed_mps) * self.dt
        change = clamp(
            float(desired_steer) - self.previous_steer,
            -maximum_change,
            maximum_change,
        )
        return clamp(self.previous_steer + change, -1.0, 1.0)

    def use_pure_pursuit(
        self,
        steer,
        pure_pursuit_info,
        reason,
        elapsed_ms=0.0,
        result=None,
    ):
        """MPC kullanılamadığında Pure Pursuit komutunu güvenle uygular."""
        self.previous_steer = float(steer)
        self.pure_pursuit.previous_steer = self.previous_steer
        self.last_info = pure_pursuit_info
        if result:
            self.last_info.update(result)
        self.last_info.update(
            {
                "controller": "pure_pursuit_fallback",
                "mpc_active": False,
                "fallback_reason": str(reason),
                "mpc_solve_ms": float(elapsed_ms),
                "steer": self.previous_steer,
                "pure_pursuit_seed_steer": float(steer),
            }
        )
        return self.previous_steer

    def empty_info(self):
        """İlk kontrol çevriminden önce kullanılacak tanı bilgilerini verir."""
        info = self.pure_pursuit.empty_info()
        info.update(
            {
                "controller": "pure_pursuit_fallback",
                "mpc_active": False,
                "fallback_reason": "not_started",
                "mpc_solve_ms": 0.0,
                "solver_success": False,
                "solver_status": None,
                "predicted_max_error_m": None,
                "pure_pursuit_seed_steer": 0.0,
            }
        )
        return info

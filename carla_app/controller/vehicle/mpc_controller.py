"""Gelecekteki rota eğriliğini kullanarak direksiyon üreten yanal MPC."""

import math
import time

import numpy as np

from carla_app.config import DrivingParameters
from carla_app.controller.vehicle.stanley_controller import StanleyController

try:
    import osqp
    from scipy import sparse
except ImportError:
    osqp = None
    sparse = None


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class LateralMPCController:
    """Hata durumunda Stanley'ye dönen lineer yanal MPC kontrolcüsü."""

    def __init__(self, dt=0.05, parameters=None):
        self.dt = float(dt)
        self.parameters = parameters or DrivingParameters(dt)
        self.wheelbase_m = 2.87
        self.maximum_wheel_angle_rad = math.radians(35.0)
        self.fallback = StanleyController(dt)
        self.previous_steer = 0.0
        self.last_solution = None
        self.last_info = self.empty_info()

    def run_step(self, state):
        """Bir MPC çevrimi çalıştırır; geçersiz sonuçta Stanley kullanır."""
        self.fallback.previous_steer = self.previous_steer
        fallback_steer = self.fallback.run_step(state)
        fallback_info = dict(self.fallback.last_info)
        path = state.get("reference_path", [])
        speed_mps = max(0.0, float(state.get("speed_mps", 0.0)))

        if len(path) < 5:
            return self.use_fallback(fallback_steer, fallback_info, "short_path")
        if speed_mps < self.parameters.mpc_minimum_speed_mps:
            return self.use_fallback(fallback_steer, fallback_info, "low_speed")
        if osqp is None or sparse is None:
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "solver_unavailable",
            )

        started_at = time.perf_counter()
        try:
            result = self.solve_mpc(state)
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "solver_error",
                elapsed_ms,
                {
                    "solver_error": f"{type(error).__name__}: {error}",
                },
            )

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if elapsed_ms > self.parameters.mpc_time_budget_ms:
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "time_budget",
                elapsed_ms,
                result,
            )
        if result["solver_status"] != "solved":
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "solver_status",
                elapsed_ms,
                result,
            )
        if result["predicted_max_error_m"] > (
            self.parameters.mpc_maximum_predicted_error_m
        ):
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "prediction_error",
                elapsed_ms,
                result,
            )

        steer = self.limit_steering_change(result["steer"], speed_mps)
        if not math.isfinite(steer):
            return self.use_fallback(
                fallback_steer,
                fallback_info,
                "non_finite_command",
                elapsed_ms,
                result,
            )

        self.previous_steer = steer
        self.fallback.previous_steer = steer
        self.last_info = fallback_info
        self.last_info.update(result)
        self.last_info.update(
            {
                "controller": "mpc",
                "mpc_active": True,
                "fallback_reason": None,
                "mpc_solve_ms": float(elapsed_ms),
                "steer": float(steer),
                "cross_track_error_m": result["mpc_cross_track_error_m"],
                "heading_error_rad": result["mpc_heading_error_rad"],
                "target_index": result["mpc_target_index"],
            }
        )
        return steer

    def solve_mpc(self, state):
        """Hata dinamiğini kurup OSQP ile direksiyon dizisini çözer."""
        speed_mps = max(
            self.parameters.mpc_minimum_speed_mps,
            float(state.get("speed_mps", 0.0)),
        )
        horizon = self.parameters.mpc_horizon_steps
        model_dt = self.parameters.mpc_step_s
        path_error = self.current_path_error(state)
        initial_state = np.array(
            [path_error["cross_track_error_m"], path_error["heading_error_rad"]],
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
        objective_matrix, objective_vector, difference = self.build_objective(
            base,
            control_matrix,
            horizon,
        )
        constraint_matrix, lower, upper = self.build_constraints(
            difference,
            speed_mps,
            horizon,
            model_dt,
        )

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(np.triu(objective_matrix)),
            q=objective_vector,
            A=sparse.csc_matrix(constraint_matrix),
            l=lower,
            u=upper,
            verbose=False,
            eps_abs=self.parameters.mpc_solver_tolerance,
            eps_rel=self.parameters.mpc_solver_tolerance,
            max_iter=self.parameters.mpc_maximum_iterations,
            polishing=False,
            warm_starting=True,
        )
        warm_start = self.shifted_warm_start(horizon)
        if warm_start is not None:
            solver.warm_start(x=warm_start)
        solution = solver.solve(raise_error=False)
        status = str(solution.info.status).strip().lower()
        if not status.startswith("solved") or solution.x is None:
            return {
                "solver_status": status,
                "steer": self.previous_steer,
                "mpc_iterations": int(solution.info.iter),
                "mpc_objective": None,
                "predicted_max_error_m": math.inf,
            }

        steering_sequence = np.asarray(solution.x, dtype=np.float64)
        if not np.isfinite(steering_sequence).all():
            raise ValueError("MPC sonlu olmayan direksiyon dizisi uretti.")
        self.last_solution = steering_sequence.copy()
        predicted_states = base + control_matrix @ steering_sequence
        predicted_lateral_errors = predicted_states[0::2]
        normalized_steer = steering_sequence[0] / self.maximum_wheel_angle_rad
        return {
            "solver_status": "solved",
            "steer": float(normalized_steer),
            "mpc_iterations": int(solution.info.iter),
            "mpc_objective": float(solution.info.obj_val),
            "predicted_max_error_m": float(
                np.max(np.abs(predicted_lateral_errors))
            ),
            "predicted_terminal_error_m": float(predicted_lateral_errors[-1]),
            "mpc_horizon_steps": int(horizon),
            "mpc_cross_track_error_m": float(initial_state[0]),
            "mpc_heading_error_rad": float(initial_state[1]),
            "mpc_target_index": int(path_error["target_index"]),
        }

    def current_path_error(self, state):
        """Araç merkezinin rotaya göre yanal ve başlık hatasını verir."""
        path = state.get("reference_path", [])
        location = state["location"]
        projection = self.fallback.find_nearest_path_projection(
            float(location.x),
            float(location.y),
            path[:80],
        )
        yaw = math.radians(float(state.get("yaw", 0.0)))
        heading_error = normalize_angle(projection["path_heading_rad"] - yaw)
        return {
            "cross_track_error_m": float(projection["cross_track_error_m"]),
            "heading_error_rad": float(heading_error),
            "target_index": int(projection["segment_index"]),
        }

    def build_prediction_model(
        self,
        initial_state,
        curvatures,
        speed_mps,
        model_dt,
    ):
        """Frenet hata modelini yığılmış tahmin matrisine dönüştürür."""
        horizon = len(curvatures)
        speed_step = speed_mps * model_dt
        state_matrix = np.array(
            [[1.0, -speed_step], [0.0, 1.0]],
            dtype=np.float64,
        )
        control_vector = np.array(
            [0.0, -speed_step / self.wheelbase_m],
            dtype=np.float64,
        )
        base = np.zeros(2 * horizon, dtype=np.float64)
        control_matrix = np.zeros((2 * horizon, horizon), dtype=np.float64)
        current_base = initial_state.copy()
        current_control = np.zeros((2, horizon), dtype=np.float64)

        for step in range(horizon):
            curvature_input = np.array(
                [0.0, speed_step * curvatures[step]],
                dtype=np.float64,
            )
            current_base = state_matrix @ current_base + curvature_input
            current_control = state_matrix @ current_control
            current_control[:, step] += control_vector
            row = 2 * step
            base[row : row + 2] = current_base
            control_matrix[row : row + 2, :] = current_control
        return base, control_matrix

    def build_objective(self, base, control_matrix, horizon):
        """Yanal hata, başlık hatası ve direksiyon değişimi maliyetini kurar."""
        state_weights = np.zeros(2 * horizon, dtype=np.float64)
        for step in range(horizon):
            row = 2 * step
            terminal_scale = 2.5 if step == horizon - 1 else 1.0
            state_weights[row] = (
                self.parameters.mpc_lateral_error_weight * terminal_scale
            )
            state_weights[row + 1] = (
                self.parameters.mpc_heading_error_weight * terminal_scale
            )
        weighted_control = control_matrix * state_weights[:, np.newaxis]
        difference = np.zeros((horizon, horizon), dtype=np.float64)
        difference[0, 0] = 1.0
        for step in range(1, horizon):
            difference[step, step] = 1.0
            difference[step, step - 1] = -1.0

        previous_wheel_angle = self.previous_steer * self.maximum_wheel_angle_rad
        difference_target = np.zeros(horizon, dtype=np.float64)
        difference_target[0] = previous_wheel_angle
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
        return objective_matrix, objective_vector, difference

    def build_constraints(self, difference, speed_mps, horizon, model_dt):
        """Direksiyon açısı ve değişim hızını bütün ufukta sınırlar."""
        normalized_limit = self.fallback.calculate_steering_limit(speed_mps)
        wheel_limit = normalized_limit * self.maximum_wheel_angle_rad
        identity = np.eye(horizon, dtype=np.float64)
        constraint_matrix = np.vstack((identity, difference))
        lower_angle = np.full(horizon, -wheel_limit, dtype=np.float64)
        upper_angle = np.full(horizon, wheel_limit, dtype=np.float64)

        normalized_rate = clamp(0.8 - 0.035 * speed_mps, 0.4, 0.8)
        predicted_change = (
            normalized_rate * model_dt * self.maximum_wheel_angle_rad
        )
        lower_rate = np.full(horizon, -predicted_change, dtype=np.float64)
        upper_rate = np.full(horizon, predicted_change, dtype=np.float64)
        current_change = normalized_rate * self.dt * self.maximum_wheel_angle_rad
        previous_wheel_angle = self.previous_steer * self.maximum_wheel_angle_rad
        lower_rate[0] = previous_wheel_angle - current_change
        upper_rate[0] = previous_wheel_angle + current_change
        lower = np.concatenate((lower_angle, lower_rate))
        upper = np.concatenate((upper_angle, upper_rate))
        return constraint_matrix, lower, upper

    def preview_curvatures(
        self,
        path,
        start_index,
        speed_mps,
        horizon,
        model_dt,
    ):
        """MPC zaman adımlarını rota üzerindeki gelecek eğriliklerle eşler."""
        if len(path) < 5:
            return [0.0] * horizon
        index = max(0, min(int(start_index), len(path) - 2))
        travelled_m = 0.0
        curvatures = []
        for step in range(horizon):
            target_distance = speed_mps * model_dt * (step + 1)
            while index + 1 < len(path) and travelled_m < target_distance:
                first = path[index]
                second = path[index + 1]
                travelled_m += math.hypot(
                    second.x - first.x,
                    second.y - first.y,
                )
                index += 1
            curvature = self.fallback.calculate_path_curvature(path, index)
            curvatures.append(float(curvature))
        return curvatures

    def shifted_warm_start(self, horizon):
        if self.last_solution is None or len(self.last_solution) != horizon:
            return None
        shifted = np.empty(horizon, dtype=np.float64)
        shifted[:-1] = self.last_solution[1:]
        shifted[-1] = self.last_solution[-1]
        return shifted

    def limit_steering_change(self, desired_steer, speed_mps):
        normalized_rate = clamp(0.8 - 0.035 * speed_mps, 0.4, 0.8)
        maximum_change = normalized_rate * self.dt
        change = clamp(
            float(desired_steer) - self.previous_steer,
            -maximum_change,
            maximum_change,
        )
        return clamp(self.previous_steer + change, -1.0, 1.0)

    def use_fallback(
        self,
        fallback_steer,
        fallback_info,
        reason,
        elapsed_ms=0.0,
        result=None,
    ):
        self.previous_steer = float(fallback_steer)
        self.fallback.previous_steer = self.previous_steer
        self.last_info = fallback_info
        self.last_info.update(
            {
                "controller": "stanley_fallback",
                "mpc_active": False,
                "fallback_reason": str(reason),
                "mpc_solve_ms": float(elapsed_ms),
                "steer": self.previous_steer,
                "solver_status": None,
                "predicted_max_error_m": None,
            }
        )
        if result:
            self.last_info.update(result)
            self.last_info["controller"] = "stanley_fallback"
            self.last_info["mpc_active"] = False
            self.last_info["fallback_reason"] = str(reason)
            self.last_info["mpc_solve_ms"] = float(elapsed_ms)
            self.last_info["steer"] = self.previous_steer
        return self.previous_steer

    def empty_info(self):
        info = self.fallback.empty_info()
        info.update(
            {
                "controller": "stanley_fallback",
                "mpc_active": False,
                "fallback_reason": "not_started",
                "mpc_solve_ms": 0.0,
                "solver_status": None,
                "predicted_max_error_m": None,
                "steer": 0.0,
            }
        )
        return info

#!/usr/bin/env python3
"""Offline longitudinal-controller benchmark; CARLA is not required.

The simple plant is intentionally approximate. It is used for regression and
parameter screening. Final validation must still be performed in CARLA because
vehicle mass, drag, tire model and slope are simulator-dependent.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

from carla_app.controller.vehicle.longitudinal_controller import LongitudinalController
from carla_app.controller.vehicle.reference_speed import RandomReferenceSpeed
from carla_app.controller.vehicle.speed_planner import CurvatureSpeedPlanner


def percentile(values, fraction):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    index = int(round(max(0.0, min(1.0, fraction)) * (len(values) - 1)))
    return values[index]


def simulate(duration_s=120.0, dt=0.05, seed=7):
    controller = LongitudinalController(dt=dt, follow_gap_m=10.0, follow_gap_margin_m=1.5)
    planner = CurvatureSpeedPlanner(dt=dt, cruise_speed_kmh=60.0)
    reference = RandomReferenceSpeed(
        dt=dt,
        minimum_speed_kmh=10.0,
        maximum_speed_kmh=55.0,
        minimum_hold_seconds=7.0,
        maximum_hold_seconds=13.0,
        seed=seed,
        enabled=True,
        initial_speed_kmh=30.0,
    )

    speed = 0.0
    actual_acceleration = 0.0
    previous_acceleration = 0.0
    rows = []
    steps = int(round(duration_s / dt))

    for index in range(steps):
        time_s = index * dt
        requested_speed, reference_info = reference.update(60.0 / 3.6)
        state = {
            "speed_mps": speed,
            "lane_width": 3.5,
            "vehicle_half_width_m": 0.95,
            "reference_path": [],
        }
        target_speed, speed_info = planner.run_step(
            state,
            lateral_info={"cross_track_error_m": 0.0, "heading_error_rad": 0.0},
            requested_speed_mps=requested_speed,
        )
        throttle, brake, info = controller.run_step(state, None, target_speed)

        # Approximate Tesla Model 3 longitudinal response in CARLA. The first-
        # order actuator prevents this regression plant from unrealistically
        # following pedal changes instantaneously.
        rolling_drag = 0.16 + 0.012 * speed if speed > 0.05 else 0.0
        desired_plant_acceleration = 3.2 * throttle - 4.8 * brake - rolling_drag
        actuator_tau_s = 0.30
        actual_acceleration += (desired_plant_acceleration - actual_acceleration) * dt / actuator_tau_s
        speed = max(0.0, speed + actual_acceleration * dt)
        jerk = (actual_acceleration - previous_acceleration) / dt
        previous_acceleration = actual_acceleration

        rows.append(
            {
                "time_s": time_s,
                "reference_kmh": requested_speed * 3.6,
                "target_kmh": target_speed * 3.6,
                "speed_kmh": speed * 3.6,
                "target_error_kmh": (target_speed - speed) * 3.6,
                "throttle": throttle,
                "brake": brake,
                "controller_acceleration_mps2": info["acceleration_mps2"],
                "plant_acceleration_mps2": actual_acceleration,
                "plant_jerk_mps3": jerk,
                "segment_index": reference_info["segment_index"],
                "speed_reason": speed_info["speed_reason"],
            }
        )
    return rows


def calculate_metrics(rows):
    warm = [row for row in rows if row["time_s"] >= 5.0]
    errors_mps = [abs(row["target_error_kmh"]) / 3.6 for row in warm]
    signed_errors_mps = [row["target_error_kmh"] / 3.6 for row in warm]
    accelerations = [row["plant_acceleration_mps2"] for row in warm]
    jerks = [abs(row["plant_jerk_mps3"]) for row in warm]
    pedal_switches = 0
    previous = None
    for row in warm:
        active = "brake" if row["brake"] > 0.01 else "throttle" if row["throttle"] > 0.01 else "coast"
        if previous is not None and active != previous:
            pedal_switches += 1
        previous = active

    return {
        "target_mae_mps": statistics.fmean(errors_mps),
        "target_rmse_mps": math.sqrt(statistics.fmean(error * error for error in signed_errors_mps)),
        "target_p95_abs_error_mps": percentile(errors_mps, 0.95),
        "maximum_acceleration_mps2": max(accelerations),
        "maximum_deceleration_mps2": min(accelerations),
        "p95_absolute_jerk_mps3": percentile(jerks, 0.95),
        "maximum_absolute_jerk_mps3": max(jerks),
        "pedal_mode_switches": pedal_switches,
    }


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    times = [row["time_s"] for row in rows]
    figure = plt.figure(figsize=(12, 7))
    axis_speed = figure.add_axes((0.08, 0.55, 0.88, 0.38))
    axis_speed.plot(times, [row["reference_kmh"] for row in rows], label="reference")
    axis_speed.plot(times, [row["target_kmh"] for row in rows], label="ramped target")
    axis_speed.plot(times, [row["speed_kmh"] for row in rows], label="actual")
    axis_speed.set_ylabel("Speed [km/h]")
    axis_speed.grid(True)
    axis_speed.legend()

    axis_control = figure.add_axes((0.08, 0.10, 0.88, 0.34))
    axis_control.plot(times, [row["throttle"] for row in rows], label="throttle")
    axis_control.plot(times, [row["brake"] for row in rows], label="brake")
    axis_control.plot(times, [row["plant_acceleration_mps2"] for row in rows], label="acceleration")
    axis_control.set_xlabel("Time [s]")
    axis_control.set_ylabel("Control / acceleration")
    axis_control.grid(True)
    axis_control.legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("data/controller_benchmark"))
    args = parser.parse_args()

    rows = simulate(args.duration, args.dt, args.seed)
    metrics = calculate_metrics(rows)
    csv_path = args.output.with_suffix(".csv")
    png_path = args.output.with_suffix(".png")
    write_csv(rows, csv_path)
    plot_written = write_plot(rows, png_path)

    print("[BENCHMARK]")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}={value:.4f}")
        else:
            print(f"{name}={value}")
    print(f"csv={csv_path}")
    if plot_written:
        print(f"plot={png_path}")


if __name__ == "__main__":
    main()

# CARLA Smooth Controller — Complete Integration Patch

This package is designed for the `controller-improvements` branch of:

`TOFAS-2026-AR-GE/Carla-Workflow`

It keeps the existing persistent route, Stanley lane controller, camera/radar lead tracking, and independent AEB. It adds the missing experiment and behavior layers.

## Implemented behavior

### 1. Smooth random reference-speed experiment

`RandomReferenceSpeed` generates deterministic random speed steps. The curvature speed planner converts each discontinuous step into a comfortable ramp before it reaches the pedal loop.

Default experiment:

- reference range: 15–55 km/h
- hold time: 6–12 s
- deterministic seed: 7
- target acceleration ramp: 1.25 m/s²
- target deceleration ramp: 2.0 m/s²

Disable random mode with:

```env
ENABLE_RANDOM_REFERENCE_SPEED=false
```

The normal reference then becomes `MAXIMUM_SPEED_KMH`.

### 2. Longitudinal controller

The controller has four layers:

1. PI speed tracking with anti-windup.
2. IDM-style interaction with the vehicle ahead.
3. Asymmetric jerk limiting.
4. Pedal mapping with throttle/brake mutual exclusion.

The base following gap is 10 m. A 1.5 m comfort margin prevents unnecessary pedal switching. A small speed-dependent term and closing-speed term are added when necessary, so 10 m is treated as the minimum comfort base rather than a dangerous fixed high-speed distance.

### 3. Turn-speed adjustment

The speed planner previews 30–75 m of the persistent route, estimates curvature with a robust percentile, and uses:

```text
v_curve = sqrt(a_lateral_max / curvature)
```

The default passenger-comfort lateral acceleration limit is 1.8 m/s². Lane-recovery speed is continuous rather than a set of abrupt 8/12/18 km/h steps.

### 4. Traffic-light obedience

The existing YOLO model is run once per perception frame. Its output is split into:

- vehicles
- traffic-light states

Expected traffic-light class names include:

- `traffic_light_red`
- `traffic_light_orange` or `traffic_light_yellow`
- `traffic_light_green`

The colour comes from the trained camera model. CARLA traffic-light actor geometry supplies stop-line distance because a 2-D bounding box is not a reliable metric range measurement.

Behavior:

- red: stop and hold
- yellow: stop only when comfortable stopping is still possible; otherwise clear the dilemma zone
- green: release hold and restart smoothly

For debugging only, CARLA state fallback can be enabled:

```env
TRAFFIC_LIGHT_GROUND_TRUTH_FALLBACK=true
```

Keep it `false` when evaluating the camera model.

### 5. Lead-vehicle behavior

The existing camera/radar tracker remains the source of the physical lead vehicle. The longitudinal controller uses:

```env
FOLLOW_GAP_M=10.0
FOLLOW_GAP_MARGIN_M=1.5
```

A red/yellow stop line is treated as a separate stationary virtual obstacle. The closest valid longitudinal constraint wins. Traffic lights never enter the AEB supervisor; AEB remains reserved for physical collision threats.

### 6. Visualization

The OpenCV dashboard displays:

- front camera detections
- current control mode
- random requested speed
- ramped controller target speed
- current speed
- throttle and brake
- acceleration command
- lead distance and desired distance
- traffic-light state and distance
- top-down route preview
- reference/target/current speed history

## Files

### New

- `carla_app/controller/vehicle/reference_speed.py`
- `carla_app/controller/vehicle/traffic_light.py`
- `scripts/controller_benchmark.py`
- `tests/test_controller_regression.py`
- `tests/test_reference_speed.py`
- `tests/test_speed_planner.py`
- `tests/test_longitudinal_controller.py`
- `tests/test_traffic_light.py`

### Replaced

- `carla_app/application.py`
- `carla_app/config.py`
- `carla_app/controller/vehicle/vehicle_controller.py`
- `carla_app/controller/vehicle/longitudinal_controller.py`
- `carla_app/controller/vehicle/speed_planner.py`
- `carla_app/perception/system.py`
- `carla_app/perception/vehicle_detector.py`
- `carla_app/visualization/viewer.py`
- `.env.example`

## Install into the repository

From this package directory:

```bash
./install_into_repo.sh /path/to/Carla-Workflow
```

Or manually copy the files over a clean checkout of `controller-improvements`.

Then update your real `.env` from `.env.example`. Do not overwrite model paths that already work on your machine.

## Run tests

```bash
cd /path/to/Carla-Workflow
pytest -q
```

The supplied pure-Python regression suite contains 16 tests and does not require CARLA to be running.

## Run the offline benchmark

```bash
cd /path/to/Carla-Workflow
python scripts/controller_benchmark.py --duration 120
```

The benchmark writes a CSV and plot. The bundled 120 s regression result produced:

| Metric | Result |
|---|---:|
| Target-speed MAE | 0.6920 m/s |
| Target-speed RMSE | 0.9275 m/s |
| 95th-percentile absolute error | 1.8962 m/s |
| Maximum acceleration | 1.3639 m/s² |
| Maximum deceleration | -2.1540 m/s² |
| 95th-percentile absolute jerk | 1.1786 m/s³ |
| Maximum absolute jerk | 2.0088 m/s³ |

This plant is a regression model, not a replacement for CARLA validation. Use the CARLA dashboard and recorded CSV data for final vehicle-specific tuning.

## Run CARLA

Use the existing project command:

```bash
unset CUDA_VISIBLE_DEVICES
python -u main.py
```

Or your existing `run.sh`.

Important log fields:

```text
speed=      current speed
reference=  raw experiment reference
target=     curvature/recovery-limited ramped target
mode=       CRUISE, FOLLOW, STOP_RED, HOLD_RED, EMERGENCY, ...
ctrl_gap=   filtered physical/virtual obstacle range
desired_gap= controller spacing target
tl=         traffic-light state and distance
```

## CARLA tuning order

1. Run with no traffic and random reference enabled.
2. Check that current speed follows the orange target curve without oscillation.
3. Run the same seed several times for repeatability.
4. Add one constant-speed lead vehicle and inspect gap convergence.
5. Add stop-and-go traffic.
6. Test red, yellow, and green transitions.
7. Test curves at several maximum speeds.
8. Change one parameter at a time and save the before/after CSV.

Do not tune AEB thresholds to improve comfort. AEB is the final safety layer; comfort should be tuned in the planner and longitudinal controller.

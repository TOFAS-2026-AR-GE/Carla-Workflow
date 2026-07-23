"""CARLA şerit ground-truth'unu yalnız testlerde kullanan doğrulama araçları."""

import itertools
import math

import numpy as np


def camera_matrix(image_width, image_height, horizontal_fov_degrees):
    """CARLA yatay FOV değerinden OpenCV iç kalibrasyon matrisi üretir."""
    focal_length = float(image_width) / (
        2.0 * math.tan(math.radians(float(horizontal_fov_degrees)) / 2.0)
    )
    return np.asarray(
        [
            [focal_length, 0.0, float(image_width) / 2.0],
            [0.0, focal_length, float(image_height) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_matrix(transform):
    """CARLA Transform benzeri nesneyi yerel-koordinattan-dünyaya matris yapar."""
    get_matrix = getattr(transform, "get_matrix", None)
    if callable(get_matrix):
        return np.asarray(get_matrix(), dtype=np.float64)

    rotation = transform.rotation
    location = transform.location
    roll = math.radians(float(rotation.roll))
    pitch = math.radians(float(rotation.pitch))
    yaw = math.radians(float(rotation.yaw))
    cosine_roll, sine_roll = math.cos(roll), math.sin(roll)
    cosine_pitch, sine_pitch = math.cos(pitch), math.sin(pitch)
    cosine_yaw, sine_yaw = math.cos(yaw), math.sin(yaw)

    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine_roll, -sine_roll],
            [0.0, sine_roll, cosine_roll],
        ]
    )
    rotation_y = np.asarray(
        [
            [cosine_pitch, 0.0, sine_pitch],
            [0.0, 1.0, 0.0],
            [-sine_pitch, 0.0, cosine_pitch],
        ]
    )
    rotation_z = np.asarray(
        [
            [cosine_yaw, -sine_yaw, 0.0],
            [sine_yaw, cosine_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
    matrix[:3, 3] = (
        float(location.x),
        float(location.y),
        float(location.z),
    )
    return matrix


def project_world_points(
    world_points,
    camera_transform,
    image_width,
    image_height,
    horizontal_fov_degrees,
):
    """Dünya noktalarını CARLA kamera eksenlerinden görüntü piksellerine taşır."""
    points = np.asarray(world_points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    points = points.reshape(-1, 3)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    world_to_camera = np.linalg.inv(transform_matrix(camera_transform))
    camera_points = (world_to_camera @ homogeneous.T).T[:, :3]

    # CARLA kamera: x ileri, y sağ, z yukarı.
    # OpenCV kamera: x sağ, y aşağı, z ileri.
    opencv_points = np.column_stack(
        (
            camera_points[:, 1],
            -camera_points[:, 2],
            camera_points[:, 0],
        )
    )
    in_front = opencv_points[:, 2] > 0.10
    opencv_points = opencv_points[in_front]
    if len(opencv_points) == 0:
        return np.empty((0, 2), dtype=np.float64)

    intrinsic = camera_matrix(
        image_width,
        image_height,
        horizontal_fov_degrees,
    )
    pixels = (intrinsic @ opencv_points.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    visible = (
        (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(image_width))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(image_height))
    )
    return pixels[visible]


def sample_ego_lane_boundaries(
    carla_map,
    ego_location,
    forward_distance_m=70.0,
    spacing_m=0.75,
):
    """Ego şeridinin işaretli sol/sağ sınırlarını CARLA haritasından örnekler."""
    try:
        import carla

        lane_type = carla.LaneType.Driving
    except ImportError:
        lane_type = None

    arguments = {"project_to_road": True}
    if lane_type is not None:
        arguments["lane_type"] = lane_type
    waypoint = carla_map.get_waypoint(ego_location, **arguments)
    if waypoint is None:
        return {"left": [], "right": []}

    left_points = []
    right_points = []
    travelled = 0.0
    while waypoint is not None and travelled <= float(forward_distance_m):
        if not waypoint.is_junction:
            center = waypoint.transform.location
            right_vector = waypoint.transform.get_right_vector()
            half_width = 0.5 * float(waypoint.lane_width)
            if _marking_is_visible(waypoint.left_lane_marking):
                left_points.append(
                    [
                        float(center.x) - float(right_vector.x) * half_width,
                        float(center.y) - float(right_vector.y) * half_width,
                        float(center.z) - float(right_vector.z) * half_width,
                    ]
                )
            if _marking_is_visible(waypoint.right_lane_marking):
                right_points.append(
                    [
                        float(center.x) + float(right_vector.x) * half_width,
                        float(center.y) + float(right_vector.y) * half_width,
                        float(center.z) + float(right_vector.z) * half_width,
                    ]
                )

        candidates = list(waypoint.next(float(spacing_m)))
        if not candidates:
            break
        waypoint = min(
            candidates,
            key=lambda candidate: _continuity_cost(waypoint, candidate),
        )
        travelled += float(spacing_m)

    return {"left": left_points, "right": right_points}


def evaluate_lane_geometry(
    predicted_lanes,
    ground_truth_lanes,
    image_width,
    tolerance_px=None,
):
    """Tahminleri GT eğrileriyle eşleyip piksel hata, kapsam ve F1 hesaplar."""
    predictions = [
        np.asarray(lane["points"], dtype=np.float64)
        for lane in predicted_lanes
        if lane.get("detected") and len(lane.get("points", [])) >= 2
    ]
    ground_truth = [
        np.asarray(points, dtype=np.float64)
        for points in ground_truth_lanes
        if len(points) >= 2
    ]
    tolerance = (
        max(8.0, 0.025 * float(image_width))
        if tolerance_px is None
        else float(tolerance_px)
    )

    candidates = []
    for prediction_index, truth_index in itertools.product(
        range(len(predictions)),
        range(len(ground_truth)),
    ):
        comparison = _curve_error(
            predictions[prediction_index],
            ground_truth[truth_index],
        )
        if comparison is None:
            continue
        score = comparison["mean_error_px"] + (
            1.0 - comparison["coverage"]
        ) * 2.0 * tolerance
        candidates.append(
            (score, prediction_index, truth_index, comparison)
        )

    matched_predictions = set()
    matched_truth = set()
    accepted = []
    for _score, prediction_index, truth_index, comparison in sorted(candidates):
        if prediction_index in matched_predictions or truth_index in matched_truth:
            continue
        if (
            comparison["mean_error_px"] <= tolerance
            and comparison["coverage"] >= 0.50
        ):
            matched_predictions.add(prediction_index)
            matched_truth.add(truth_index)
            accepted.append(comparison)

    matched_count = len(accepted)
    precision = matched_count / len(predictions) if predictions else 0.0
    recall = matched_count / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0.0
        else 0.0
    )
    all_errors = np.concatenate(
        [comparison["errors_px"] for comparison in accepted]
    ) if accepted else np.asarray([], dtype=np.float64)
    return {
        "predicted_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "matched_count": matched_count,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_error_px": (
            float(np.mean(all_errors)) if len(all_errors) else None
        ),
        "p95_error_px": (
            float(np.percentile(all_errors, 95.0)) if len(all_errors) else None
        ),
        "mean_normalized_error": (
            float(np.mean(all_errors) / float(image_width))
            if len(all_errors)
            else None
        ),
        "mean_coverage": (
            float(np.mean([item["coverage"] for item in accepted]))
            if accepted
            else 0.0
        ),
    }


def _curve_error(prediction, truth):
    prediction = prediction[np.argsort(prediction[:, 1])]
    truth = truth[np.argsort(truth[:, 1])]
    y_min = max(float(prediction[0, 1]), float(truth[0, 1]))
    y_max = min(float(prediction[-1, 1]), float(truth[-1, 1]))
    truth_span = max(1.0, float(truth[-1, 1] - truth[0, 1]))
    if y_max - y_min < 12.0:
        return None
    sample_y = np.arange(y_min, y_max + 1.0, 4.0)
    if len(sample_y) < 4:
        return None
    prediction_x = np.interp(
        sample_y,
        prediction[:, 1],
        prediction[:, 0],
    )
    truth_x = np.interp(sample_y, truth[:, 1], truth[:, 0])
    errors = np.abs(prediction_x - truth_x)
    return {
        "mean_error_px": float(np.mean(errors)),
        "p95_error_px": float(np.percentile(errors, 95.0)),
        "coverage": min(1.0, float(y_max - y_min) / truth_span),
        "errors_px": errors,
    }


def _marking_is_visible(marking):
    marking_type = str(getattr(marking, "type", "")).lower()
    marking_name = marking_type.rsplit(".", 1)[-1]
    return marking_name not in {"", "none", "other"}


def _continuity_cost(current, candidate):
    current_yaw = float(current.transform.rotation.yaw)
    candidate_yaw = float(candidate.transform.rotation.yaw)
    heading_delta = abs((candidate_yaw - current_yaw + 180.0) % 360.0 - 180.0)
    lane_penalty = 0.0
    if getattr(candidate, "road_id", None) != getattr(current, "road_id", None):
        lane_penalty += 5.0
    if getattr(candidate, "lane_id", None) != getattr(current, "lane_id", None):
        lane_penalty += 20.0
    return heading_delta + lane_penalty

"""Associate camera bounding boxes with front-radar detections."""

import math


def camera_focal_length_px(image_width, fov_deg):
    fov_rad = math.radians(float(fov_deg))
    return float(image_width) / (2.0 * math.tan(fov_rad / 2.0))


def pixel_bearing_deg(pixel_x, image_width, fov_deg):
    center_x = float(image_width) / 2.0
    focal_length = camera_focal_length_px(image_width, fov_deg)
    return math.degrees(math.atan((float(pixel_x) - center_x) / focal_length))


def bbox_bearing_deg(bbox, image_width, fov_deg):
    x1, _, x2, _ = bbox
    return pixel_bearing_deg((x1 + x2) / 2.0, image_width, fov_deg)


def bbox_bearing_interval_deg(bbox, image_width, fov_deg):
    x1, _, x2, _ = bbox
    left = pixel_bearing_deg(x1, image_width, fov_deg)
    right = pixel_bearing_deg(x2, image_width, fov_deg)
    return min(left, right), max(left, right)


def median(values):
    ordered = sorted(values)
    if not ordered:
        return None

    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _cluster_by_depth(points, minimum_gap_m=2.5):
    """Split angularly matched radar returns into separate depth layers."""
    if not points:
        return []

    ordered = sorted(points, key=lambda point: point["depth_m"])
    clusters = [[ordered[0]]]

    for point in ordered[1:]:
        previous_depth = clusters[-1][-1]["depth_m"]
        adaptive_gap = max(minimum_gap_m, 0.05 * previous_depth)
        if point["depth_m"] - previous_depth > adaptive_gap:
            clusters.append([point])
        else:
            clusters[-1].append(point)

    return clusters


def _summarize_nearest_cluster(points):
    clusters = _cluster_by_depth(points)
    if not clusters:
        return None

    multi_point_clusters = [cluster for cluster in clusters if len(cluster) >= 2]
    usable_clusters = multi_point_clusters or clusters
    cluster = min(
        usable_clusters,
        key=lambda item: median([point["depth_m"] for point in item]),
    )

    depths = sorted(point["depth_m"] for point in cluster)
    lower_quartile_index = int(round(0.25 * (len(depths) - 1)))

    return {
        "range_m": depths[lower_quartile_index],
        "bearing_deg": median([point["azimuth_deg"] for point in cluster]),
        "radar_velocity_mps": median(
            [point["relative_velocity_mps"] for point in cluster]
        ),
        "matched_points": len(cluster),
    }


def fuse_detections_with_radar(
    detections,
    detection_frame_id,
    radar_points,
    radar_frame_id,
    image_width,
    camera_fov_deg,
    fixed_delta_seconds,
    angular_padding_deg=0.75,
):
    """Add metric radar range and velocity to each camera detection.

    Radar points already belong to ``radar_frame_id``. Their range must not be
    extrapolated a second time. Frame age is used only to widen the angular
    association gate for an older camera box.
    """
    frame_delta = 0
    if detection_frame_id is not None and radar_frame_id is not None:
        frame_delta = max(0, int(radar_frame_id) - int(detection_frame_id))
    delta_t = frame_delta * float(fixed_delta_seconds)
    age_padding = min(4.0, 10.0 * delta_t)
    total_padding = float(angular_padding_deg) + age_padding

    fused = []
    for detection in detections:
        center_bearing = bbox_bearing_deg(
            detection["bbox"], image_width, camera_fov_deg
        )
        left, right = bbox_bearing_interval_deg(
            detection["bbox"], image_width, camera_fov_deg
        )
        left -= total_padding
        right += total_padding

        candidates = [
            point
            for point in radar_points
            if left <= point["azimuth_deg"] <= right
            and abs(point.get("altitude_deg", 0.0)) <= 6.0
            and 0.5 < point["depth_m"] <= 100.0
        ]
        match = _summarize_nearest_cluster(candidates)
        item = dict(detection)
        item["bbox_bearing_deg"] = center_bearing
        item["delta_t_s"] = delta_t

        if match is None:
            item.update(
                {
                    "bearing_deg": center_bearing,
                    "has_range": False,
                    "range_m": None,
                    "relative_velocity_mps": None,
                    "radar_points_matched": 0,
                }
            )
        else:
            item.update(
                {
                    "bearing_deg": match["bearing_deg"],
                    "has_range": True,
                    "range_m": match["range_m"],
                    # Preserve CARLA's signed radial velocity. The controller
                    # uses the observed sign directly: negative is closing and
                    # positive is pulling away in the live sensor stream.
                    "relative_velocity_mps": match["radar_velocity_mps"],
                    "radar_points_matched": match["matched_points"],
                }
            )

        fused.append(item)

    return fused

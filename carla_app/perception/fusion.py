"""On kamera kutularini on radar noktalariyla eslestirir."""

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



def sanitize_radar_points(points):
    # Return only finite, numeric radar samples.
    sanitized = []
    for point in points or []:
        try:
            depth = float(point["depth_m"])
            azimuth = float(point["azimuth_deg"])
            altitude = float(point.get("altitude_deg", 0.0))
            velocity = float(point["relative_velocity_mps"])
        except (KeyError, TypeError, ValueError):
            continue

        values = (depth, azimuth, altitude, velocity)
        if not all(math.isfinite(value) for value in values):
            continue
        if depth <= 0.0:
            continue

        normalized = dict(point)
        normalized.update(
            {
                "depth_m": depth,
                "azimuth_deg": azimuth,
                "altitude_deg": altitude,
                "relative_velocity_mps": velocity,
            }
        )
        sanitized.append(normalized)
    return sanitized


def cluster_by_depth(points, minimum_gap_m=2.5):
    """Ayni aciya dusen radar noktalarini mesafe katmanlarina ayirir."""
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


def summarize_nearest_cluster(points):
    """En yakin kararli radar kumesinin mesafe, aci ve hizini ozetler."""
    clusters = cluster_by_depth(points)
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
    """Her kamera tespitine radar mesafesi ve bagil hizi ekler.

    Radar noktasi zaten ``radar_frame_id`` karesine aittir; mesafesi ikinci
    kez ileri tasinmaz. Kare yasi yalnizca eski kamera kutusunun aci eslestirme
    payini bir miktar genisletmek icin kullanilir.
    """
    radar_points = sanitize_radar_points(radar_points)

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
        match = summarize_nearest_cluster(candidates)
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
                    # CARLA'nin bagil hiz isaretini koru. Canli akista negatif
                    # deger yaklasmayi, pozitif deger uzaklasmayi gosterir.
                    "relative_velocity_mps": match["radar_velocity_mps"],
                    "radar_points_matched": match["matched_points"],
                }
            )

        fused.append(item)

    return fused

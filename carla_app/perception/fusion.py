"""
Kamera (YOLO bbox) + radar füzyonu.

YOLO sadece 2D bbox verir, mesafe/hız bilgisi yoktur. Bu modül,
bbox merkezinin acisal konumunu (bearing) kamera intrinsiklerinden
hesaplar ve ayni yondeki radar noktalariyla eslestirerek her
tespite gercek mesafe (range) ve goreli hiz (relative velocity)
kazandirir.

"""
import math


def camera_focal_length_px(image_width, fov_deg):
    fov_rad = math.radians(fov_deg)
    return image_width / (2.0 * math.tan(fov_rad / 2.0))


def bbox_bearing_deg(bbox, image_width, fov_deg):
    x1, _, x2, _ = bbox
    bbox_center_x = (x1 + x2) / 2.0
    image_center_x = image_width / 2.0
    focal_length = camera_focal_length_px(image_width, fov_deg)
    return math.degrees(
        math.atan((bbox_center_x - image_center_x) / focal_length)
    )


def _summarize_matches(points):
    if not points:
        return None

    depths = [point["depth_m"] for point in points]
    velocities = sorted(point["relative_velocity_mps"] for point in points)

    # En yakin nokta genelde tampon/govde donusu -> hedefin on
    # yuzeyine en iyi mesafe tahmini budur.
    closest_index = depths.index(min(depths))

    count = len(velocities)
    middle = count // 2
    if count % 2 == 1:
        median_velocity = velocities[middle]
    else:
        median_velocity = 0.5 * (velocities[middle - 1] + velocities[middle])

    return {
        "range_m": depths[closest_index],
        "relative_velocity_mps": median_velocity,
        "matched_points": count,
    }


def fuse_detections_with_radar(
    detections,
    radar_points,
    image_width,
    camera_fov_deg,
    angular_gate_deg=3.0,
):
    """
    detections: PerceptionSystem.detect(...)['vehicles'] listesi
                (her biri 'bbox' iceren dict).
    radar_points: sensors.processors.radar_to_list(...) formatinda
                  liste (depth_m, relative_velocity_mps, azimuth_deg,
                  altitude_deg).

    Donen liste, her tespite asagidaki alanlari ekler:
        bearing_deg, has_range, range_m,
        relative_velocity_mps, radar_points_matched
    """
    fused = []

    for detection in detections:
        bearing = bbox_bearing_deg(
            detection["bbox"], image_width, camera_fov_deg
        )
        

        candidates = [
            point
            for point in radar_points
            if abs(point["azimuth_deg"] - bearing) <= angular_gate_deg
        ]

        match = _summarize_matches(candidates)

        fused_detection = dict(detection)
        fused_detection["bearing_deg"] = bearing

        if match is None:
            fused_detection["has_range"] = False
            fused_detection["range_m"] = None
            fused_detection["relative_velocity_mps"] = None
            fused_detection["radar_points_matched"] = 0
        else:
            fused_detection["has_range"] = True
            fused_detection["range_m"] = match["range_m"]
            fused_detection["relative_velocity_mps"] = (
                match["relative_velocity_mps"]
            )
            fused_detection["radar_points_matched"] = match["matched_points"]

        fused.append(fused_detection)

    return fused
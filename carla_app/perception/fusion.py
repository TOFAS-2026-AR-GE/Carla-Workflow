"""
Kamera bbox ve radar eslestirmesi.

Eslestirme:
- bbox'in tam acisal genisligini kullanir
- radar noktalarini derinlige gore cluster'lar
- minimum tek nokta yerine dusuk yuzdelik kullanir
- median radar hizi ve acisi kullanir
"""

import math


def camera_focal_length_px(
    image_width,
    fov_deg,
):
    fov_rad = math.radians(fov_deg)

    return image_width / (
        2.0
        * math.tan(fov_rad / 2.0)
    )


def pixel_bearing_deg(
    pixel_x,
    image_width,
    fov_deg,
):
    image_center_x = (
        image_width / 2.0
    )

    focal_length = (
        camera_focal_length_px(
            image_width,
            fov_deg,
        )
    )

    return math.degrees(
        math.atan(
            (
                pixel_x
                - image_center_x
            )
            / focal_length
        )
    )


def bbox_bearing_deg(
    bbox,
    image_width,
    fov_deg,
):
    x1, _, x2, _ = bbox

    return pixel_bearing_deg(
        (x1 + x2) / 2.0,
        image_width,
        fov_deg,
    )


def bbox_bearing_interval_deg(
    bbox,
    image_width,
    fov_deg,
):
    x1, _, x2, _ = bbox

    left = pixel_bearing_deg(
        x1,
        image_width,
        fov_deg,
    )

    right = pixel_bearing_deg(
        x2,
        image_width,
        fov_deg,
    )

    return (
        min(left, right),
        max(left, right),
    )


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2

    if len(ordered) % 2:
        return ordered[middle]

    return 0.5 * (
        ordered[middle - 1]
        + ordered[middle]
    )


def _cluster_by_depth(
    points,
    maximum_gap_m=2.5,
):
    if not points:
        return []

    ordered = sorted(
        points,
        key=lambda point: (
            point["depth_m"]
        ),
    )

    clusters = [[ordered[0]]]

    for point in ordered[1:]:
        previous_depth = (
            clusters[-1][-1][
                "depth_m"
            ]
        )

        adaptive_gap = max(
            maximum_gap_m,
            0.05 * previous_depth,
        )

        if (
            point["depth_m"]
            - previous_depth
            > adaptive_gap
        ):
            clusters.append([point])
        else:
            clusters[-1].append(point)

    return clusters


def _summarize_matches(points):
    if not points:
        return None

    clusters = _cluster_by_depth(
        points
    )

    multi_point_clusters = [
        cluster
        for cluster in clusters
        if len(cluster) >= 2
    ]

    usable_clusters = (
        multi_point_clusters
        or clusters
    )

    cluster = min(
        usable_clusters,
        key=lambda item: _median(
            [
                point["depth_m"]
                for point in item
            ]
        ),
    )

    depths = sorted(
        point["depth_m"]
        for point in cluster
    )

    # En yakin tek outlier yerine
    # cluster'in dusuk yuzdeligi.
    percentile_index = int(
        round(
            0.25
            * (len(depths) - 1)
        )
    )

    range_m = depths[
        percentile_index
    ]

    return {
        "range_m": range_m,
        "bearing_deg": _median(
            [
                point["azimuth_deg"]
                for point in cluster
            ]
        ),
        "relative_velocity_mps": (
            _median(
                [
                    point[
                        "relative_velocity_mps"
                    ]
                    for point in cluster
                ]
            )
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
    delta_t = 0.0

    if (
        detection_frame_id is not None
        and radar_frame_id is not None
    ):
        delta_frames = (
            radar_frame_id
            - detection_frame_id
        )

        delta_t = max(
            0.0,
            delta_frames
            * fixed_delta_seconds,
        )

    fused = []

    for detection in detections:
        bbox_bearing = (
            bbox_bearing_deg(
                detection["bbox"],
                image_width,
                camera_fov_deg,
            )
        )

        (
            left_bearing,
            right_bearing,
        ) = bbox_bearing_interval_deg(
            detection["bbox"],
            image_width,
            camera_fov_deg,
        )

        left_bearing -= (
            angular_padding_deg
        )
        right_bearing += (
            angular_padding_deg
        )

        candidates = [
            point
            for point in radar_points
            if (
                left_bearing
                <= point["azimuth_deg"]
                <= right_bearing
                and abs(
                    point.get(
                        "altitude_deg",
                        0.0,
                    )
                )
                <= 6.0
                and 0.5
                < point["depth_m"]
                <= 100.0
            )
        ]

        match = _summarize_matches(
            candidates
        )

        fused_detection = dict(
            detection
        )

        fused_detection[
            "bbox_bearing_deg"
        ] = bbox_bearing

        fused_detection[
            "delta_t_s"
        ] = delta_t

        if match is None:
            fused_detection.update(
                {
                    "bearing_deg": (
                        bbox_bearing
                    ),
                    "has_range": False,
                    "range_m": None,
                    "raw_range_m": None,
                    "relative_velocity_mps": (
                        None
                    ),
                    "radar_points_matched": 0,
                }
            )
        else:
            raw_range = match[
                "range_m"
            ]

            relative_velocity = match[
                "relative_velocity_mps"
            ]

            adjusted_range = max(
                0.0,
                raw_range
                - relative_velocity
                * delta_t,
            )

            fused_detection.update(
                {
                    "bearing_deg": (
                        match[
                            "bearing_deg"
                        ]
                    ),
                    "has_range": True,
                    "range_m": (
                        adjusted_range
                    ),
                    "raw_range_m": (
                        raw_range
                    ),
                    "relative_velocity_mps": (
                        relative_velocity
                    ),
                    "radar_points_matched": (
                        match[
                            "matched_points"
                        ]
                    ),
                }
            )

        fused.append(
            fused_detection
        )

    return fused
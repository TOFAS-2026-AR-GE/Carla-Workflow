"""
Kamera (YOLO bbox) + radar füzyonu.

YOLO sadece 2D bbox verir, mesafe/hız bilgisi yoktur. Bu modül,
bbox merkezinin acisal konumunu (bearing) kamera intrinsiklerinden
hesaplar ve ayni yondeki radar noktalariyla eslestirerek her
tespite gercek mesafe (range) ve goreli hiz (relative velocity)
kazandirir.

Varsayimlar (Faz 1 checkpoint'inde dogrulanmali):
- camera_front_wide ve radar_front_long ikisi de yaw=0 (arac
  ileri yonune hizali), bu yuzden azimuth/bearing dogrudan
  karsilastirilabilir. Kucuk pitch farki (-4 derece kamera,
  0 derece radar) yatay bearing'i etkilemez.
- CARLA radar azimuth isareti: pozitif = sag. Bbox bearing de
  ayni isaret kuralinda hesaplaniyor (pikselin goruntu merkezine
  gore sagda olmasi -> pozitif bearing). Sahnede sagdaki/soldaki
  bir aracla test edip bu isaretin tutarli oldugunu dogrula.
- CARLA radar "velocity": nesnenin sensore gore radyal hizi.
  Isaretin "yaklasiyor" mu "uzaklasiyor" mu anlamina geldigini
  bilinen bir sahnede (durgun ego + yaklasan arac) dogrula ve
  gerekirse asagidaki fonksiyonlari isaret ters cevirecek sekilde
  guncelle.
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
    detection_frame_id,
    radar_points,
    radar_frame_id,
    image_width,
    camera_fov_deg,
    fixed_delta_seconds,
    angular_gate_deg=3.0,
):
    """
    detections: PerceptionSystem.detect(...)['vehicles'] listesi
                (her biri 'bbox' iceren dict).
    detection_frame_id: perception_result['frame_id'] -> YOLO'nun
                hangi kamera frame'ini isledigini soyler.
    radar_points: sensors.processors.radar_to_list(...) formatinda
                  liste (depth_m, relative_velocity_mps, azimuth_deg,
                  altitude_deg).
    radar_frame_id: bu radar paketinin frame numarasi (RadarStream
                herhangi bir bekleme yapmadigi icin, cagrildigi anki
                "en guncel" frame'dir).
    fixed_delta_seconds: simulasyon tick suresi (config.py), frame
                farkini saniyeye cevirmek icin.

    YOLO worker asenkron ve gecikmeli calistigi icin
    (perception_every_n_frames + inference suresi), detection_frame_id
    genelde radar_frame_id'den birkac tick GERIDE kalir. Bu fark
    dikkate alinmazsa, "su anki" radar mesafesi "birkac tick onceki"
    bbox ile eslestirilmis olur -> araç hizli hareket ediyorsa
    (ornegin 20 m/s, 3 tick = 150ms -> ~3m) sistematik hata olusur.

    Duzeltme: radar mesafesini, olcum aninda bilinen goreli hizla,
    detection'in ait oldugu (daha eski) ana geri ekstrapole ediyoruz:
        delta_t = (radar_frame_id - detection_frame_id) * dt
        range_adjusted = range_raw - relative_velocity * delta_t

    Donen liste, her tespite asagidaki alanlari ekler:
        bearing_deg, has_range, range_m (duzeltilmis),
        raw_range_m (ham radar olcumu), delta_t_s,
        relative_velocity_mps, radar_points_matched
    """
    delta_t = 0.0
    if detection_frame_id is not None and radar_frame_id is not None:
        delta_frames = radar_frame_id - detection_frame_id
        delta_t = delta_frames * fixed_delta_seconds

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
        fused_detection["delta_t_s"] = delta_t

        if match is None:
            fused_detection["has_range"] = False
            fused_detection["range_m"] = None
            fused_detection["raw_range_m"] = None
            fused_detection["relative_velocity_mps"] = None
            fused_detection["radar_points_matched"] = 0
        else:
            raw_range = match["range_m"]
            relative_velocity = match["relative_velocity_mps"]
            adjusted_range = raw_range - relative_velocity * delta_t
            # Negatif/anlamsiz mesafeye karsi guvenlik payi -
            # ekstrapolasyon buyuk sapma verirse (ornegin isaret
            # yanlissa) burada fark edilir.
            adjusted_range = max(adjusted_range, 0.0)

            fused_detection["has_range"] = True
            fused_detection["range_m"] = adjusted_range
            fused_detection["raw_range_m"] = raw_range
            fused_detection["relative_velocity_mps"] = relative_velocity
            fused_detection["radar_points_matched"] = match["matched_points"]

        fused.append(fused_detection)

    return fused
"""Kalman filtresiyle birden fazla aracı kareler arasında takip eder.

Ne işe yarar?
``fusion.py`` her çevrimde gürültülü bir mesafe-açı ölçümü verir.
Bu modül aynı aracı kareler arasında izleyip sabit bir kimlik verir. Ölçümü
"araç sabit hıza yakın hareket eder" varsayımıyla yumuşatır ve hızını tahmin
eder.

Nasıl çalışır?
Her eksen için pozisyon ve hız tutulur. İki basit adım vardır:

1. Tahmin: Yeni ölçüm gelmeden önce eski hıza göre konum ilerletilir.
2. Güncelleme: Yeni ölçüm geldiğinde tahmin ile ölçüm güven oranına göre
   birleştirilir.

X ve Y eksenleri ayrı hesaplanır. Bu bilinçli sadeleştirme kodu okunabilir
tutar ve bu projenin araç takibi için yeterlidir.
"""

import math


class Axis1DKalman:
    """Tek eksen için sabit hızlı Kalman filtresi.

    ``pos`` konumu, ``vel`` hızı; ``p_pp``, ``p_pv`` ve ``p_vv`` ise
    tahminin belirsizliğini tutar.
    """

    def __init__(self, position, process_noise=1.0, measurement_noise=1.5):
        self.pos = position
        self.vel = 0.0

        # Başlangıç konumu ölçümden gelir; hız bilinmediği için hız
        # belirsizliği yüksek başlatılır.
        self.p_pp = 2.0
        self.p_pv = 0.0
        self.p_vv = 8.0

        self.q = process_noise  # Hareket modelinin belirsizliği.
        self.r = measurement_noise  # Sensör ölçümünün belirsizliği.

    def predict(self, dt):
        # 1) Pozisyonu hıza göre ilerlet; hız sabit kalır.
        self.pos += self.vel * dt

        # 2) Zaman geçtikçe tahmin belirsizliğini büyüt.
        p_pp = self.p_pp + 2 * dt * self.p_pv + dt * dt * self.p_vv
        p_pv = self.p_pv + dt * self.p_vv
        p_vv = self.p_vv

        # 3) Hareket modelinin kendi belirsizliğini ekle.
        self.p_pp = p_pp + self.q * dt**4 / 4
        self.p_pv = p_pv + self.q * dt**3 / 2
        self.p_vv = p_vv + self.q * dt**2

    def update(self, measured_position):
        # Tahmin ile gerçek ölçüm arasındaki fark.
        residual = measured_position - self.pos

        # Tahmin belirsizliği büyükse ölçüme daha çok, sensör belirsizliği
        # büyükse ölçüme daha az güvenilir. Bu oran her adımda hesaplanır.
        innovation_var = self.p_pp + self.r
        k_pos = self.p_pp / innovation_var
        k_vel = self.p_pv / innovation_var

        self.pos += k_pos * residual
        self.vel += k_vel * residual

        # Ölçüm geldiği için belirsizlik azalır.
        new_p_pp = (1 - k_pos) * self.p_pp
        new_p_pv = (1 - k_pos) * self.p_pv
        new_p_vv = self.p_vv - k_vel * self.p_pv

        self.p_pp, self.p_pv, self.p_vv = new_p_pp, new_p_pv, new_p_vv


class Track:
    """Tek bir araç için X ve Y eksenlerindeki takip bilgisini tutar."""

    def __init__(
        self, track_id, x, y, class_name, process_noise=1.0, measurement_noise=1.5
    ):
        self.id = track_id
        self.class_name = class_name

        self.kx = Axis1DKalman(x, process_noise, measurement_noise)
        self.ky = Axis1DKalman(y, process_noise, measurement_noise)
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

        self.hit_count = 1  # Gerçek ölçümle kaç kez eşleşti.
        self.miss_count = 0  # Üst üste kaç kez eşleşmedi.
        self.confirmed = False  # Üç eşleşmeden sonra güvenilir sayılır.

        # Durum kaydı için son ham ölçüm.
        self.last_range_m = None
        self.last_bearing_deg = None
        self.last_relative_velocity_mps = None
        self.last_measurement_frame_id = None

    def predict(self, dt):
        self.kx.predict(dt)
        self.ky.predict(dt)
        self.copy_filter_values()

    def update(self, measured_x, measured_y):
        self.kx.update(measured_x)
        self.ky.update(measured_y)
        self.copy_filter_values()

        self.hit_count += 1
        self.miss_count = 0
        if self.hit_count >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.miss_count += 1

    def copy_filter_values(self):
        """Kalman sonucunu dışarıdan okunabilen basit alanlara kopyalar."""
        self.x = self.kx.pos
        self.y = self.ky.pos
        self.vx = self.kx.vel
        self.vy = self.ky.vel


def polar_to_world(range_m, bearing_deg, ego_x, ego_y, ego_yaw_deg):
    """Araca göre mesafe-açı ölçümünü dünya koordinatına çevirir."""
    bearing_rad = math.radians(bearing_deg)
    yaw_rad = math.radians(ego_yaw_deg)

    x_local = range_m * math.cos(bearing_rad)  # Araca göre ileri yön.
    y_local = range_m * math.sin(bearing_rad)  # Araca göre sağ yön.

    world_x = ego_x + x_local * math.cos(yaw_rad) - y_local * math.sin(yaw_rad)
    world_y = ego_y + x_local * math.sin(yaw_rad) + y_local * math.cos(yaw_rad)
    return world_x, world_y


class Tracker:
    """Ölçümleri en yakın araç takibiyle eşleştirir ve eski takipleri siler."""

    def __init__(self, gate_distance_m=5.0, max_misses=5):
        self.tracks = []
        self.next_id = 1
        # Bir ölçümün mevcut takiple eşleşebileceği en büyük uzaklık.
        self.gate_distance_m = gate_distance_m
        # Takip bu kadar çevrim kaybolduktan sonra silinir.
        self.max_misses = max_misses

    def step(self, dt, measurements):
        """Bir çevrim tahmin, eşleştirme, güncelleme ve temizleme yapar.

        Ölçüm yoksa boş liste verilebilir; mevcut takipler hareket modeliyle
        kısa süre tahmin edilmeye devam eder.
        """
        # 1) Bütün takipleri bir adım ileri taşı.
        for track in self.tracks:
            track.predict(dt)

        # 2) En yakın çiftlerden başlayarak ölçüm ile takibi eşleştir.
        used_tracks = set()
        used_measurements = set()
        pairs = []
        for ti, track in enumerate(self.tracks):
            for mi, meas in enumerate(measurements):
                dist = math.hypot(track.x - meas["x"], track.y - meas["y"])
                if dist <= self.gate_distance_m:
                    pairs.append((dist, ti, mi))
        # Üçlüler önce mesafeye göre sıralanır. Eşitlikte takip ve ölçüm
        # sırası kullanılır; gizli bir sıralama kuralı yoktur.
        pairs.sort()

        for _, ti, mi in pairs:
            if ti in used_tracks or mi in used_measurements:
                continue
            used_tracks.add(ti)
            used_measurements.add(mi)

            track, meas = self.tracks[ti], measurements[mi]
            track.update(meas["x"], meas["y"])
            track.class_name = meas["class_name"]
            track.last_range_m = meas.get("range_m")
            track.last_bearing_deg = meas.get("bearing_deg")
            track.last_relative_velocity_mps = meas.get("relative_velocity_mps")
            track.last_measurement_frame_id = meas.get("measurement_frame_id")

        # 3) Eşleşmeyen takiplerin kayıp sayacını artır.
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.mark_missed()

        # 4) Eşleşmeyen ölçümler için yeni takip aç.
        for mi, meas in enumerate(measurements):
            if mi in used_measurements:
                continue
            new_track = Track(self.next_id, meas["x"], meas["y"], meas["class_name"])
            new_track.last_range_m = meas.get("range_m")
            new_track.last_bearing_deg = meas.get("bearing_deg")
            new_track.last_relative_velocity_mps = meas.get("relative_velocity_mps")
            new_track.last_measurement_frame_id = meas.get("measurement_frame_id")
            self.tracks.append(new_track)
            self.next_id += 1

        # 5) Uzun süredir eşleşmeyen takipleri temizle.
        remaining_tracks = []
        for track in self.tracks:
            if track.miss_count <= self.max_misses:
                remaining_tracks.append(track)
        self.tracks = remaining_tracks
        return self.tracks

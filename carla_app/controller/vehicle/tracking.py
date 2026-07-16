"""
Coklu nesne takibi (multi-object tracking) - Kalman filtresi.

NE ISE YARAR?
fusion.py her tick bagimsiz, gurultulu bir olcum verir (mesafe, aci).
Bu modul, ayni araci frame'ler arasinda takip ederek ona sabit bir ID
verir ve gurultulu olcumleri "arac fizik kurallarina gore hareket
eder" varsayimiyla yumusatip, hizini da tahmin eder.

NASIL CALISIR? (Kalman filtresi, sade anlatim)
Her eksen (x ve y) icin ayri ayri, 2 sayidan olusan bir durum tutariz:
[pozisyon, hiz]. Iki adim var:

  1) TAHMIN (predict): Yeni olcum gelmeden once, "simdiye kadarki
     hizima gore su an muhtemelen neredeyim" diye pozisyonu ilerletiriz.
     Bununla birlikte "ne kadar emin oldugumuzu" (belirsizlik) da
     buyuturuz - zaman gectikce daha az eminizdir.

  2) GUNCELLEME (update): Gercek bir olcum geldiginde, tahminimiz ile
     olcum arasindaki farka bakariz. Bu farki ne kadar ciddiye
     alacagimizi (KAZANC) modelin ve olcumun ne kadar guvenilir
     oldugunu karsilastirarak HESAPLARIZ (sabit bir sayi degil,
     her adimda yeniden hesaplanir) - bu, Kalman filtresini basit
     "eski deger + orandan bir kismi ekle" yontemlerinden ayiran asil
     fark: kazanc otomatik ayarlanir.

Eksenler (x,y) birbirinden bagimsiz varsayilir (aralarindaki capraz
belirsizlik ihmal edilir) - bu kucuk bir basitlestirme ama pratikte
performansi neredeyse hic etkilemez, karsiliginda kod cok daha
okunabilir olur.
"""

import math


class Axis1DKalman:
    """
    Tek bir eksen (x veya y) icin sabit-hiz modelli Kalman filtresi.
    State: pozisyon (pos) ve hiz (vel). Belirsizlik 3 sayiyla tutulur:
    p_pp (pozisyon belirsizligi), p_vv (hiz belirsizligi), p_pv
    (ikisi arasindaki iliski).
    """

    def __init__(self, position, process_noise=1.0, measurement_noise=1.5):
        self.pos = position
        self.vel = 0.0

        # Baslangicta pozisyonu biliyoruz (olcumden geldi) ama hizi
        # bilmiyoruz -> hiz belirsizligini yuksek baslatiyoruz.
        self.p_pp = 2.0
        self.p_pv = 0.0
        self.p_vv = 8.0

        self.q = process_noise  # model ne kadar "kararsiz" (surec gurultusu)
        self.r = measurement_noise  # sensor ne kadar gurultulu (olcum gurultusu)

    def predict(self, dt):
        # 1) Pozisyonu hiza gore ilerlet, hiz sabit kalir.
        self.pos += self.vel * dt

        # 2) Belirsizligi de ilerlet (zaman gectikce daha az eminiz).
        p_pp = self.p_pp + 2 * dt * self.p_pv + dt * dt * self.p_vv
        p_pv = self.p_pv + dt * self.p_vv
        p_vv = self.p_vv

        # 3) Ustune "model tahmini de mukemmel degil" gurultusu ekle.
        self.p_pp = p_pp + self.q * dt**4 / 4
        self.p_pv = p_pv + self.q * dt**3 / 2
        self.p_vv = p_vv + self.q * dt**2

    def update(self, measured_position):
        # Tahmin ile gercek olcum arasindaki fark.
        residual = measured_position - self.pos

        # Kazanc: belirsizligimiz (p_pp) buyukse olcume daha cok
        # guven; sensor gurultusu (r) buyukse olcume daha az guven.
        # Bu oran HER ADIMDA yeniden hesaplanir - Kalman'i "basit
        # sabit oranli düzeltme" yontemlerinden ayiran asil nokta budur.
        innovation_var = self.p_pp + self.r
        k_pos = self.p_pp / innovation_var
        k_vel = self.p_pv / innovation_var

        self.pos += k_pos * residual
        self.vel += k_vel * residual

        # Olcum geldigi icin belirsizlik azalir.
        new_p_pp = (1 - k_pos) * self.p_pp
        new_p_pv = (1 - k_pos) * self.p_pv
        new_p_vv = self.p_vv - k_vel * self.p_pv

        self.p_pp, self.p_pv, self.p_vv = new_p_pp, new_p_pv, new_p_vv


class Track:
    """
    Tek bir aracin takibi. Icinde x ve y icin ayri birer
    Axis1DKalman calisir (birbirinden bagimsiz).
    """

    def __init__(
        self, track_id, x, y, class_name, process_noise=1.0, measurement_noise=1.5
    ):
        self.id = track_id
        self.class_name = class_name

        self.kx = Axis1DKalman(x, process_noise, measurement_noise)
        self.ky = Axis1DKalman(y, process_noise, measurement_noise)

        self.hit_count = 1  # kac kere gercek olcumle eslesti
        self.miss_count = 0  # ust uste kac kere eslesmedi
        self.confirmed = False  # yeterince eslesince "guvenilir" sayilir

        # Debug/log icin son ham olcum.
        self.last_range_m = None
        self.last_bearing_deg = None
        self.last_relative_velocity_mps = None

    def predict(self, dt):
        self.kx.predict(dt)
        self.ky.predict(dt)

    def update(self, measured_x, measured_y):
        self.kx.update(measured_x)
        self.ky.update(measured_y)

        self.hit_count += 1
        self.miss_count = 0
        if self.hit_count >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.miss_count += 1

    @property
    def x(self):
        return self.kx.pos

    @property
    def y(self):
        return self.ky.pos

    @property
    def vx(self):
        return self.kx.vel

    @property
    def vy(self):
        return self.ky.vel


def polar_to_world(range_m, bearing_deg, ego_x, ego_y, ego_yaw_deg):
    """Ego-goreli (mesafe, aci) olcumunu dunya koordinatina (x,y) cevirir."""
    bearing_rad = math.radians(bearing_deg)
    yaw_rad = math.radians(ego_yaw_deg)

    x_local = range_m * math.cos(bearing_rad)  # ego'ya gore ileri
    y_local = range_m * math.sin(bearing_rad)  # ego'ya gore sag

    world_x = ego_x + x_local * math.cos(yaw_rad) - y_local * math.sin(yaw_rad)
    world_y = ego_y + x_local * math.sin(yaw_rad) + y_local * math.cos(yaw_rad)
    return world_x, world_y


class Tracker:
    """
    Tum track'leri yonetir: yeni olcumleri en yakin track'e
    eslestirir, eslesmeyen olcumler icin yeni track acar, uzun
    suredir eslesmeyen track'leri siler.
    """

    def __init__(self, gate_distance_m=5.0, max_misses=5):
        self.tracks = []
        self._next_id = 1
        self.gate_distance_m = gate_distance_m  # eslesme icin izin verilen max mesafe
        self.max_misses = max_misses  # bu kadar kayiptan sonra track silinir

    def step(self, dt, measurements):
        """
        measurements: [{"x":.., "y":.., "class_name":.., "range_m":..,
                         "bearing_deg":.., "relative_velocity_mps":..}, ...]
        Olcum yoksa bos liste verilebilir - track'ler yine de predict
        edilir (fizik modeliyle "kaymaya" devam eder).
        """
        # 1) Once herkesi bir adim ileri tasi.
        for track in self.tracks:
            track.predict(dt)

        # 2) En yakin ciftlerden baslayarak olcum-track eslestir.
        used_tracks, used_measurements = set(), set()
        pairs = []
        for ti, track in enumerate(self.tracks):
            for mi, meas in enumerate(measurements):
                dist = math.hypot(track.x - meas["x"], track.y - meas["y"])
                if dist <= self.gate_distance_m:
                    pairs.append((dist, ti, mi))
        pairs.sort(key=lambda p: p[0])

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

        # 3) Eslesmeyen track'lerin kayip sayacini arttir.
        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.mark_missed()

        # 4) Eslesmeyen olcumler icin yeni track ac.
        for mi, meas in enumerate(measurements):
            if mi in used_measurements:
                continue
            new_track = Track(self._next_id, meas["x"], meas["y"], meas["class_name"])
            new_track.last_range_m = meas.get("range_m")
            new_track.last_bearing_deg = meas.get("bearing_deg")
            new_track.last_relative_velocity_mps = meas.get("relative_velocity_mps")
            self.tracks.append(new_track)
            self._next_id += 1

        # 5) Cok uzun suredir eslesmeyenleri temizle.
        self.tracks = [t for t in self.tracks if t.miss_count <= self.max_misses]
        return self.tracks

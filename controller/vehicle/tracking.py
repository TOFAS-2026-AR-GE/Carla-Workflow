"""
Coklu nesne takibi (multi-object tracking).

fusion.py bize her tick icin bagimsiz, "hafizasiz" olcumler verir
(range, bearing, relative_velocity). Bu modul bu olcumleri zaman
icinde birbirine baglayarak:
  - gurultuyu filtreler (Kalman filtresi, sabit hiz modeli),
  - her nesneye kalici bir ID verir,
  - gelecek pozisyonu tahmin etmeyi mumkun kilar (MPC'nin ihtiyaci).

Tasarim karari: track state'i EGO-GORELI degil, DUNYA (world-frame)
koordinatinda tutuluyor. Sebep: ego'nun kendi manevralari (viraj,
hizlanma) ego-goreli cercevede "sabit hiz" varsayimini bozar; dunya
çerçevesinde diger aracin kendi hareketi gercekten sabit hiza daha
yakindir. Kontrol katmaninda (MPC), o anki ego pozuyla tekrar
ego-goreli çevrilecek.

Bagimlilik: sadece numpy + math. CARLA import edilmiyor (test
edilebilirlik icin), world-frame donusumu için ego pozu disaridan
(application.py'den, read_vehicle_state cikisindan) parametre olarak
geliyor.
"""

import math

import numpy as np


def polar_to_world(range_m, bearing_deg, ego_x, ego_y, ego_yaw_deg):
    """
    Ego-goreli kutupsal olcumu (range, bearing) dunya kartezyenine
    cevirir. CARLA konvansiyonu: x=ileri, y=sag, yaw saat yonunde
    pozitif (derece).
    """
    bearing_rad = math.radians(bearing_deg)
    yaw_rad = math.radians(ego_yaw_deg)

    x_local = range_m * math.cos(bearing_rad)
    y_local = range_m * math.sin(bearing_rad)

    world_x = ego_x + x_local * math.cos(yaw_rad) - y_local * math.sin(yaw_rad)
    world_y = ego_y + x_local * math.sin(yaw_rad) + y_local * math.cos(yaw_rad)

    return world_x, world_y


class KalmanTrack:
    """
    Sabit hiz (constant velocity) modelli tek bir nesne takibi.
    State: [x, y, vx, vy] (dunya cercevesi, metre ve m/s).
    """

    def __init__(
        self,
        track_id,
        x,
        y,
        class_name,
        process_noise=1.5,
        measurement_noise=1.5,
    ):
        self.id = track_id
        self.class_name = class_name

        self.state = np.array([x, y, 0.0, 0.0])
        # Baslangicta konum bilgisi var ama hiz bilinmiyor ->
        # hiz bilesenlerinde belirsizlik yuksek baslatiliyor.
        self.P = np.diag([2.0, 2.0, 8.0, 8.0])

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        self.hits = 1
        self.misses = 0
        self.age = 0
        self.confirmed = False

        # Son eslesen ham olcum - debug/log icin saklaniyor.
        self.last_range_m = None
        self.last_bearing_deg = None
        self.last_relative_velocity_mps = None

    def predict(self, dt):
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # "White noise acceleration" surec gurultusu modeli:
        # aracin ivmesi ongorulemez kucuk bir gurultu olarak
        # modellenir, bu da dt buyudukce belirsizligi artirir.
        q = self.process_noise
        process_covariance = q * np.array(
            [
                [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
                [0.0, dt**4 / 4, 0.0, dt**3 / 2],
                [dt**3 / 2, 0.0, dt**2, 0.0],
                [0.0, dt**3 / 2, 0.0, dt**2],
            ]
        )

        self.state = transition @ self.state
        self.P = transition @ self.P @ transition.T + process_covariance
        self.age += 1

    def update(self, x, y):
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        measurement_covariance = np.eye(2) * self.measurement_noise

        measurement = np.array([x, y])
        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = (
            measurement_matrix @ self.P @ measurement_matrix.T
            + measurement_covariance
        )
        kalman_gain = (
            self.P
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = self.state + kalman_gain @ innovation
        self.P = (
            np.eye(4) - kalman_gain @ measurement_matrix
        ) @ self.P

        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.confirmed = True

    def mark_missed(self):
        self.misses += 1

    @property
    def x(self):
        return float(self.state[0])

    @property
    def y(self):
        return float(self.state[1])

    @property
    def vx(self):
        return float(self.state[2])

    @property
    def vy(self):
        return float(self.state[3])

    @property
    def speed_mps(self):
        return float(math.hypot(self.state[2], self.state[3]))

    def predicted_position(self, horizon_s):
        # MPC'nin ileri ufuk (t+horizon) icin kullanacagi basit
        # dogrusal projeksiyon.
        return (
            self.x + self.vx * horizon_s,
            self.y + self.vy * horizon_s,
        )


class Tracker:
    """
    Track yasam dongusunu (olustur / guncelle / sil) ve
    olcum-track eslestirmesini (nearest-neighbor + gating) yonetir.
    """

    def __init__(
        self,
        gate_distance_m=5.0,
        max_misses=5,
        process_noise=1.5,
        measurement_noise=1.5,
    ):
        self.tracks = []
        self._next_id = 1

        self.gate_distance_m = gate_distance_m
        self.max_misses = max_misses
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def step(self, dt, measurements):
        """
        measurements: [{x, y, class_name, range_m, bearing_deg,
                         relative_velocity_mps}, ...] (dunya
                         cercevesi x,y - donusumu cagiran taraf yapar)

        Her cagrida once tum track'ler predict edilir (dt kadar
        ileri), sonra varsa yeni olcumlerle guncellenir. Bu sayede
        yeni algi verisi olmasa bile (ornegin YOLO henuz yeni frame
        islememisse) track'ler fizik modeliyle "kaymaya" devam eder.
        """
        for track in self.tracks:
            track.predict(dt)

        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_measurements = set(range(len(measurements)))

        candidate_pairs = []
        for track_index, track in enumerate(self.tracks):
            for meas_index, meas in enumerate(measurements):
                distance = math.hypot(
                    track.x - meas["x"], track.y - meas["y"]
                )
                if distance <= self.gate_distance_m:
                    candidate_pairs.append(
                        (distance, track_index, meas_index)
                    )

        # Greedy nearest-neighbor: en yakin eslesmeden basla,
        # her track/olcum sadece bir kere kullanilsin.
        candidate_pairs.sort(key=lambda item: item[0])

        assignments = []
        for _, track_index, meas_index in candidate_pairs:
            if (
                track_index in unmatched_tracks
                and meas_index in unmatched_measurements
            ):
                assignments.append((track_index, meas_index))
                unmatched_tracks.discard(track_index)
                unmatched_measurements.discard(meas_index)

        for track_index, meas_index in assignments:
            meas = measurements[meas_index]
            track = self.tracks[track_index]

            track.update(meas["x"], meas["y"])
            track.class_name = meas["class_name"]
            track.last_range_m = meas.get("range_m")
            track.last_bearing_deg = meas.get("bearing_deg")
            track.last_relative_velocity_mps = meas.get(
                "relative_velocity_mps"
            )

        for track_index in unmatched_tracks:
            self.tracks[track_index].mark_missed()

        for meas_index in unmatched_measurements:
            meas = measurements[meas_index]
            new_track = KalmanTrack(
                self._next_id,
                meas["x"],
                meas["y"],
                meas["class_name"],
                process_noise=self.process_noise,
                measurement_noise=self.measurement_noise,
            )
            new_track.last_range_m = meas.get("range_m")
            new_track.last_bearing_deg = meas.get("bearing_deg")
            new_track.last_relative_velocity_mps = meas.get(
                "relative_velocity_mps"
            )
            self.tracks.append(new_track)
            self._next_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.misses <= self.max_misses
        ]

        return self.tracks

    def confirmed_tracks(self):
        return [track for track in self.tracks if track.confirmed]
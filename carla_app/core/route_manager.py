"""Araç için değişmeden ilerleyen bir referans rota üretir."""

import math

import carla


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PersistentRouteManager:
    """Aracı her çevrim en yakın şeride taşımadan sürekli rota tutar.

    Araç kısa süreliğine şerit çizgisini geçse bile rota değişmez. Yalnızca
    rota uzun süre gerçekten kaybedilirse yeniden bulunur.
    """

    def __init__(
        self,
        carla_map,
        spacing_m=1.0,
        horizon_m=80.0,
        recovery_distance_m=8.0,
        recovery_ticks=20,
    ):
        self.map = carla_map
        self.spacing_m = float(spacing_m)
        self.horizon_m = float(horizon_m)

        self.required_points = max(
            20,
            int(round(horizon_m / spacing_m)),
        )

        self.recovery_distance_m = float(recovery_distance_m)
        self.recovery_ticks = int(recovery_ticks)

        self.waypoints = []
        self.lost_ticks = 0

    def update(self, vehicle_location):
        if not self.waypoints:
            self.initialize(vehicle_location)

        self.trim_passed_waypoints(vehicle_location)
        self.extend_route()

        deviation = self.distance_to_route(vehicle_location)

        if deviation > self.recovery_distance_m:
            self.lost_ticks += 1
        else:
            self.lost_ticks = 0

        # Araç komşu şeride kısa süreliğine geçince rota değiştirme.
        # Yalnızca eski rota uzun süre gerçekten kaybedilirse tekrar bul.
        if self.lost_ticks >= self.recovery_ticks:
            self.initialize(vehicle_location)
            self.lost_ticks = 0

        self.trim_passed_waypoints(vehicle_location)
        self.extend_route()

        return (
            self.current_waypoint(vehicle_location),
            self.reference_locations(),
        )

    def initialize(self, vehicle_location):
        waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            raise RuntimeError("Arac icin surus seridi bulunamadi.")

        self.waypoints = [waypoint]
        self.extend_route()

    def trim_passed_waypoints(self, vehicle_location):
        if len(self.waypoints) < 2:
            return

        # Araca en yakın bölgeyi bul. İzdüşümün kararlı kalması için
        # aracın arkasında iki yol noktası bırak.
        search_count = min(25, len(self.waypoints))

        nearest_index = 0
        nearest_distance = math.inf
        for index in range(search_count):
            distance = self.distance_between(
                vehicle_location,
                self.waypoints[index].transform.location,
            )
            if distance < nearest_distance:
                nearest_index = index
                nearest_distance = distance

        if nearest_index > 2:
            del self.waypoints[: nearest_index - 2]

        # Aracın tamamen gerisinde kalmış yol noktalarını temizle.
        while len(self.waypoints) >= 3:
            first = self.waypoints[0].transform.location
            second = self.waypoints[1].transform.location

            segment_x = second.x - first.x
            segment_y = second.y - first.y
            segment_length = math.hypot(
                segment_x,
                segment_y,
            )

            if segment_length < 1e-4:
                self.waypoints.pop(0)
                continue

            relative_x = vehicle_location.x - first.x
            relative_y = vehicle_location.y - first.y

            along_track = (
                relative_x * segment_x + relative_y * segment_y
            ) / segment_length

            if along_track <= segment_length:
                break

            self.waypoints.pop(0)

    def extend_route(self):
        while len(self.waypoints) < self.required_points:
            last = self.waypoints[-1]
            candidates = list(last.next(self.spacing_m))

            if not candidates:
                break

            next_waypoint = self.choose_continuous_candidate(
                last,
                candidates,
            )
            self.waypoints.append(next_waypoint)

    def choose_continuous_candidate(
        self,
        last,
        candidates,
    ):
        if len(self.waypoints) >= 2:
            previous_location = self.waypoints[-2].transform.location
            last_location = last.transform.location

            previous_heading = math.atan2(
                last_location.y - previous_location.y,
                last_location.x - previous_location.x,
            )
        else:
            previous_heading = math.radians(last.transform.rotation.yaw)

        best_candidate = None
        best_score = math.inf
        for candidate in candidates:
            location = candidate.transform.location
            last_location = last.transform.location

            heading = math.atan2(
                location.y - last_location.y,
                location.x - last_location.x,
            )

            heading_change = abs(normalize_angle(heading - previous_heading))

            lane_penalty = 0.0

            # Kavşakta hedef rota bilinmediği için geometrik olarak en
            # yumuşak devam seçilir.
            if not last.is_junction:
                if candidate.road_id != last.road_id:
                    lane_penalty += 0.20

                if candidate.lane_id != last.lane_id:
                    lane_penalty += 0.35

            score = heading_change + lane_penalty
            if score < best_score:
                best_candidate = candidate
                best_score = score

        return best_candidate

    def current_waypoint(self, vehicle_location):
        if not self.waypoints:
            return None

        search_count = min(
            20,
            len(self.waypoints),
        )

        nearest_waypoint = None
        nearest_distance = math.inf
        for waypoint in self.waypoints[:search_count]:
            distance = self.distance_between(
                vehicle_location,
                waypoint.transform.location,
            )
            if distance < nearest_distance:
                nearest_waypoint = waypoint
                nearest_distance = distance
        return nearest_waypoint

    def reference_locations(self):
        locations = []
        for waypoint in self.waypoints:
            locations.append(waypoint.transform.location)
        return locations

    def distance_to_route(self, vehicle_location):
        if not self.waypoints:
            return math.inf

        search_count = min(
            30,
            len(self.waypoints),
        )

        nearest_distance = math.inf
        for waypoint in self.waypoints[:search_count]:
            distance = self.distance_between(
                vehicle_location,
                waypoint.transform.location,
            )
            if distance < nearest_distance:
                nearest_distance = distance
        return nearest_distance

    def distance_between(self, first, second):
        return math.hypot(
            first.x - second.x,
            first.y - second.y,
        )

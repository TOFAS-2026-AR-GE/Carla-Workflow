import math

import carla


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class PersistentRouteManager:
    """
    Araci her tick en yakin seride tekrar yapistirmak yerine,
    tek ve devamli bir referans rota tutar.

    Arac kisa sureligine serit cizgisini gecse bile rota degismez.
    Yalnizca rota gercekten uzun sure kaybedilirse yeniden bulunur.
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

        self._waypoints = []
        self._lost_ticks = 0

    def update(self, vehicle_location):
        if not self._waypoints:
            self._initialize(vehicle_location)

        self._trim_passed_waypoints(vehicle_location)
        self._extend_route()

        deviation = self.distance_to_route(vehicle_location)

        if deviation > self.recovery_distance_m:
            self._lost_ticks += 1
        else:
            self._lost_ticks = 0

        # Arac komsu seride kisa sureligine gecince rota degistirme.
        # Yalnizca eski rota uzun sure gercekten kaybedilirse tekrar bul.
        if self._lost_ticks >= self.recovery_ticks:
            self._initialize(vehicle_location)
            self._lost_ticks = 0

        self._trim_passed_waypoints(vehicle_location)
        self._extend_route()

        return (
            self.current_waypoint(vehicle_location),
            self.reference_locations(),
        )

    def _initialize(self, vehicle_location):
        waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if waypoint is None:
            raise RuntimeError("Arac icin surus seridi bulunamadi.")

        self._waypoints = [waypoint]
        self._extend_route()

    def _trim_passed_waypoints(self, vehicle_location):
        if len(self._waypoints) < 2:
            return

        # Araca en yakin bolgeyi bul.
        # Projeksyonun kararlı kalmasi icin arkada iki waypoint birak.
        search_count = min(25, len(self._waypoints))

        nearest_index = min(
            range(search_count),
            key=lambda index: self._distance(
                vehicle_location,
                self._waypoints[index].transform.location,
            ),
        )

        if nearest_index > 2:
            del self._waypoints[: nearest_index - 2]

        # Aracin tamamen gerisinde kalmis waypoint'leri temizle.
        while len(self._waypoints) >= 3:
            first = self._waypoints[0].transform.location
            second = self._waypoints[1].transform.location

            segment_x = second.x - first.x
            segment_y = second.y - first.y
            segment_length = math.hypot(
                segment_x,
                segment_y,
            )

            if segment_length < 1e-4:
                self._waypoints.pop(0)
                continue

            relative_x = vehicle_location.x - first.x
            relative_y = vehicle_location.y - first.y

            along_track = (
                relative_x * segment_x + relative_y * segment_y
            ) / segment_length

            if along_track <= segment_length:
                break

            self._waypoints.pop(0)

    def _extend_route(self):
        while len(self._waypoints) < self.required_points:
            last = self._waypoints[-1]
            candidates = list(last.next(self.spacing_m))

            if not candidates:
                break

            next_waypoint = self._choose_continuous_candidate(
                last,
                candidates,
            )
            self._waypoints.append(next_waypoint)

    def _choose_continuous_candidate(
        self,
        last,
        candidates,
    ):
        if len(self._waypoints) >= 2:
            previous_location = self._waypoints[-2].transform.location
            last_location = last.transform.location

            previous_heading = math.atan2(
                last_location.y - previous_location.y,
                last_location.x - previous_location.x,
            )
        else:
            previous_heading = math.radians(last.transform.rotation.yaw)

        def candidate_score(candidate):
            location = candidate.transform.location
            last_location = last.transform.location

            heading = math.atan2(
                location.y - last_location.y,
                location.x - last_location.x,
            )

            heading_change = abs(normalize_angle(heading - previous_heading))

            lane_penalty = 0.0

            # Kavsakta hedef rota henuz bulunmadigi icin
            # geometrik olarak en yumusak devam secilir.
            if not last.is_junction:
                if candidate.road_id != last.road_id:
                    lane_penalty += 0.20

                if candidate.lane_id != last.lane_id:
                    lane_penalty += 0.35

            return heading_change + lane_penalty

        return min(
            candidates,
            key=candidate_score,
        )

    def current_waypoint(self, vehicle_location):
        if not self._waypoints:
            return None

        search_count = min(
            20,
            len(self._waypoints),
        )

        return min(
            self._waypoints[:search_count],
            key=lambda waypoint: self._distance(
                vehicle_location,
                waypoint.transform.location,
            ),
        )

    def reference_locations(self):
        return [waypoint.transform.location for waypoint in self._waypoints]

    def distance_to_route(self, vehicle_location):
        if not self._waypoints:
            return math.inf

        search_count = min(
            30,
            len(self._waypoints),
        )

        return min(
            self._distance(
                vehicle_location,
                waypoint.transform.location,
            )
            for waypoint in self._waypoints[:search_count]
        )

    @staticmethod
    def _distance(first, second):
        return math.hypot(
            first.x - second.x,
            first.y - second.y,
        )

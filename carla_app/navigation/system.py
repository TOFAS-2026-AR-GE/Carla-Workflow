"""Harita hedefi, rota onayı, canlı ilerleme ve varışı yönetir."""

import math

import carla

from carla_app.navigation.route_planner import RoutePlanner


class NavigationSystem:
    """Kullanıcının haritadan seçtiği hedefi güvenli sürüş durumuna çevirir."""

    def __init__(
        self,
        carla_map,
        route_manager,
        cruise_speed_kmh=50.0,
        arrival_distance_m=2.5,
    ):
        self.map = carla_map
        self.route_manager = route_manager
        self.planner = RoutePlanner(carla_map)
        self.cruise_speed_mps = max(1.0, float(cruise_speed_kmh) / 3.6)
        self.arrival_distance_m = max(1.0, float(arrival_distance_m))

        self.pending_destination = None
        self.destination = None
        self.start_location = None
        self.full_route = ()
        self.status = "WAITING"
        self.message = "Haritada sol tık ile hedef seçin"
        self.remaining_distance_m = None
        self.last_error = None

    def select_destination(self, world_x, world_y):
        """Harita pikselinden gelen dünya noktasını sürüş şeridine taşır."""
        clicked = carla.Location(x=float(world_x), y=float(world_y), z=0.0)
        waypoint = self.map.get_waypoint(
            clicked,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            self.last_error = "Bu noktaya yakın sürüş yolu bulunamadı."
            self.message = self.last_error
            return False

        self.pending_destination = waypoint.transform.location
        self.status = "PENDING"
        self.message = "Hedef hazır; ONAYLA düğmesine basın"
        self.last_error = None
        return True

    def confirm_destination(self, vehicle_location):
        """Bekleyen hedef için en kısa rotayı hesaplar ve etkinleştirir."""
        if self.pending_destination is None:
            return False

        try:
            waypoints = self.planner.trace_route(
                vehicle_location,
                self.pending_destination,
            )
        except Exception as error:
            self.last_error = str(error)
            self.status = "ERROR"
            self.message = f"Rota oluşturulamadı: {error}"
            return False

        if len(waypoints) < 2:
            self.last_error = "Rota yeterli yol noktası içermiyor."
            self.status = "ERROR"
            self.message = self.last_error
            return False

        self.destination = self.pending_destination
        self.pending_destination = None
        self.full_route = tuple(
            waypoint.transform.location for waypoint in waypoints
        )
        self.route_manager.set_planned_route(waypoints)
        self.status = "DRIVING"
        self.message = "Rota aktif"
        self.last_error = None
        return True

    def cancel_pending(self):
        """Onaylanmamış hedef işaretini kaldırır."""
        self.pending_destination = None
        if self.status in {"PENDING", "ERROR"}:
            self.status = "WAITING"
            self.message = "Haritada sol tık ile hedef seçin"
            self.last_error = None

    def update(self, vehicle_location, speed_mps):
        """Canlı konumu işler ve kontrolcüye sade navigasyon bilgisi verir."""
        if self.start_location is None:
            self.start_location = carla.Location(
                x=float(vehicle_location.x),
                y=float(vehicle_location.y),
                z=float(vehicle_location.z),
            )

        if self.status == "DRIVING":
            self.remaining_distance_m = self.route_manager.remaining_distance(
                vehicle_location
            )
            if self.remaining_distance_m <= self.arrival_distance_m:
                self.status = "ARRIVED"
                self.message = "Hedefe varıldı"

        drive_enabled = self.status == "DRIVING"
        target_speed = 0.0
        if drive_enabled:
            stopping_distance = max(
                0.0,
                float(self.remaining_distance_m) - self.arrival_distance_m,
            )
            braking_speed = math.sqrt(2.0 * 2.0 * stopping_distance)
            target_speed = min(self.cruise_speed_mps, braking_speed)

        return {
            "status": self.status,
            "message": self.message,
            "drive_enabled": drive_enabled,
            "target_speed_mps": target_speed,
            "remaining_distance_m": self.remaining_distance_m,
            "speed_mps": float(speed_mps),
            "destination": self.destination,
            "start_location": self.start_location,
            "pending_destination": self.pending_destination,
            "route": self.full_route,
            "last_error": self.last_error,
        }

    def snapshot(self):
        """Görüntüleme için değiştirilmeyecek navigasyon özeti döndürür."""
        return {
            "status": self.status,
            "message": self.message,
            "remaining_distance_m": self.remaining_distance_m,
            "destination": self.destination,
            "start_location": self.start_location,
            "pending_destination": self.pending_destination,
            "route": self.full_route,
            "last_error": self.last_error,
        }

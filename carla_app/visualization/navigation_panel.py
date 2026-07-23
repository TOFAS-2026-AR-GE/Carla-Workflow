"""Navigasyon panelinin çizimini ve fare etkileşimini yönetir."""

from carla_app.visualization.map_renderer import MapRenderer


class NavigationPanel:
    """Harita görünümünü NavigationSystem'dan bağımsız bir UI katmanında tutar."""

    def __init__(
        self,
        carla_map,
        navigation,
        width,
        height,
        render_every_n_frames=2,
        renderer=None,
    ):
        self.navigation = navigation
        self.renderer = renderer or MapRenderer(
            carla_map,
            width=width,
            height=height,
        )
        self.render_every_n_frames = max(1, int(render_every_n_frames))
        self.cached_image = None
        self.cached_navigation_key = None

    def render(
        self,
        navigation_state,
        vehicle_state,
        current_frame_id=None,
    ):
        """Paneli gerektiğinde yeniler, ara karelerde son görüntüyü kullanır."""
        navigation_state = navigation_state or {}
        vehicle_state = vehicle_state or {}
        navigation_key = self._navigation_key(navigation_state)
        frame_number = int(current_frame_id or 0)
        render_due = (
            self.cached_image is None
            or navigation_key != self.cached_navigation_key
            or frame_number % self.render_every_n_frames == 0
        )
        if render_due:
            self.cached_image = self.renderer.render(
                navigation_state,
                vehicle_state.get("location"),
                vehicle_state.get("yaw", 0.0),
                vehicle_state.get("speed_kmh", 0.0),
            )
            self.cached_navigation_key = navigation_key
        return self.cached_image

    def handle_left_click(self, x, y, vehicle_location):
        """Sol tıklamayı onay, iptal veya yeni hedef seçimi olarak işler."""
        if self.renderer.is_confirm_button(x, y):
            if vehicle_location is None:
                return "vehicle_location_unavailable"
            confirmed = self.navigation.confirm_destination(vehicle_location)
            return "confirmed" if confirmed else "confirm_failed"

        if self.renderer.is_cancel_button(x, y):
            self.navigation.cancel_pending()
            return "cancelled"

        world_point = self.renderer.screen_to_world(x, y)
        if world_point is None:
            return None
        selected = self.navigation.select_destination(*world_point)
        return "selected" if selected else "selection_failed"

    @staticmethod
    def _navigation_key(navigation):
        pending = navigation.get("pending_destination")
        pending_key = None
        if pending is not None:
            pending_key = (
                round(float(pending.x), 2),
                round(float(pending.y), 2),
            )
        return (
            navigation.get("status"),
            pending_key,
            id(navigation.get("route")),
        )

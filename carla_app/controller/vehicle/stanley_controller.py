"""Eski içe aktarmalar için Pure Pursuit uyumluluk dosyası."""

from carla_app.controller.vehicle.pure_pursuit_controller import (
    PurePursuitController,
)

# Eski test veya kod bu adı içe aktarırsa yine Pure Pursuit çalışır.
StanleyController = PurePursuitController

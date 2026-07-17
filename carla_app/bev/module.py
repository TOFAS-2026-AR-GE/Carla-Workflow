"""BEV projektörü ile görüntüleyicisini tek ve çıkarılabilir modülde toplar."""

from carla_app.bev.projector import BevProjector
from carla_app.bev.renderer import BevRenderer


class BevModule:
    """Sensör snapshot'ından kuş bakışı BGR görüntüsü üretir."""

    def __init__(self, layout, width=800, height=600):
        self.projector = BevProjector(layout)
        self.renderer = BevRenderer(width=width, height=height)

    def render(
        self,
        sensor_snapshot,
        perception_result,
        vehicle_state,
        current_frame_id=None,
    ):
        scene = self.projector.build_scene(
            sensor_snapshot,
            perception_result,
            vehicle_state,
        )
        return self.renderer.render(scene, current_frame_id)

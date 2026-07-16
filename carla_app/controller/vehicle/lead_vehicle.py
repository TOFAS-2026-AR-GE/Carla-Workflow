import math

from carla_app.controller.vehicle.tracking import (
    Tracker,
    polar_to_world,
)
from carla_app.perception.fusion import (
    fuse_detections_with_radar,
)


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class LeadVehicleTracker:
    """Aynı şeritteki öndeki aracı seçer."""

    def __init__(
        self,
        dt,
        image_width,
        camera_fov_deg,
    ):
        self.dt = dt
        self.image_width = image_width
        self.camera_fov_deg = camera_fov_deg

        # YOLO sonucu 0.5 saniyeden eskiyse
        # kontrol için kullanılmaz.
        self.max_perception_age_frames = max(
            1,
            int(round(0.50 / dt)),
        )

        # Repodaki mevcut Kalman tracker.
        self.tracker = Tracker(
            gate_distance_m=6.0,
            max_misses=max(
                6,
                int(round(0.75 / dt)),
            ),
        )

        # Radar hızındaki küçük titreşimleri azaltır.
        self.filtered_relative_speed = {}

    def update(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        measurements = self._build_measurements(
            current_frame_id,
            state,
            perception_result,
            radar_frame_id,
            radar_points,
        )

        tracks = self.tracker.step(
            self.dt,
            measurements,
        )

        return self._select_lead_vehicle(
            tracks,
            state,
        )

    def _build_measurements(
        self,
        current_frame_id,
        state,
        perception_result,
        radar_frame_id,
        radar_points,
    ):
        if perception_result is None:
            return []

        if (
            radar_frame_id is None
            or not radar_points
        ):
            return []

        detection_frame_id = (
            perception_result.get("frame_id")
        )

        if detection_frame_id is None:
            return []

        age_frames = (
            current_frame_id
            - detection_frame_id
        )

        if age_frames < 0:
            return []

        if (
            age_frames
            > self.max_perception_age_frames
        ):
            return []

        fused = fuse_detections_with_radar(
            detections=(
                perception_result["vehicles"]
            ),
            detection_frame_id=(
                detection_frame_id
            ),
            radar_points=radar_points,
            radar_frame_id=radar_frame_id,
            image_width=self.image_width,
            camera_fov_deg=(
                self.camera_fov_deg
            ),
            fixed_delta_seconds=self.dt,
        )

        ego_location = state["location"]
        measurements = []

        for item in fused:
            if not item.get(
                "has_range",
                False,
            ):
                continue

            # Kontrol için en güncel radar mesafesini
            # kullanıyoruz.
            raw_range = item.get(
                "raw_range_m"
            )

            bearing = item.get(
                "bearing_deg"
            )

            if (
                raw_range is None
                or bearing is None
            ):
                continue

            if (
                raw_range <= 0.5
                or raw_range > 100.0
            ):
                continue

            # Front long radar yaklaşık ±15 derece.
            if abs(bearing) > 16.0:
                continue

            world_x, world_y = (
                polar_to_world(
                    range_m=raw_range,
                    bearing_deg=bearing,
                    ego_x=ego_location.x,
                    ego_y=ego_location.y,
                    ego_yaw_deg=state["yaw"],
                )
            )

            measurements.append(
                {
                    "x": world_x,
                    "y": world_y,
                    "class_name": (
                        item["class_name"]
                    ),
                    "range_m": raw_range,
                    "bearing_deg": bearing,
                    "relative_velocity_mps": (
                        item.get(
                            "relative_velocity_mps"
                        )
                    ),
                }
            )

        return measurements

    def _select_lead_vehicle(
        self,
        tracks,
        state,
    ):
        location = state["location"]
        yaw = math.radians(state["yaw"])

        lane_half_width = max(
            1.50,
            0.55 * float(
                state["lane_width"]
            ),
        )

        best = None

        for track in tracks:
            if not track.confirmed:
                continue

            forward, lateral = (
                self._world_to_ego(
                    track.x,
                    track.y,
                    location.x,
                    location.y,
                    yaw,
                )
            )

            if (
                forward <= 0.5
                or forward > 100.0
            ):
                continue

            # Virajlarda koridor biraz genişler.
            allowed_lateral = (
                lane_half_width
                + 0.03 * forward
            )

            if (
                abs(lateral)
                > allowed_lateral
            ):
                continue

            relative_speed = (
                self._relative_speed(
                    track,
                    state,
                    yaw,
                )
            )

            lead_speed = max(
                0.0,
                state["speed_mps"]
                + relative_speed,
            )

            candidate = {
                "track_id": track.id,
                "class_name": (
                    track.class_name
                ),
                "distance_m": float(
                    forward
                ),
                "lateral_m": float(
                    lateral
                ),
                "lead_speed_mps": float(
                    lead_speed
                ),
                "relative_speed_mps": float(
                    relative_speed
                ),
            }

            if (
                best is None
                or candidate["distance_m"]
                < best["distance_m"]
            ):
                best = candidate

        return best

    def _relative_speed(
        self,
        track,
        state,
        yaw,
    ):
        radar_velocity = (
            track.last_relative_velocity_mps
        )

        if radar_velocity is not None:
            # CARLA radar velocity:
            # Pozitif = nesne sensöre yaklaşıyor.
            #
            # Bizim convention:
            # relative = lead_speed - ego_speed
            #
            # Ego yaklaşıyorsa negatif olmalı.
            measured_relative_speed = (
                -float(radar_velocity)
            )

        else:
            lead_speed = (
                track.vx * math.cos(yaw)
                + track.vy * math.sin(yaw)
            )

            measured_relative_speed = (
                lead_speed
                - state["speed_mps"]
            )

        measured_relative_speed = clamp(
            measured_relative_speed,
            -20.0,
            20.0,
        )

        previous = (
            self.filtered_relative_speed.get(
                track.id,
                measured_relative_speed,
            )
        )

        filtered = (
            0.65 * previous
            + 0.35
            * measured_relative_speed
        )

        self.filtered_relative_speed[
            track.id
        ] = filtered

        return filtered

    @staticmethod
    def _world_to_ego(
        world_x,
        world_y,
        ego_x,
        ego_y,
        ego_yaw,
    ):
        dx = world_x - ego_x
        dy = world_y - ego_y

        forward = (
            dx * math.cos(ego_yaw)
            + dy * math.sin(ego_yaw)
        )

        lateral = (
            -dx * math.sin(ego_yaw)
            + dy * math.cos(ego_yaw)
        )

        return forward, lateral
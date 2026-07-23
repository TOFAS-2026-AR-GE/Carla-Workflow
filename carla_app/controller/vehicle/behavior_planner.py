"""Trafik kurallarını tek, açıklanabilir hedef hız kararında birleştirir."""

import math

from carla_app.config import DrivingParameters


def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


class BehaviorPlanner:
    """Trafik kuralı ve viraj kısıtlarını önceliklendirir."""

    def __init__(self, dt=0.05, parameters=None):
        self.dt = float(dt)
        self.parameters = parameters or DrivingParameters(dt)
        self.previous_target_speed_mps = None
        self.last_light_color = None
        self.yellow_decisions = {}
        self.latched_stop_light = None
        self.simulator_green_hits = 0
        self.green_override_active = False
        self.released_light = None

    def plan(
        self,
        state,
        curve_target_speed_mps,
        speed_plan,
        road_context=None,
        lead_vehicle=None,
    ):
        context = road_context or {}
        ego_speed = max(0.0, float(state.get("speed_mps", 0.0)))
        maximum_speed = min(
            float(curve_target_speed_mps),
            self.parameters.maximum_target_speed_mps,
        )
        constraints = {
            "curve_speed_mps": float(curve_target_speed_mps),
            "system_maximum_mps": self.parameters.maximum_target_speed_mps,
        }
        mode = "CRUISE"
        reason = "normal_cruise"
        confidence = 1.0
        primary_detection = None
        stop_obstacle = None
        brake_urgency = "NONE"

        speed_limit_kmh = context.get("speed_limit_kmh")
        if speed_limit_kmh is not None:
            speed_limit_mps = max(0.0, float(speed_limit_kmh) / 3.6)
            constraints["speed_limit_mps"] = speed_limit_mps
            maximum_speed = min(maximum_speed, speed_limit_mps)

        if speed_plan.get("speed_reason") in {"curve", "lane_recovery"}:
            mode = "SLOW_FOR_CURVE"
            reason = speed_plan.get("speed_reason")

        pedestrian = context.get("pedestrian")
        pedestrian_risk = context.get("pedestrian_risk", "NONE")
        if pedestrian_risk == "SLOW":
            pedestrian_speed = 15.0 / 3.6
            constraints["pedestrian_speed_mps"] = pedestrian_speed
            maximum_speed = min(maximum_speed, pedestrian_speed)
            mode = "SLOW_FOR_PEDESTRIAN"
            reason = "pedestrian_near_driving_corridor"
            primary_detection = pedestrian
            confidence = self.detection_confidence(pedestrian)
        elif pedestrian_risk in {"PREPARE_STOP", "EMERGENCY"}:
            distance = self.detection_distance(pedestrian)
            stop_obstacle = self.make_stop_obstacle(
                distance,
                ego_speed,
                "pedestrian",
                pedestrian,
            )
            maximum_speed = 0.0
            constraints["pedestrian_speed_mps"] = 0.0
            mode = "STOP_FOR_PEDESTRIAN"
            reason = "pedestrian_in_driving_corridor"
            primary_detection = pedestrian
            confidence = self.detection_confidence(pedestrian)
            brake_urgency = "EMERGENCY" if pedestrian_risk == "EMERGENCY" else "NORMAL"

        light = self.resolve_traffic_light(context, state, ego_speed)
        if light is not None and pedestrian_risk not in {"PREPARE_STOP", "EMERGENCY"}:
            light_decision = self.evaluate_traffic_light(light, state, ego_speed)
            if light_decision["applies"]:
                mode = light_decision["mode"]
                reason = light_decision["reason"]
                primary_detection = light
                confidence = self.detection_confidence(light)
                brake_urgency = light_decision["brake_urgency"]
                constraints["traffic_light_speed_mps"] = light_decision[
                    "target_speed_mps"
                ]
                maximum_speed = min(
                    maximum_speed,
                    light_decision["target_speed_mps"],
                )
                stop_obstacle = light_decision["stop_obstacle"]

        perception_age = int(context.get("perception_age_frames", 0) or 0)
        fault_age = int(context.get("sensor_fault_age_frames", 0) or 0)
        sensor_risk_age = max(perception_age, fault_age)
        if context.get("sensor_fault") and sensor_risk_age > (
            self.parameters.sensor_timeout_frames
        ):
            maximum_speed = 0.0
            constraints["sensor_fault_speed_mps"] = 0.0
            stop_distance = self.stopping_distance_m(
                ego_speed,
                self.parameters.comfortable_deceleration_mps2,
            )
            stop_obstacle = self.make_stop_obstacle(
                stop_distance,
                ego_speed,
                "sensor_fault",
                None,
            )
            mode = "SENSOR_DEGRADED"
            reason = "perception_timeout_or_error"
            confidence = 0.0
            brake_urgency = "NORMAL"

        if lead_vehicle is not None and mode in {"CRUISE", "SLOW_FOR_CURVE"}:
            mode = "FOLLOW_VEHICLE"
            reason = "confirmed_lead_vehicle"
            primary_detection = lead_vehicle
            confidence = float(lead_vehicle.get("confidence", 1.0))

        target_speed = self.limit_target_speed(maximum_speed)
        return {
            "mode": mode,
            "target_speed_mps": target_speed,
            "target_stop_distance_m": (
                stop_obstacle.get("distance_m") if stop_obstacle else None
            ),
            "brake_urgency": brake_urgency,
            "reason": reason,
            "primary_detection": primary_detection,
            "confidence": clamp(confidence, 0.0, 1.0),
            "constraints": constraints,
            "control_obstacle": stop_obstacle,
        }

    def evaluate_traffic_light(self, light, state, ego_speed):
        color = str(light.get("color", "unknown"))
        distance = self.detection_distance(light)
        if distance is None:
            return self.no_light_decision()
        stop_distance = max(
            0.25,
            distance - self.parameters.traffic_light_stop_offset_m,
        )
        previous_color = self.last_light_color
        self.last_light_color = color

        if color == "red":
            if not light.get("latched_without_detection", False):
                self.latch_stop_light(light, state)
            target_speed = self.approach_speed_mps(stop_distance)
            stopped_at_line = (
                ego_speed <= self.parameters.traffic_light_stopped_speed_mps
                and stop_distance <= 3.0
            )
            if stopped_at_line:
                target_speed = 0.0
            mode = "STOPPED_AT_RED" if stopped_at_line else "APPROACH_RED_LIGHT"
            urgency = self.stopping_urgency(ego_speed, stop_distance)
            return {
                "applies": True,
                "mode": mode,
                "reason": "confirmed_red_light",
                "target_speed_mps": target_speed,
                "stop_obstacle": self.make_stop_obstacle(
                    stop_distance,
                    ego_speed,
                    "traffic_light_red",
                    light,
                ),
                "brake_urgency": urgency,
            }

        if color == "orange":
            track_id = light.get("track_id", "unknown")
            decision = self.yellow_decisions.get(track_id)
            if decision is None:
                required = self.required_deceleration_mps2(ego_speed, stop_distance)
                can_stop = (
                    not bool(state.get("is_junction", False))
                    and required <= self.parameters.maximum_normal_deceleration_mps2
                )
                decision = "stop" if can_stop else "proceed"
                self.yellow_decisions[track_id] = decision
            if decision == "proceed":
                self.latched_stop_light = None
                return {
                    "applies": True,
                    "mode": "YELLOW_DECISION",
                    "reason": "yellow_proceed_dilemma_zone",
                    "target_speed_mps": self.parameters.maximum_target_speed_mps,
                    "stop_obstacle": None,
                    "brake_urgency": "NONE",
                }
            if not light.get("latched_without_detection", False):
                self.latch_stop_light(light, state)
            return {
                "applies": True,
                "mode": "YELLOW_DECISION",
                "reason": "yellow_safe_stop",
                "target_speed_mps": self.approach_speed_mps(stop_distance),
                "stop_obstacle": self.make_stop_obstacle(
                    stop_distance,
                    ego_speed,
                    "traffic_light_yellow",
                    light,
                ),
                "brake_urgency": self.stopping_urgency(ego_speed, stop_distance),
            }

        if color == "green":
            self.latched_stop_light = None
            for track_id in list(self.yellow_decisions):
                self.yellow_decisions.pop(track_id, None)
            if previous_color in {"red", "orange"} and ego_speed <= 0.50:
                return {
                    "applies": True,
                    "mode": "START_ON_GREEN",
                    "reason": "green_confirmed_clear_to_start",
                    "target_speed_mps": self.parameters.maximum_target_speed_mps,
                    "stop_obstacle": None,
                    "brake_urgency": "NONE",
                }
        return self.no_light_decision()

    def resolve_traffic_light(self, context, state, ego_speed):
        """Kamera ışığını seçer ve CARLA durumunu güvenlik yedeği yapar.

        Kamera kırmızıyı uzaktan bulup aracı yumuşak biçimde durdurur. Kamera
        ışığı kaçırsa bile CARLA bu aracı etkileyen kırmızı/turuncu ışığı
        bildiriyorsa geçişe izin verilmez. Çok yakında lamba görüntünün üstünden
        çıktığında CARLA yeşili de yalnız kilitli ve durmuş araçta doğrulanır.
        """
        camera_light = context.get("lead_traffic_light")
        if camera_light is not None:
            light = camera_light
        else:
            light = self.recall_stop_light(state)

        simulator_light = state.get("simulator_traffic_light", {}) or {}
        simulator_is_reliable = (
            bool(simulator_light.get("available"))
            and bool(simulator_light.get("affected"))
        )
        simulator_color = str(simulator_light.get("color", "unknown"))
        simulator_says_green = (
            simulator_is_reliable
            and simulator_color == "green"
        )
        simulator_stop_light = None
        if simulator_is_reliable and simulator_color in {"red", "orange"}:
            simulator_stop_light = self.make_simulator_stop_light(
                simulator_light,
                ego_speed,
            )

        # CARLA yalnız gerçekten ego aracını etkileyen ışığı bildirir. Bu
        # nedenle etkileyen kırmızı, kameranın eksik/eski yeşilinden daha
        # güvenlidir. İki kaynak da dur diyorsa çizgiye daha yakın mesafe
        # seçilerek ışığı geçme riski azaltılır.
        if simulator_stop_light is not None:
            light_color = str((light or {}).get("color", "unknown"))
            if light is None or light_color not in {"red", "orange"}:
                light = simulator_stop_light
            else:
                camera_distance = self.detection_distance(light)
                simulator_distance = self.detection_distance(
                    simulator_stop_light
                )
                if (
                    camera_distance is None
                    or (
                        simulator_distance is not None
                        and simulator_distance < camera_distance
                    )
                ):
                    light = simulator_stop_light

        # Yeşil doğrulandıktan sonra kamera takipçisi birkaç kare daha eski
        # kararlı kırmızıyı taşıyabilir. Aynı ışığın bu gecikmiş kırmızısını
        # tekrar kilitlemeyiz; aksi durumda araç kalkıp hemen yeniden durur.
        if self.green_override_active:
            # Yeşil serbest bırakma yalnız o anki CARLA durumu yeşilken
            # geçerlidir. Aynı ışık tekrar kırmızı/turuncu olduğunda eski
            # kamera takibini bastırmaya devam etmek kırmızı ihlaline yol
            # açardı; yetkili simülatör durumu bu kilidi derhal iptal eder.
            if simulator_stop_light is not None:
                self.green_override_active = False
                self.released_light = None
                self.simulator_green_hits = 0
                return simulator_stop_light

            released_track_id = (self.released_light or {}).get("track_id")
            camera_track_id = (camera_light or {}).get("track_id")
            same_released_light = (
                camera_light is not None
                and camera_track_id == released_track_id
            )
            if simulator_says_green:
                return self.make_confirmed_green(light or self.released_light)
            if same_released_light and str(camera_light.get("color")) != "green":
                return None
            self.green_override_active = False
            self.released_light = None

        waiting_at_latched_red = (
            self.latched_stop_light is not None
            and light is not None
            and str(light.get("color")) in {"red", "orange"}
            and ego_speed <= 0.50
        )

        if simulator_says_green and waiting_at_latched_red:
            self.simulator_green_hits += 1
        else:
            self.simulator_green_hits = 0

        required_hits = self.parameters.simulator_green_confirmation_frames
        if self.simulator_green_hits >= required_hits:
            self.green_override_active = True
            self.released_light = dict(light)
            return self.make_confirmed_green(light)

        return light

    def make_simulator_stop_light(self, simulator_light, ego_speed):
        """CARLA kırmızı/turuncusunu planlayıcının ışık biçimine dönüştürür."""
        distance = self.detection_distance(simulator_light)
        if distance is None:
            # Eski CARLA sürümlerinde stop waypoint API'si yoktur. Araç ışığın
            # tetikleme alanındayken, normal azami yavaşlama ile durabileceği
            # mesafe güvenli ve sınırlı bir yedektir.
            stop_distance = self.stopping_distance_m(
                ego_speed,
                self.parameters.maximum_normal_deceleration_mps2,
            )
            distance = (
                self.parameters.traffic_light_stop_offset_m
                + stop_distance
            )

        actor_id = simulator_light.get("actor_id")
        return {
            "color": str(simulator_light.get("color", "unknown")),
            "observed_color": str(simulator_light.get("color", "unknown")),
            "estimated_distance_m": max(0.25, float(distance)),
            "confidence": 1.0,
            "smoothed_confidence": 1.0,
            "track_id": (
                f"carla-light-{actor_id}"
                if actor_id is not None
                else "carla-affecting-light"
            ),
            "state_source": "carla_vehicle_state",
            "latched_without_detection": False,
        }

    def make_confirmed_green(self, light):
        """Kırmızı kaydından yeşil serbest bırakma algılaması üretir."""
        green_light = dict(light or {})
        green_light["color"] = "green"
        green_light["observed_color"] = "green"
        green_light["state_source"] = "carla_vehicle_state"
        green_light["latched_without_detection"] = False
        return green_light

    def latch_stop_light(self, light, state):
        """Dur ışığının dünya üzerindeki yaklaşık yerini saklar."""
        distance = self.detection_distance(light)
        location = state.get("location")
        if distance is None or location is None:
            return
        yaw = math.radians(float(state.get("yaw", 0.0)))
        self.latched_stop_light = {
            "light": dict(light),
            "world_x": float(location.x) + distance * math.cos(yaw),
            "world_y": float(location.y) + distance * math.sin(yaw),
        }

    def recall_stop_light(self, state):
        """Kamera kutuyu kaçırsa da kırmızı durma noktasını korur."""
        if self.latched_stop_light is None:
            return None
        location = state.get("location")
        if location is None:
            return None

        yaw = math.radians(float(state.get("yaw", 0.0)))
        delta_x = self.latched_stop_light["world_x"] - float(location.x)
        delta_y = self.latched_stop_light["world_y"] - float(location.y)
        forward_distance = delta_x * math.cos(yaw) + delta_y * math.sin(yaw)

        # Araç ışık çizgisini tamamen geçtiyse eski ışık artık kontrol etmez.
        if forward_distance < -2.0:
            self.latched_stop_light = None
            return None

        light = dict(self.latched_stop_light["light"])
        light["estimated_distance_m"] = max(0.25, forward_distance)
        light["latched_without_detection"] = True
        light["state_source"] = "latched_camera_red"
        return light

    def no_light_decision(self):
        return {
            "applies": False,
            "mode": "CRUISE",
            "reason": "no_controlling_light",
            "target_speed_mps": self.parameters.maximum_target_speed_mps,
            "stop_obstacle": None,
            "brake_urgency": "NONE",
        }

    def make_stop_obstacle(self, distance, ego_speed, source, detection):
        if distance is None:
            distance = self.stopping_distance_m(
                ego_speed,
                self.parameters.comfortable_deceleration_mps2,
            )
        obstacle = {
            "track_id": detection.get("track_id") if detection else -3,
            "distance_m": max(0.25, float(distance)),
            "relative_speed_mps": -max(0.0, float(ego_speed)),
            "source": source,
            "bearing_deg": 0.0,
            "measurement_frame_id": (
                detection.get("frame_id") if detection else None
            ),
        }
        return obstacle

    def approach_speed_mps(self, stop_distance):
        usable = max(
            0.0,
            float(stop_distance) - self.parameters.stopping_safety_margin_m,
        )
        return math.sqrt(
            2.0 * self.parameters.comfortable_deceleration_mps2 * usable
        )

    def stopping_distance_m(self, speed_mps, deceleration_mps2):
        reaction = speed_mps * self.parameters.reaction_time_s
        braking = speed_mps**2 / (2.0 * max(0.10, deceleration_mps2))
        return reaction + braking + self.parameters.stopping_safety_margin_m

    def required_deceleration_mps2(self, speed_mps, distance_m):
        reaction_distance = speed_mps * self.parameters.reaction_time_s
        usable = max(
            0.10,
            distance_m - reaction_distance - self.parameters.stopping_safety_margin_m,
        )
        return speed_mps**2 / (2.0 * usable)

    def stopping_urgency(self, speed_mps, distance_m):
        required = self.required_deceleration_mps2(speed_mps, distance_m)
        if required >= self.parameters.emergency_deceleration_mps2:
            return "EMERGENCY"
        if required >= self.parameters.maximum_normal_deceleration_mps2:
            return "HIGH"
        return "NORMAL"

    def limit_target_speed(self, desired_speed):
        desired_speed = clamp(
            float(desired_speed),
            0.0,
            self.parameters.maximum_target_speed_mps,
        )
        if self.previous_target_speed_mps is None:
            self.previous_target_speed_mps = desired_speed
            return desired_speed
        if desired_speed <= self.previous_target_speed_mps:
            self.previous_target_speed_mps = desired_speed
            return desired_speed
        maximum_change = self.parameters.green_start_acceleration_mps2 * self.dt
        self.previous_target_speed_mps += min(
            maximum_change,
            desired_speed - self.previous_target_speed_mps,
        )
        return self.previous_target_speed_mps

    def detection_distance(self, detection):
        if not detection:
            return None
        value = detection.get("estimated_distance_m", detection.get("distance_m"))
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) and value > 0.0 else None

    def detection_confidence(self, detection):
        if not detection:
            return 0.0
        return float(
            detection.get("smoothed_confidence", detection.get("confidence", 0.0))
        )

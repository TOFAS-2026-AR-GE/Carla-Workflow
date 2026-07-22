from carla_app.controller.vehicle.traffic_light import TrafficLightObserver


def observer():
    return TrafficLightObserver(dt=0.05, image_width=800, image_height=600)


def test_class_name_normalization():
    item = observer()
    assert item.class_name_to_state("traffic_light_red") == "red"
    assert item.class_name_to_state("traffic-light-orange") == "yellow"
    assert item.class_name_to_state("traffic_light_green") == "green"
    assert item.class_name_to_state("speed_limit_30") == "unknown"


def test_relevant_detection_prefers_central_confident_light():
    item = observer()
    result = {
        "traffic_lights": [
            {"bbox": (5, 20, 25, 80), "confidence": 0.99, "class_name": "traffic_light_red"},
            {"bbox": (360, 30, 410, 130), "confidence": 0.80, "class_name": "traffic_light_green"},
        ]
    }
    selected = item.select_relevant_detection(result)
    assert selected["class_name"] == "traffic_light_green"


def test_two_votes_confirm_state():
    item = observer()
    item._add_vote("red")
    assert item.confirmed_state == "unknown"
    item._add_vote("red")
    assert item.confirmed_state == "red"


def test_yellow_dilemma_zone_continues_when_too_close():
    item = observer()
    assert item.requires_stop("yellow", distance_m=30.0, speed_mps=10.0) is True
    assert item.requires_stop("yellow", distance_m=5.0, speed_mps=10.0) is False
    assert item.requires_stop("red", distance_m=5.0, speed_mps=10.0) is True

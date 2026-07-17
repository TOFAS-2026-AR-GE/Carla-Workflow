"""Farklı sensörlerin aynı fiziksel nesneye ait ölçümlerini eşleştirir."""

import math


def measurement_distance(first, second):
    return math.hypot(
        float(first["x_m"]) - float(second["x_m"]),
        float(first["y_m"]) - float(second["y_m"]),
    )


def association_gate(first, second):
    """Mesafe arttıkça kamera izdüşüm belirsizliğine küçük pay bırakır."""
    first_range = math.hypot(float(first["x_m"]), float(first["y_m"]))
    second_range = math.hypot(float(second["x_m"]), float(second["y_m"]))
    average_range = 0.5 * (first_range + second_range)

    base_gate = 1.8 + 0.025 * average_range
    uncertainty = float(first.get("uncertainty_m", 1.0)) + float(
        second.get("uncertainty_m", 1.0)
    )
    return min(4.0, max(base_gate, 1.2 * uncertainty))


def group_associated_measurements(measurements):
    """Yakın ölçümleri gruplar; bir sensör bir grupta yalnızca bir kez olur."""
    groups = []
    for measurement in measurements:
        measurement_sensors = set(measurement.get("sensor_names", []))
        best_group = None
        best_distance = None

        for group in groups:
            group_sensors = set()
            for grouped_measurement in group:
                group_sensors.update(
                    grouped_measurement.get("sensor_names", [])
                )

            # Aynı kameranın iki ayrı bbox'ı dolaylı bir radar bağlantısıyla
            # bile tek nesneye dönüşmemelidir.
            if measurement_sensors.intersection(group_sensors):
                continue

            closest_distance = None
            for grouped_measurement in group:
                distance = measurement_distance(
                    measurement,
                    grouped_measurement,
                )
                gate = association_gate(measurement, grouped_measurement)
                if distance > gate:
                    continue
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance

            if closest_distance is None:
                continue
            if best_distance is None or closest_distance < best_distance:
                best_distance = closest_distance
                best_group = group

        if best_group is None:
            groups.append([measurement])
        else:
            best_group.append(measurement)
    return groups

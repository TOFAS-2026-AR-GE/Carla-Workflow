"""Farklı sensörlerin aynı fiziksel nesneye ait ölçümlerini eşleştirir."""

import math

import numpy as np


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


def measurement_covariance(measurement):
    """Ölçümün 2B covariance matrisini güvenli ve pozitif tanımlı döndürür."""
    covariance = measurement.get("covariance_xy")
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=np.float64)
        if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)):
            covariance = 0.5 * (covariance + covariance.T)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if eigenvalues[0] > 1e-6:
                return covariance

    uncertainty = max(0.10, float(measurement.get("uncertainty_m", 1.0)))
    return np.eye(2, dtype=np.float64) * uncertainty**2


def association_cost(first, second):
    """Birleşik belirsizlik altında karesel Mahalanobis uzaklığı üretir."""
    delta = np.array(
        [
            float(first["x_m"]) - float(second["x_m"]),
            float(first["y_m"]) - float(second["y_m"]),
        ],
        dtype=np.float64,
    )
    covariance = measurement_covariance(first) + measurement_covariance(second)
    covariance += np.eye(2, dtype=np.float64) * 0.04
    return float(delta.T @ np.linalg.solve(covariance, delta))


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

            closest_cost = None
            for grouped_measurement in group:
                distance = measurement_distance(
                    measurement,
                    grouped_measurement,
                )
                gate = association_gate(measurement, grouped_measurement)
                if distance > gate:
                    continue
                cost = association_cost(measurement, grouped_measurement)
                # İki serbestlik derecesinde %99 chi-square kapısı.
                if cost > 9.21:
                    continue
                if closest_cost is None or cost < closest_cost:
                    closest_cost = cost

            if closest_cost is None:
                continue
            if best_distance is None or closest_cost < best_distance:
                best_distance = closest_cost
                best_group = group

        if best_group is None:
            groups.append([measurement])
        else:
            best_group.append(measurement)
    return groups

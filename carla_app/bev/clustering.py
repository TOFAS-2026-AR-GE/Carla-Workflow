"""Radar ve LiDAR noktalarını ek bağımlılık olmadan komşulukla kümeler."""

import math

import numpy as np


def point_cell(x_m, y_m, cell_size_m):
    return (
        int(math.floor(float(x_m) / cell_size_m)),
        int(math.floor(float(y_m) / cell_size_m)),
    )


def cluster_xy_points(points, cell_size_m, minimum_points):
    """Komşu 2B hücrelerdeki nokta indekslerini aynı kümede toplar."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 2 or len(points) == 0:
        return []

    cells = {}
    for index, point in enumerate(points):
        cell = point_cell(point[0], point[1], cell_size_m)
        if cell not in cells:
            cells[cell] = []
        cells[cell].append(index)

    clusters = []
    visited = set()
    for start_cell in cells:
        if start_cell in visited:
            continue

        pending = [start_cell]
        visited.add(start_cell)
        indices = []

        while pending:
            cell_x, cell_y = pending.pop()
            indices.extend(cells.get((cell_x, cell_y), []))

            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    neighbor = (cell_x + offset_x, cell_y + offset_y)
                    if neighbor in visited or neighbor not in cells:
                        continue
                    visited.add(neighbor)
                    pending.append(neighbor)

        if len(indices) >= int(minimum_points):
            clusters.append(indices)

    return clusters


def cluster_summary(points, indices):
    """Bir nokta kümesinin merkezini, boyutunu ve eleman sayısını verir."""
    cluster = np.asarray(points)[indices]
    minimum = np.min(cluster[:, :2], axis=0)
    maximum = np.max(cluster[:, :2], axis=0)
    center = np.median(cluster[:, :2], axis=0)
    return {
        "x_m": float(center[0]),
        "y_m": float(center[1]),
        "length_m": float(maximum[0] - minimum[0]),
        "width_m": float(maximum[1] - minimum[1]),
        "point_count": len(indices),
    }

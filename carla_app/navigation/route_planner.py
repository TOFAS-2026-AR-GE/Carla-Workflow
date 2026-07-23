"""CARLA yol ağı üzerinde başlangıçtan hedefe sürüş rotası üretir."""

import heapq
import math


def _distance(first, second):
    return math.hypot(first.x - second.x, first.y - second.y)


class RoutePlanner:
    """Önce CARLA'nın resmi planlayıcısını, gerekirse sade A* kullanır."""

    def __init__(self, carla_map, sampling_resolution=2.0):
        self.map = carla_map
        self.resolution = max(1.0, float(sampling_resolution))
        self.official_planner = self._create_official_planner()
        self.graph = None
        self.positions = None

    def _create_official_planner(self):
        try:
            from agents.navigation.global_route_planner import (
                GlobalRoutePlanner,
            )
        except ImportError:
            return None

        return GlobalRoutePlanner(
            self.map,
            sampling_resolution=self.resolution,
        )

    def trace_route(self, start_location, destination_location):
        """İki konum arasında sıralı CARLA waypoint listesi döndürür."""
        if self.official_planner is not None:
            traced = self.official_planner.trace_route(
                start_location,
                destination_location,
            )
            waypoints = [waypoint for waypoint, _road_option in traced]
            if len(waypoints) >= 2:
                return waypoints

        return self._trace_with_astar(start_location, destination_location)

    def _trace_with_astar(self, start_location, destination_location):
        if self.graph is None:
            self._build_graph()

        start_waypoint = self.map.get_waypoint(
            start_location,
            project_to_road=True,
        )
        destination_waypoint = self.map.get_waypoint(
            destination_location,
            project_to_road=True,
        )
        if start_waypoint is None or destination_waypoint is None:
            raise RuntimeError("Başlangıç veya hedef sürüş yoluna taşınamadı.")

        start_node = self._nearest_node(start_waypoint.transform.location)
        destination_node = self._nearest_node(
            destination_waypoint.transform.location
        )
        node_path = self._astar(start_node, destination_node)
        if not node_path:
            raise RuntimeError("Seçilen hedefe giden sürüş rotası bulunamadı.")

        route = [start_waypoint]
        for first, second in zip(node_path, node_path[1:]):
            edge = self.graph[first][second]
            route.extend(edge["waypoints"])
        route.append(destination_waypoint)
        return self._remove_duplicates(route)

    def _build_graph(self):
        self.graph = {}
        self.positions = {}

        for entry, exit_waypoint in self.map.get_topology():
            first = self._node_key(entry.transform.location)
            second = self._node_key(exit_waypoint.transform.location)
            self.positions[first] = entry.transform.location
            self.positions[second] = exit_waypoint.transform.location

            points = self._sample_segment(entry, exit_waypoint)
            weight = self._path_length(points)
            self.graph.setdefault(first, {})[second] = {
                "weight": max(weight, self.resolution),
                "waypoints": points,
            }
            self.graph.setdefault(second, {})

    def _sample_segment(self, entry, exit_waypoint):
        points = [entry]
        current = entry
        maximum_steps = 2000

        for _ in range(maximum_steps):
            if _distance(
                current.transform.location,
                exit_waypoint.transform.location,
            ) <= self.resolution * 1.25:
                break

            candidates = list(current.next(self.resolution))
            if not candidates:
                break

            same_lane = [
                candidate
                for candidate in candidates
                if candidate.road_id == exit_waypoint.road_id
                and candidate.lane_id == exit_waypoint.lane_id
            ]
            pool = same_lane or candidates
            current = min(
                pool,
                key=lambda item: _distance(
                    item.transform.location,
                    exit_waypoint.transform.location,
                ),
            )
            points.append(current)

        points.append(exit_waypoint)
        return self._remove_duplicates(points)

    def _astar(self, start, destination):
        queue = [(0.0, start)]
        previous = {}
        costs = {start: 0.0}

        while queue:
            _priority, current = heapq.heappop(queue)
            if current == destination:
                break

            for neighbour, edge in self.graph.get(current, {}).items():
                new_cost = costs[current] + edge["weight"]
                if new_cost >= costs.get(neighbour, math.inf):
                    continue

                costs[neighbour] = new_cost
                previous[neighbour] = current
                heuristic = _distance(
                    self.positions[neighbour],
                    self.positions[destination],
                )
                heapq.heappush(queue, (new_cost + heuristic, neighbour))

        if destination not in costs:
            return []

        path = [destination]
        while path[-1] != start:
            path.append(previous[path[-1]])
        path.reverse()
        return path

    def _nearest_node(self, location):
        return min(
            self.positions,
            key=lambda node: _distance(location, self.positions[node]),
        )

    def _node_key(self, location):
        precision = max(0.5, self.resolution * 0.25)
        return (
            int(round(location.x / precision)),
            int(round(location.y / precision)),
            int(round(location.z / 2.0)),
        )

    @staticmethod
    def _path_length(waypoints):
        total = 0.0
        for first, second in zip(waypoints, waypoints[1:]):
            total += _distance(
                first.transform.location,
                second.transform.location,
            )
        return total

    @staticmethod
    def _remove_duplicates(waypoints):
        result = []
        for waypoint in waypoints:
            if result:
                distance = _distance(
                    result[-1].transform.location,
                    waypoint.transform.location,
                )
                if distance < 0.20:
                    continue
            result.append(waypoint)
        return result

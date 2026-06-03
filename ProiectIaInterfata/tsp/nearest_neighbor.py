import numpy as np
from .base import TSPSolver

class NearestNeighborSolver(TSPSolver):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix, "NearestNeighbor")

    def solve(self, start=0, multistart=False):
        best_route, best_cost = None, float('inf')
        starts = range(self.n) if multistart else [start]
        for s in starts:
            route, cost = self._nn_from_start(s)
            if cost < best_cost:
                best_cost = cost
                best_route = route
        stats = {'multistart': multistart, 'evaluated_starts': len(starts)}
        return best_route, best_cost, stats

    def _nn_from_start(self, start):
        n = self.n
        visited = [False] * n
        route = [start]
        visited[start] = True
        current = start
        total_cost = 0
        for _ in range(n - 1):
            min_dist = float('inf')
            nxt = None
            for j in range(n):
                if not visited[j] and self.distance_matrix[current, j] < min_dist:
                    min_dist = self.distance_matrix[current, j]
                    nxt = j
            route.append(nxt)
            visited[nxt] = True
            total_cost += min_dist
            current = nxt
        total_cost += self.distance_matrix[current, start]
        return route, total_cost
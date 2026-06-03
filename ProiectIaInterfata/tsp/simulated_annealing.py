import math
import random
import time
from .base import TSPSolver
from .nearest_neighbor import NearestNeighborSolver

class CustomSimulatedAnnealing:
    def __init__(self, distance_matrix, T_max=10000, T_min=1, alpha=0.99, use_nn_start=True):
        self.dist = distance_matrix
        self.n = distance_matrix.shape[0]
        self.T_max = T_max
        self.T_min = T_min
        self.alpha = alpha
        self.use_nn_start = use_nn_start

    def solve(self):
        n = self.n
        if self.use_nn_start:
            nn = NearestNeighborSolver(self.dist)
            current_route, _ = nn._nn_from_start(0)
        else:
            current_route = list(range(n))
            random.shuffle(current_route)
        best_route = current_route[:]
        current_cost = self._cost(current_route)
        best_cost = current_cost
        T = self.T_max
        while T > self.T_min:
            i, j = sorted(random.sample(range(n), 2))
            new_route = current_route[:i] + current_route[i:j + 1][::-1] + current_route[j + 1:]
            new_cost = self._cost(new_route)
            delta = new_cost - current_cost
            if delta <= 0 or random.random() < math.exp(-delta / T):
                current_route = new_route
                current_cost = new_cost
                if current_cost < best_cost:
                    best_route = current_route[:]
                    best_cost = current_cost
            T *= self.alpha
        return best_route, best_cost

    def _cost(self, route):
        return sum(self.dist[route[i], route[(i + 1) % self.n]] for i in range(self.n))

class SimulatedAnnealingSolver(TSPSolver):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix, "SimulatedAnnealing")

    def solve(self, T_max=10000, T_min=1, alpha=0.995, use_nn_start=True, simanneal=False):
        if simanneal:
            from .simanneal_adapter import SimannealAdapter
            solver = SimannealAdapter(self.distance_matrix, Tmax=T_max, Tmin=T_min, steps=50000)
            return solver.solve()
        else:
            sa = CustomSimulatedAnnealing(self.distance_matrix, T_max, T_min, alpha, use_nn_start)
            start = time.perf_counter()
            route, cost = sa.solve()
            elapsed = time.perf_counter() - start
            return route, cost, {'time': elapsed, 'T_max': T_max, 'alpha': alpha}
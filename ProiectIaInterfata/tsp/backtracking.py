import time
import sys
import numpy as np
from .base import TSPSolver

class BacktrackingSolver(TSPSolver):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix, "Backtracking")

    def solve(self, mode='toate', timp_max=60, y_max=10):
        n = self.n
        visited = [False] * n
        visited[0] = True
        best_cost = sys.maxsize
        best_route = []
        sol_count = 0
        stop_flag = False

        start_time = time.perf_counter()

        def backtrack(current, route, cost):
            nonlocal best_cost, best_route, sol_count, stop_flag
            if stop_flag:
                return
            if len(route) == n:
                total_cost = cost + self.distance_matrix[current, route[0]]
                sol_count += 1
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_route = route[:]
                if mode == 'prima':
                    stop_flag = True
                    return
                if mode == 'y_solutii' and sol_count >= y_max:
                    stop_flag = True
                    return
                return
            if mode == 'timp' and time.perf_counter() - start_time >= timp_max:
                stop_flag = True
                return
            for nxt in range(n):
                if not visited[nxt]:
                    new_cost = cost + self.distance_matrix[current, nxt]
                    if mode not in ('toate', 'timp') and new_cost >= best_cost:
                        continue
                    visited[nxt] = True
                    route.append(nxt)
                    backtrack(nxt, route, new_cost)
                    route.pop()
                    visited[nxt] = False

        backtrack(0, [0], 0)
        elapsed = time.perf_counter() - start_time
        stats = {
            'mode': mode,
            'solutions_found': sol_count,
            'time': elapsed,
            'optimal': mode == 'toate'
        }
        return best_route, best_cost, stats
from abc import ABC, abstractmethod
import numpy as np

class TSPSolver(ABC):
    def __init__(self, distance_matrix: np.ndarray, name: str = "Solver"):
        self.distance_matrix = distance_matrix
        self.n = distance_matrix.shape[0]
        self.name = name

    @abstractmethod
    def solve(self, **kwargs):
        
        pass

    def route_cost(self, route):
        d = self.distance_matrix
        return sum(d[route[i], route[(i+1) % self.n]] for i in range(self.n))
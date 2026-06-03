import random
import numpy as np
from simpleai.search import SearchProblem, hill_climbing_random_restarts
from .base import TSPSolver

class TSPHillClimbingProblem(SearchProblem):
    def __init__(self, distance_matrix, initial_state=None):
        self.dist = distance_matrix
        self.n = distance_matrix.shape[0]
        super().__init__(initial_state)

    def actions(self, state):
        for i in range(self.n):
            for j in range(i + 1, self.n):
                yield (i, j)

    def result(self, state, action):
        i, j = action
        new_state = list(state)
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return tuple(new_state)

    def value(self, state):
        return -sum(self.dist[state[i], state[(i + 1) % self.n]] for i in range(self.n))

    def generate_random_state(self):
        state = list(range(self.n))
        random.shuffle(state)
        return tuple(state)

class HillClimbingSolver(TSPSolver):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix, "HillClimbing")

    def solve(self, restarts_limit=100, iterations_limit=100):
        problem = TSPHillClimbingProblem(self.distance_matrix)
        result = hill_climbing_random_restarts(problem, restarts_limit, iterations_limit)
        route = list(result.state)
        cost = -problem.value(result.state)
        stats = {'restarts': restarts_limit, 'iterations_limit': iterations_limit}
        return route, cost, stats
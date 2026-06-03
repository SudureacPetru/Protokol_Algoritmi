import random
import time
from simanneal import Annealer

class TSPAnnealer(Annealer):
    def __init__(self, state, distance_matrix):
        self.distance_matrix = distance_matrix
        super().__init__(state)

    def move(self):
        n = len(self.state)
        i, j = sorted(random.sample(range(n), 2))
        self.state[i:j + 1] = self.state[i:j + 1][::-1]

    def energy(self):
        d = self.distance_matrix
        return sum(d[self.state[i], self.state[(i + 1) % len(self.state)]] for i in range(len(self.state)))

class SimannealAdapter:
    def __init__(self, distance_matrix, Tmax=10000, Tmin=1, steps=50000):
        self.dist = distance_matrix
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.steps = steps

    def solve(self):
        n = self.dist.shape[0]
        init_state = list(range(n))
        random.shuffle(init_state)
        annealer = TSPAnnealer(init_state, self.dist)
        annealer.Tmax = self.Tmax
        annealer.Tmin = self.Tmin
        annealer.steps = self.steps
        start = time.time()
        best_state, best_energy = annealer.anneal()
        elapsed = time.time() - start
        return best_state, best_energy, {'time': elapsed, 'simanneal': True}
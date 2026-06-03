import random
import time
import numpy as np
import pygad
from .base import TSPSolver

class GeneticAlgorithmSolver(TSPSolver):
    def __init__(self, distance_matrix):
        super().__init__(distance_matrix, "GeneticAlgorithm")

    def solve(self, pop_size=100, n_generations=500, mutation_rate=50,
              selection="tournament", k_tournament=3, keep_elitism=2):
        n = self.n
        dist = self.distance_matrix

        def fitness_func(ga_instance, solution, solution_idx):
            solution = solution.astype(int)   # <-- adaugă această linie
            cost = sum(dist[solution[i], solution[(i + 1) % n]] for i in range(n))
            return -cost

        def ox_crossover(parents, offspring_size, ga_instance):
            offspring = []
            idx = 0
            while len(offspring) < offspring_size[0]:
                p1 = parents[idx % len(parents)].astype(int).tolist()
                p2 = parents[(idx + 1) % len(parents)].astype(int).tolist()
                cx1, cx2 = sorted(random.sample(range(n), 2))
                child = [-1] * n
                child[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
                segment_set = set(child[cx1:cx2 + 1])
                remaining = [g for g in p2 if g not in segment_set]
                free_positions = [i for i in range(n) if child[i] == -1]
                for pos, gene in zip(free_positions, remaining):
                    child[pos] = gene
                offspring.append(child)
                idx += 1
            return np.array(offspring, dtype=int)

        def swap_mutation(offspring, ga_instance):
            rate = ga_instance.mutation_percent_genes / 100.0
            for i in range(offspring.shape[0]):
                if random.random() < rate:
                    j1, j2 = random.sample(range(n), 2)
                    offspring[i, j1], offspring[i, j2] = offspring[i, j2], offspring[i, j1]
            return offspring

        initial_pop = [random.sample(range(n), n) for _ in range(pop_size)]
        ga = pygad.GA(
            num_generations=n_generations,
            num_parents_mating=max(2, pop_size // 2),
            fitness_func=fitness_func,
            initial_population=np.array(initial_pop, dtype=int),
            crossover_type=ox_crossover,
            mutation_type=swap_mutation,
            mutation_percent_genes=mutation_rate,
            parent_selection_type=selection,
            K_tournament=k_tournament,
            keep_elitism=keep_elitism,
            keep_parents=0,
            suppress_warnings=True
        )

        start = time.time()
        ga.run()
        elapsed = time.time() - start
        solution, fitness, _ = ga.best_solution()
        cost = -fitness
        return list(map(int, solution)), cost, {
            'time': elapsed,
            'generations': n_generations,
            'pop_size': pop_size,
            'convergence': [-f for f in ga.best_solutions_fitness]
        }
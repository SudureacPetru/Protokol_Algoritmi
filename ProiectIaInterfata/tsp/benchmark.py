import time
import matplotlib.pyplot as plt
import numpy as np
from .utils import generate_symmetric_matrix
from .backtracking import BacktrackingSolver
from .nearest_neighbor import NearestNeighborSolver
from .hill_climbing import HillClimbingSolver
from .simulated_annealing import SimulatedAnnealingSolver
from .genetic_algorithm import GeneticAlgorithmSolver

def run_comparison(sizes=[5, 8, 10, 12], seeds=[42, 123]):
    results = {name: {'times': [], 'costs': []} for name in ['BKT', 'NN', 'HC', 'SA', 'GA']}
    for n in sizes:
        for seed in seeds:
            mat = generate_symmetric_matrix(n, seed=seed)
            solvers = {
                'BKT': BacktrackingSolver(mat),
                'NN': NearestNeighborSolver(mat),
                'HC': HillClimbingSolver(mat),
                'SA': SimulatedAnnealingSolver(mat),
                'GA': GeneticAlgorithmSolver(mat)
            }
            for name, solver in solvers.items():
                t0 = time.perf_counter()
                if name == 'BKT':
                    _, cost, _ = solver.solve(mode='timp', timp_max=10) if n > 10 else solver.solve(mode='toate')
                elif name == 'NN':
                    _, cost, _ = solver.solve(multistart=True)
                elif name == 'HC':
                    _, cost, _ = solver.solve(restarts_limit=50, iterations_limit=50)
                elif name == 'SA':
                    _, cost, _ = solver.solve(T_max=5000, T_min=1, alpha=0.99)
                elif name == 'GA':
                    _, cost, _ = solver.solve(pop_size=100, n_generations=300, mutation_rate=50)
                elapsed = time.perf_counter() - t0
                results[name]['times'].append(elapsed)
                results[name]['costs'].append(cost)

    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name in results:
        ax1.plot(sizes, results[name]['times'], label=name)
    ax1.set_xlabel('N')
    ax1.set_ylabel('Time (s)')
    ax1.set_title('Runtime vs N')
    ax1.legend()
    for name in results:
        ax2.plot(sizes, results[name]['costs'], label=name)
    ax2.set_xlabel('N')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost vs N')
    ax2.legend()
    plt.tight_layout()
    plt.savefig('tsp_comparison.png')
    plt.show()
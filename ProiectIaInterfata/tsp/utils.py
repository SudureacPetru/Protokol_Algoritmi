import numpy as np

def generate_symmetric_matrix(n: int, low=1, high=100, seed=42):
    rng = np.random.default_rng(seed)
    m = rng.integers(low, high + 1, size=(n, n))
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0)
    return m

def load_matrix_from_file(path: str):
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    n = int(lines[0])
    matrix = np.array([list(map(int, line.split())) for line in lines[1:1 + n]])
    return matrix
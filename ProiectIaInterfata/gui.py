import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import time

# Importuri TSP
from tsp.utils import generate_symmetric_matrix
from tsp.backtracking import BacktrackingSolver
from tsp.nearest_neighbor import NearestNeighborSolver
from tsp.hill_climbing import HillClimbingSolver
from tsp.simulated_annealing import SimulatedAnnealingSolver
from tsp.genetic_algorithm import GeneticAlgorithmSolver

# Importuri NLP
from nlp.datasets import load_20news, load_imdb_nltk
from nlp.pipeline import build_pipeline
from nlp.evaluate import evaluate_model


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Proiect IA – TSP & NLP")
        self.root.geometry("1100x750")
        self.root.configure(bg='#f0f2f5')

        # Stiluri
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f2f5')
        self.style.configure('TLabel', background='#f0f2f5', font=('Segoe UI', 10))
        self.style.configure('TLabelframe', background='#f0f2f5', font=('Segoe UI', 10, 'bold'))
        self.style.configure('TLabelframe.Label', background='#f0f2f5', foreground='#1a2639')
        self.style.configure('TButton', font=('Segoe UI', 10), padding=6)
        self.style.map('TButton', background=[('active', '#2d4373')])
        self.style.configure('Accent.TButton', background='#2d4373', foreground='white', borderwidth=0)
        self.style.map('Accent.TButton', background=[('active', '#1e3a5f')])
        self.style.configure('TNotebook', background='#f0f2f5', borderwidth=0)
        self.style.configure('TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[15, 5])
        self.style.map('TNotebook.Tab', background=[('selected', '#2d4373')], foreground=[('selected', 'white')])

        # Header
        header = tk.Frame(root, bg='#2d4373', height=50)
        header.pack(fill='x')
        title = tk.Label(header, text="⚡ Optimizare TSP & Clasificare NLP",
                         bg='#2d4373', fg='white', font=('Segoe UI', 14, 'bold'))
        title.pack(side='left', padx=10, pady=8)
        btn_team = ttk.Button(header, text="👥 Despre echipă", command=self.show_team_info)
        btn_team.pack(side='right', padx=10, pady=8)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)

        # Doar două tab-uri
        self.tab_tsp = ttk.Frame(self.notebook)
        self.tab_nlp = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_tsp, text="   🚚 TSP   ")
        self.notebook.add(self.tab_nlp, text="   📚 NLP   ")

        self.build_tsp_tab()
        self.build_nlp_tab()

    # ─── Fereastra "Despre echipă" ───────────────────────────────
    def show_team_info(self):
        top = tk.Toplevel(self.root)
        top.title("Despre echipă")
        top.geometry("400x350")
        top.configure(bg='#f0f2f5')
        info = """
🎓 Nume echipă: [Protokol]

👤 Membri:
    • [Sudureac Petru Daniel]
    • [Alexandroaie Valentin]
    • [Hrituc Denis]



"""
        lbl = tk.Label(top, text=info, bg='#f0f2f5', font=('Segoe UI', 11), justify='left')
        lbl.pack(pady=20, padx=20)

    # ─── TSP Tab (cu toate funcționalitățile) ────────────────────
    def build_tsp_tab(self):
        main = ttk.Frame(self.tab_tsp, padding=10)
        main.pack(fill='both', expand=True)

        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky='nw', padx=(0,10))

        # Configurare de bază
        cfg_frame = ttk.LabelFrame(left, text="⚙️ Configurare de bază", padding=10)
        cfg_frame.pack(fill='x', pady=(0,5))

        ttk.Label(cfg_frame, text="Număr orașe (N):").grid(row=0, column=0, sticky='w')
        self.n_entry = ttk.Entry(cfg_frame, width=10, font=('Segoe UI', 10))
        self.n_entry.insert(0, "8")
        self.n_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(cfg_frame, text="Random seed:").grid(row=1, column=0, sticky='w')
        self.seed_entry = ttk.Entry(cfg_frame, width=10, font=('Segoe UI', 10))
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(cfg_frame, text="Algoritm:").grid(row=2, column=0, sticky='w', pady=5)
        self.algo_var = tk.StringVar(value="BKT")
        self.algo_menu = ttk.Combobox(cfg_frame, textvariable=self.algo_var,
                                      values=["BKT", "NN", "HC", "SA", "GA"],
                                      state="readonly", width=12, font=('Segoe UI', 10))
        self.algo_menu.grid(row=2, column=1, padx=5, pady=5)
        self.algo_menu.bind('<<ComboboxSelected>>', self.on_algo_change)

        # Checkbox opțiuni avansate
        self.advanced_var = tk.BooleanVar(value=False)
        self.chk_advanced = ttk.Checkbutton(cfg_frame, text="Opțiuni avansate",
                                            variable=self.advanced_var,
                                            command=self.toggle_advanced_options)
        self.chk_advanced.grid(row=3, column=0, columnspan=2, sticky='w', pady=(5,0))

        self.advanced_frame = ttk.Frame(cfg_frame)
        self.advanced_params = {}

        # Buton execuție unică
        self.btn_run_tsp = ttk.Button(cfg_frame, text="▶️ Rulează", style='Accent.TButton',
                                      command=self.run_tsp)
        self.btn_run_tsp.grid(row=10, column=0, columnspan=2, pady=10)

        # Rezultate
        self.result_box = tk.Frame(left, bg='white', highlightbackground='#e0e0e0', highlightthickness=1)
        self.result_box.pack(fill='x', pady=(0,5))
        self.lbl_algo = tk.Label(self.result_box, text="Algoritm: -", bg='white', fg='#1f2937',
                                 font=('Segoe UI', 11, 'bold'), anchor='w')
        self.lbl_algo.pack(fill='x', padx=10, pady=(10,0))
        self.lbl_cost = tk.Label(self.result_box, text="Cost: -", bg='white', fg='#1f2937', font=('Segoe UI', 10))
        self.lbl_cost.pack(fill='x', padx=10)
        self.lbl_route = tk.Label(self.result_box, text="Traseu: -", bg='white', fg='#1f2937',
                                  font=('Segoe UI', 9), wraplength=250, justify='left')
        self.lbl_route.pack(fill='x', padx=10)
        self.lbl_time = tk.Label(self.result_box, text="Timp: -", bg='white', fg='#6b7280', font=('Segoe UI', 9))
        self.lbl_time.pack(fill='x', padx=10, pady=(0,10))

        # Secțiunea Benchmark
        bench_frame = ttk.LabelFrame(left, text="📊 Benchmark", padding=10)
        bench_frame.pack(fill='x', pady=5)
        ttk.Label(bench_frame, text="N-uri (ex: 10,20,50):").grid(row=0, column=0, sticky='w')
        self.bench_n_entry = ttk.Entry(bench_frame, width=20)
        self.bench_n_entry.insert(0, "10,20,50")
        self.bench_n_entry.grid(row=0, column=1, padx=5, pady=5)
        self.btn_bench = ttk.Button(bench_frame, text="▶️ Rulează benchmark", command=self.run_benchmark)
        self.btn_bench.grid(row=1, column=0, columnspan=2, pady=5)

        # Butoane suplimentare
        extra_frame = ttk.LabelFrame(left, text="📈 Grafice rapide", padding=10)
        extra_frame.pack(fill='x', pady=5)
        ttk.Button(extra_frame, text="Curba de convergență (GA)", command=self.show_convergence_ga).pack(fill='x', pady=2)
        ttk.Button(extra_frame, text="Heatmap distanțe", command=self.show_heatmap_distances).pack(fill='x', pady=2)
        ttk.Button(extra_frame, text="💾 Salvează matrice", command=self.save_matrix).pack(fill='x', pady=2)

        # Containere grafice
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.route_frame = ttk.LabelFrame(right, text="🗺️ Traseu", padding=10)
        self.route_figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor='#f0f2f5')
        self.route_canvas = FigureCanvasTkAgg(self.route_figure, master=self.route_frame)
        self.route_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.extra_graph_frame = ttk.LabelFrame(right, text="📊 Grafic", padding=10)
        self.extra_figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor='#f0f2f5')
        self.extra_canvas = FigureCanvasTkAgg(self.extra_figure, master=self.extra_graph_frame)
        self.extra_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.show_route_graph()
        self.toggle_advanced_options()

    def show_route_graph(self):
        self.extra_graph_frame.pack_forget()
        self.route_frame.pack(fill='both', expand=True)

    def show_extra_graph(self):
        self.route_frame.pack_forget()
        self.extra_graph_frame.pack(fill='both', expand=True)

    def on_algo_change(self, event=None):
        self.update_tsp_advanced()

    def toggle_advanced_options(self):
        if self.advanced_var.get():
            self.advanced_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)
            self.update_tsp_advanced()
        else:
            self.advanced_frame.grid_forget()

    def update_tsp_advanced(self, event=None):
        for widget in self.advanced_frame.winfo_children():
            widget.destroy()
        self.advanced_params.clear()
        algo = self.algo_var.get()
        params = []
        if algo == 'GA':
            params = [('pop_size', 'Dimensiune populație', '100'),
                      ('n_generations', 'Număr generații', '300'),
                      ('mutation_rate', 'Rată mutație (%)', '50')]
        elif algo == 'SA':
            params = [('T_max', 'Temperatură maximă', '5000'),
                      ('alpha', 'Factor răcire (alpha)', '0.995'),
                      ('use_nn_start', 'Start NN (0/1)', '1')]
        elif algo == 'HC':
            params = [('restarts_limit', 'Număr reporniri', '50'),
                      ('iterations_limit', 'Iterații per repornire', '50')]
        elif algo == 'BKT':
            params = [('mode', 'Mod oprire (toate/prima/timp/y_solutii)', 'toate'),
                      ('timp_max', 'Timp maxim (secunde)', '30'),
                      ('y_max', 'Număr maxim soluții', '10')]
        elif algo == 'NN':
            params = [('multistart', 'Multistart (0/1)', '1')]
        for i, (key, label, default) in enumerate(params):
            ttk.Label(self.advanced_frame, text=label+':').grid(row=i, column=0, sticky='w', pady=2)
            entry = ttk.Entry(self.advanced_frame, width=15, font=('Segoe UI', 9))
            entry.insert(0, default)
            entry.grid(row=i, column=1, pady=2, padx=5)
            self.advanced_params[key] = entry

    def run_tsp(self):
        try:
            n = int(self.n_entry.get())
            seed = int(self.seed_entry.get())
        except:
            messagebox.showerror("Eroare", "N și seed trebuie să fie numere întregi")
            return
        algo = self.algo_var.get()
        self.dist_matrix = generate_symmetric_matrix(n, seed=seed)
        advanced_kwargs = {}
        if self.advanced_var.get():
            for key, entry in self.advanced_params.items():
                val = entry.get()
                if key in ('pop_size', 'n_generations', 'mutation_rate',
                           'restarts_limit', 'iterations_limit', 'T_max',
                           'timp_max', 'y_max'):
                    advanced_kwargs[key] = int(val)
                elif key in ('alpha',):
                    advanced_kwargs[key] = float(val)
                elif key in ('use_nn_start', 'multistart'):
                    advanced_kwargs[key] = bool(int(val))
                else:
                    advanced_kwargs[key] = val
        self.btn_run_tsp.config(state='disabled', text="🔄 Se rulează...")
        threading.Thread(target=self._solve_tsp, args=(algo, n, seed, advanced_kwargs), daemon=True).start()

    def _solve_tsp(self, algo, n, seed, kwargs):
        try:
            solver = None
            if algo == 'BKT':
                solver = BacktrackingSolver(self.dist_matrix)
                mode = kwargs.get('mode', 'toate')
                timp_max = kwargs.get('timp_max', 30)
                y_max = kwargs.get('y_max', 10)
                route, cost, stats = solver.solve(mode=mode, timp_max=timp_max, y_max=y_max)
            elif algo == 'NN':
                solver = NearestNeighborSolver(self.dist_matrix)
                multistart = kwargs.get('multistart', True)
                route, cost, stats = solver.solve(multistart=multistart)
            elif algo == 'HC':
                solver = HillClimbingSolver(self.dist_matrix)
                restarts = kwargs.get('restarts_limit', 50)
                iterations = kwargs.get('iterations_limit', 50)
                route, cost, stats = solver.solve(restarts_limit=restarts, iterations_limit=iterations)
            elif algo == 'SA':
                solver = SimulatedAnnealingSolver(self.dist_matrix)
                T_max = kwargs.get('T_max', 5000)
                alpha = kwargs.get('alpha', 0.995)
                use_nn = kwargs.get('use_nn_start', True)
                route, cost, stats = solver.solve(T_max=T_max, T_min=1, alpha=alpha, use_nn_start=use_nn)
            elif algo == 'GA':
                solver = GeneticAlgorithmSolver(self.dist_matrix)
                pop_size = kwargs.get('pop_size', 100)
                n_gen = kwargs.get('n_generations', 300)
                mut_rate = kwargs.get('mutation_rate', 50)
                route, cost, stats = solver.solve(pop_size=pop_size, n_generations=n_gen, mutation_rate=mut_rate)
            self.root.after(0, self._update_tsp_results, route, cost, stats)
        except Exception as ex:
            self.root.after(0, lambda err=ex: messagebox.showerror("Eroare", str(err)))
        finally:
            self.root.after(0, lambda: self.btn_run_tsp.config(state='normal', text="▶️ Rulează"))

    def _update_tsp_results(self, route, cost, stats):
        self.lbl_algo.config(text=f"Algoritm: {self.algo_var.get()}")
        self.lbl_cost.config(text=f"Cost: {cost:.2f}")
        self.lbl_route.config(text=f"Traseu: {' → '.join(map(str, route))} → {route[0]}")
        t = stats.get('time', 0)
        self.lbl_time.config(text=f"Timp: {t:.4f} s")
        self.route_figure.clear()
        ax = self.route_figure.add_subplot(111)
        np.random.seed(42)
        cities = np.random.rand(len(route), 2) * 100
        ordered = [cities[i] for i in route] + [cities[route[0]]]
        xs, ys = zip(*ordered)
        ax.plot(xs, ys, '#2d4373', marker='o', linewidth=2, markersize=8)
        for i, (x, y) in enumerate(cities):
            ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(5,5), fontsize=10, color='#1a2639')
        ax.set_title(f"Traseu optim ({self.algo_var.get()}) – Cost: {cost:.2f}", fontsize=12, color='#1a2639')
        ax.set_facecolor('#f0f2f5')
        self.route_figure.tight_layout()
        self.route_canvas.draw()
        self.show_route_graph()

    # ─── Salvare matrice ────────────────────────────────────────
    def save_matrix(self):
        if not hasattr(self, 'dist_matrix'):
            messagebox.showerror("Eroare", "Mai întâi rulează un algoritm pentru a genera matricea.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not filepath:
            return
        n = self.dist_matrix.shape[0]
        with open(filepath, 'w') as f:
            f.write(f"{n}\n")
            for i in range(n):
                row = ' '.join(map(str, self.dist_matrix[i].astype(int)))
                f.write(row + '\n')
        messagebox.showinfo("Succes", f"Matricea a fost salvată în {filepath}")

    # ─── Benchmark ──────────────────────────────────────────────
    def run_benchmark(self):
        raw = self.bench_n_entry.get()
        try:
            sizes = [int(x.strip()) for x in raw.split(',')]
        except:
            messagebox.showerror("Eroare", "Introduceți N-uri valide separate prin virgulă.")
            return
        self.btn_bench.config(state='disabled', text="🔄 Rulează...")
        threading.Thread(target=self._benchmark_thread, args=(sizes,), daemon=True).start()

    def _benchmark_thread(self, sizes):
        try:
            algorithms = ['NN', 'HC', 'SA', 'GA']
            times = {a: [] for a in algorithms}
            costs = {a: [] for a in algorithms}
            bkt_times, bkt_costs = [], []
            for n in sizes:
                mat = generate_symmetric_matrix(n, seed=42)
                # NN
                t0 = time.perf_counter()
                solver = NearestNeighborSolver(mat)
                _, cost, _ = solver.solve(multistart=True)
                times['NN'].append(time.perf_counter() - t0)
                costs['NN'].append(cost)
                # HC
                t0 = time.perf_counter()
                solver = HillClimbingSolver(mat)
                _, cost, _ = solver.solve(restarts_limit=20, iterations_limit=20)
                times['HC'].append(time.perf_counter() - t0)
                costs['HC'].append(cost)
                # SA
                t0 = time.perf_counter()
                solver = SimulatedAnnealingSolver(mat)
                _, cost, _ = solver.solve(T_max=5000, T_min=1, alpha=0.995)
                times['SA'].append(time.perf_counter() - t0)
                costs['SA'].append(cost)
                # GA (parametri reduși pentru N mare)
                pop = 50 if n > 30 else 100
                gen = 100 if n > 30 else 300
                t0 = time.perf_counter()
                solver = GeneticAlgorithmSolver(mat)
                _, cost, _ = solver.solve(pop_size=pop, n_generations=gen, mutation_rate=50)
                times['GA'].append(time.perf_counter() - t0)
                costs['GA'].append(cost)
                # BKT doar pentru N<=12
                if n <= 12:
                    t0 = time.perf_counter()
                    solver = BacktrackingSolver(mat)
                    _, cost, _ = solver.solve(mode='timp', timp_max=30)
                    bkt_times.append(time.perf_counter() - t0)
                    bkt_costs.append(cost)
            self.root.after(0, self._update_benchmark_graph, sizes, times, costs,
                            bkt_times if bkt_times else None, bkt_costs if bkt_costs else None)
        except Exception as ex:
            self.root.after(0, lambda err=ex: messagebox.showerror("Eroare", str(err)))
        finally:
            self.root.after(0, lambda: self.btn_bench.config(state='normal', text="▶️ Rulează benchmark"))

    def _update_benchmark_graph(self, sizes, times, costs, bkt_times=None, bkt_costs=None):
        self.extra_figure.clear()
        # Grafic 1: Timp vs N
        ax1 = self.extra_figure.add_subplot(221)
        for algo in times:
            ax1.plot(sizes, times[algo], marker='o', label=algo)
        if bkt_times:
            ax1.plot([s for s in sizes if s <= 12], bkt_times, marker='s', label='BKT')
        ax1.set_xlabel('N')
        ax1.set_ylabel('Timp (s)')
        ax1.set_title('Timp de execuție vs. N')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Grafic 2: Cost vs N
        ax2 = self.extra_figure.add_subplot(222)
        for algo in costs:
            ax2.plot(sizes, costs[algo], marker='o', label=algo)
        if bkt_costs:
            ax2.plot([s for s in sizes if s <= 12], bkt_costs, marker='s', label='BKT')
        ax2.set_xlabel('N')
        ax2.set_ylabel('Cost')
        ax2.set_title('Cost vs. N')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Grafic 3: Cost vs Timp (scatter)
        ax3 = self.extra_figure.add_subplot(223)
        for algo in times:
            ax3.scatter(times[algo], costs[algo], label=algo)
        if bkt_times and bkt_costs:
            ax3.scatter(bkt_times, bkt_costs, marker='s', label='BKT')
        ax3.set_xlabel('Timp (s)')
        ax3.set_ylabel('Cost')
        ax3.set_title('Cost vs. Timp')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        self.extra_figure.tight_layout()
        self.extra_canvas.draw()
        self.show_extra_graph()

    # ─── Curba de convergență GA ────────────────────────────────
    def show_convergence_ga(self):
        try:
            n = int(self.n_entry.get())
            seed = int(self.seed_entry.get())
        except:
            messagebox.showerror("Eroare", "Introdu N și seed valide")
            return
        mat = generate_symmetric_matrix(n, seed=seed)
        solver = GeneticAlgorithmSolver(mat)
        _, _, stats = solver.solve(pop_size=100, n_generations=300, mutation_rate=50)
        if 'convergence' not in stats:
            messagebox.showerror("Eroare", "Convergența nu a fost returnată")
            return
        conv = stats['convergence']
        self.extra_figure.clear()
        ax = self.extra_figure.add_subplot(111)
        ax.plot(conv, color='#2d4373')
        ax.set_xlabel("Generație")
        ax.set_ylabel("Cost")
        ax.set_title("Curba de convergență – GA")
        ax.grid(True, alpha=0.3)
        self.extra_figure.tight_layout()
        self.extra_canvas.draw()
        self.show_extra_graph()

    def show_heatmap_distances(self):
        try:
            n = int(self.n_entry.get())
            seed = int(self.seed_entry.get())
        except:
            messagebox.showerror("Eroare", "Introdu N și seed valide")
            return
        mat = generate_symmetric_matrix(n, seed=seed)
        self.extra_figure.clear()
        ax = self.extra_figure.add_subplot(111)
        im = ax.imshow(mat, cmap='YlOrRd', interpolation='nearest')
        self.extra_figure.colorbar(im, ax=ax)
        ax.set_title("Heatmap distanțe")
        ax.set_xlabel("Oraș")
        ax.set_ylabel("Oraș")
        self.extra_figure.tight_layout()
        self.extra_canvas.draw()
        self.show_extra_graph()

    # ─── NLP Tab ─────────────────────────────────────────────────
    def build_nlp_tab(self):
        main = ttk.Frame(self.tab_nlp, padding=10)
        main.pack(fill='both', expand=True)

        left = ttk.LabelFrame(main, text="⚙️ Configurare", padding=15)
        left.grid(row=0, column=0, sticky='nw', padx=(0,10))

        ttk.Label(left, text="Dataset:").grid(row=0, column=0, sticky='w')
        self.dataset_var = tk.StringVar(value="20news")
        dataset_menu = ttk.Combobox(left, textvariable=self.dataset_var,
                                    values=["20news", "imdb"], state="readonly", width=18, font=('Segoe UI', 10))
        dataset_menu.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(left, text="Clasificator:").grid(row=1, column=0, sticky='w', pady=5)
        self.clf_var = tk.StringVar(value="LinearSVC")
        clf_menu = ttk.Combobox(left, textvariable=self.clf_var,
                                values=["NaiveBayes", "LinearSVC", "LogisticRegression", "RandomForest"],
                                state="readonly", width=18, font=('Segoe UI', 10))
        clf_menu.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(left, text="ngram_range (ex: 1,2):").grid(row=2, column=0, sticky='w')
        self.ngram_entry = ttk.Entry(left, width=10, font=('Segoe UI', 10))
        self.ngram_entry.insert(0, "1,2")
        self.ngram_entry.grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(left, text="max_features (0=all):").grid(row=3, column=0, sticky='w')
        self.maxfeat_entry = ttk.Entry(left, width=10, font=('Segoe UI', 10))
        self.maxfeat_entry.insert(0, "10000")
        self.maxfeat_entry.grid(row=3, column=1, padx=5, pady=5)

        self.btn_run_nlp = ttk.Button(left, text="▶️ Rulează", style='Accent.TButton', command=self.run_nlp)
        self.btn_run_nlp.grid(row=4, column=0, columnspan=2, pady=20)

        result_box = tk.Frame(left, bg='white', highlightbackground='#e0e0e0', highlightthickness=1)
        result_box.grid(row=5, column=0, columnspan=2, sticky='ew', pady=(10,5))
        self.lbl_clf = tk.Label(result_box, text="Clasificator: -", bg='white', fg='#1f2937',
                                font=('Segoe UI', 11, 'bold'), anchor='w')
        self.lbl_clf.pack(fill='x', padx=10, pady=(10,0))
        self.lbl_acc = tk.Label(result_box, text="Acuratețe: -", bg='white', fg='#1f2937', font=('Segoe UI', 10))
        self.lbl_acc.pack(fill='x', padx=10)
        self.lbl_nlp_time = tk.Label(result_box, text="Timp antrenament: -", bg='white', fg='#6b7280', font=('Segoe UI', 9))
        self.lbl_nlp_time.pack(fill='x', padx=10, pady=(0,10))

        # Butoane suplimentare NLP
        extra_btn_frame = ttk.Frame(left)
        extra_btn_frame.grid(row=6, column=0, columnspan=2, pady=5)
        ttk.Button(extra_btn_frame, text="Compară clasificatori", command=self.show_classifier_comparison).pack(fill='x', pady=2)
        ttk.Button(extra_btn_frame, text="Studiu ngram_range", command=self.show_ngram_study).pack(fill='x', pady=2)
        ttk.Button(extra_btn_frame, text="Studiu max_features", command=self.show_maxfeat_study).pack(fill='x', pady=2)
        ttk.Button(extra_btn_frame, text="Acuratețe vs Timp", command=self.show_acc_vs_time).pack(fill='x', pady=2)

        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky='nsew')
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        self.nlp_route_frame = ttk.LabelFrame(right, text="📊 Matrice de confuzie", padding=10)
        self.nlp_route_figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor='#f0f2f5')
        self.nlp_route_canvas = FigureCanvasTkAgg(self.nlp_route_figure, master=self.nlp_route_frame)
        self.nlp_route_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.nlp_extra_frame = ttk.LabelFrame(right, text="📊 Grafic", padding=10)
        self.nlp_extra_figure = plt.Figure(figsize=(5, 4), dpi=100, facecolor='#f0f2f5')
        self.nlp_extra_canvas = FigureCanvasTkAgg(self.nlp_extra_figure, master=self.nlp_extra_frame)
        self.nlp_extra_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.show_nlp_route_graph()

    def show_nlp_route_graph(self):
        self.nlp_extra_frame.pack_forget()
        self.nlp_route_frame.pack(fill='both', expand=True)

    def show_nlp_extra_graph(self):
        self.nlp_route_frame.pack_forget()
        self.nlp_extra_frame.pack(fill='both', expand=True)

    def run_nlp(self):
        self.btn_run_nlp.config(state='disabled', text="🔄 Se rulează...")
        threading.Thread(target=self._solve_nlp, daemon=True).start()

    def _solve_nlp(self):
        try:
            dataset = self.dataset_var.get()
            if dataset == '20news':
                X_train, y_train, target_names = load_20news(subset='train')
                X_test, y_test, _ = load_20news(subset='test')
            else:
                X, y, target_names = load_imdb_nltk()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            ng_text = self.ngram_entry.get()
            ngram = tuple(map(int, ng_text.split(',')))
            mf_text = self.maxfeat_entry.get()
            max_feat = int(mf_text) if mf_text != '0' else None
            clf_name = self.clf_var.get()
            pipe = build_pipeline(clf_name, ngram_range=ngram, max_features=max_feat)
            acc, pred, t = evaluate_model(pipe, X_train, y_train, X_test, y_test, target_names, verbose=False)
            self.root.after(0, self._update_nlp_results, acc, t, clf_name, y_test, pred, target_names)
        except Exception as ex:
            self.root.after(0, lambda err=ex: messagebox.showerror("Eroare", str(err)))
        finally:
            self.root.after(0, lambda: self.btn_run_nlp.config(state='normal', text="▶️ Rulează"))

    def _update_nlp_results(self, acc, t, clf_name, y_test, pred, target_names):
        self.lbl_clf.config(text=f"Clasificator: {clf_name}")
        self.lbl_acc.config(text=f"Acuratețe: {acc:.4f}")
        self.lbl_nlp_time.config(text=f"Timp antrenament: {t:.2f} s")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, pred)
        self.nlp_route_figure.clear()
        ax = self.nlp_route_figure.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        self.nlp_route_figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(target_names)))
        ax.set_yticks(np.arange(len(target_names)))
        ax.set_xticklabels(target_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(target_names, fontsize=8)
        ax.set_title(f"Matrice de confuzie – {clf_name}", fontsize=12, color='#1a2639')
        for i in range(len(target_names)):
            for j in range(len(target_names)):
                ax.text(j, i, cm[i, j], ha='center', va='center',
                        color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=9)
        self.nlp_route_figure.tight_layout()
        self.nlp_route_canvas.draw()
        self.show_nlp_route_graph()

    # ─── Grafice NLP suplimentare ────────────────────────────────
    def show_classifier_comparison(self):
        try:
            dataset = self.dataset_var.get()
            if dataset == '20news':
                X_train, y_train, target_names = load_20news(subset='train')
                X_test, y_test, _ = load_20news(subset='test')
            else:
                X, y, target_names = load_imdb_nltk()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            classifiers = ["NaiveBayes", "LinearSVC", "LogisticRegression", "RandomForest"]
            accuracies = []
            for clf in classifiers:
                pipe = build_pipeline(clf, ngram_range=(1,2), max_features=10000)
                acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, target_names, verbose=False)
                accuracies.append(acc)
            self.nlp_extra_figure.clear()
            ax = self.nlp_extra_figure.add_subplot(111)
            ax.bar(classifiers, accuracies, color='#2d4373')
            ax.set_ylabel("Acuratețe")
            ax.set_title("Comparație clasificatori")
            for i, v in enumerate(accuracies):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
            self.nlp_extra_figure.tight_layout()
            self.nlp_extra_canvas.draw()
            self.show_nlp_extra_graph()
        except Exception as ex:
            messagebox.showerror("Eroare", str(ex))

    def show_ngram_study(self):
        try:
            dataset = self.dataset_var.get()
            if dataset == '20news':
                X_train, y_train, target_names = load_20news(subset='train')
                X_test, y_test, _ = load_20news(subset='test')
            else:
                X, y, target_names = load_imdb_nltk()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            ngrams = [(1,1), (1,2), (2,2), (1,3)]
            accs = []
            for ng in ngrams:
                pipe = build_pipeline("LinearSVC", ngram_range=ng, max_features=10000)
                acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, target_names, verbose=False)
                accs.append(acc)
            self.nlp_extra_figure.clear()
            ax = self.nlp_extra_figure.add_subplot(111)
            ax.bar([str(ng) for ng in ngrams], accs, color='#2d4373')
            ax.set_ylabel("Acuratețe")
            ax.set_title("Studiu ngram_range")
            for i, v in enumerate(accs):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
            self.nlp_extra_figure.tight_layout()
            self.nlp_extra_canvas.draw()
            self.show_nlp_extra_graph()
        except Exception as ex:
            messagebox.showerror("Eroare", str(ex))

    def show_maxfeat_study(self):
        try:
            dataset = self.dataset_var.get()
            if dataset == '20news':
                X_train, y_train, target_names = load_20news(subset='train')
                X_test, y_test, _ = load_20news(subset='test')
            else:
                X, y, target_names = load_imdb_nltk()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            maxfeats = [100, 500, 1000, 5000, 10000, None]
            accs = []
            labels = [str(mf) if mf is not None else 'all' for mf in maxfeats]
            for mf in maxfeats:
                pipe = build_pipeline("LinearSVC", ngram_range=(1,2), max_features=mf)
                acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, target_names, verbose=False)
                accs.append(acc)
            self.nlp_extra_figure.clear()
            ax = self.nlp_extra_figure.add_subplot(111)
            ax.bar(labels, accs, color='#2d4373')
            ax.set_ylabel("Acuratețe")
            ax.set_title("Studiu max_features")
            for i, v in enumerate(accs):
                ax.text(i, v + 0.01, f"{v:.3f}", ha='center')
            self.nlp_extra_figure.tight_layout()
            self.nlp_extra_canvas.draw()
            self.show_nlp_extra_graph()
        except Exception as ex:
            messagebox.showerror("Eroare", str(ex))

    def show_acc_vs_time(self):
        try:
            dataset = self.dataset_var.get()
            if dataset == '20news':
                X_train, y_train, target_names = load_20news(subset='train')
                X_test, y_test, _ = load_20news(subset='test')
            else:
                X, y, target_names = load_imdb_nltk()
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            classifiers = ["NaiveBayes", "LinearSVC", "LogisticRegression", "RandomForest"]
            accs, times = [], []
            for clf in classifiers:
                pipe = build_pipeline(clf, ngram_range=(1,2), max_features=10000)
                t0 = time.perf_counter()
                acc, _, _ = evaluate_model(pipe, X_train, y_train, X_test, y_test, target_names, verbose=False)
                elapsed = time.perf_counter() - t0
                accs.append(acc)
                times.append(elapsed)
            self.nlp_extra_figure.clear()
            ax = self.nlp_extra_figure.add_subplot(111)
            for i, clf in enumerate(classifiers):
                ax.scatter(times[i], accs[i], label=clf, s=100)
                ax.annotate(clf, (times[i], accs[i]), textcoords="offset points", xytext=(5,5))
            ax.set_xlabel('Timp antrenament (s)')
            ax.set_ylabel('Acuratețe')
            ax.set_title('Acuratețe vs. Timp')
            ax.grid(True, alpha=0.3)
            ax.legend()
            self.nlp_extra_figure.tight_layout()
            self.nlp_extra_canvas.draw()
            self.show_nlp_extra_graph()
        except Exception as ex:
            messagebox.showerror("Eroare", str(ex))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
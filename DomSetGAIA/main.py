import sys
import matplotlib.pyplot as plt
import networkx as nx
import random
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit,
                             QGridLayout, QComboBox, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

class GeneticAlgorithm:
    def __init__(self):
        pass

    def genetic_algorithm_with_progress(self, graph, population_size, generations):
        def initialize_population(size, graph):
            return [set(random.sample(list(graph.nodes()),
                                      k=random.randint(1, graph.number_of_nodes()))) for _ in range(size)]

        def fitness(solution, graph):
            dominated_nodes = set(solution)
            for node in solution:
                dominated_nodes |= set(graph.neighbors(node))
            return len(dominated_nodes), -len(solution)

        def select(population, graph, k=5):
            fitness_scores = [(chromosome, fitness(chromosome, graph)) for chromosome in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            return [chromosome for chromosome, _ in fitness_scores[:k]]

        def crossover(parent1, parent2):
            child1 = parent1.union(parent2)
            child2 = parent1.intersection(parent2)
            return [child1, child2]

        def mutate(solution, graph, mutation_rate=0.1):
            if random.random() < mutation_rate:
                if random.random() > 0.5 and solution:
                    solution.remove(random.choice(list(solution)))
                else:
                    solution.add(random.choice(list(set(graph.nodes()) - solution)))
            return solution

        population = initialize_population(population_size, graph)
        progress = []

        for _ in range(generations):
            new_generation = []
            selected = select(population, graph)
            for i in range(0, len(selected), 2):
                for child in crossover(selected[i], selected[(i + 1) % len(selected)]):
                    new_generation.append(mutate(child, graph))
            population = select(population + new_generation, graph, k=population_size)
            best_solution = select(population, graph, k=1)[0]
            progress.append(-fitness(best_solution, graph)[1])

        return best_solution, progress

class ImmuneAlgorithm:
    def __init__(self):
        pass

    def immune_algorithm_with_progress(self, graph, population_size, generations):
        def initialize_population(size, graph):
            return [set(random.sample(list(graph.nodes()), k=random.randint(1, graph.number_of_nodes()))) for _ in range(size)]

        def evaluate(solution, graph):
            dominated_nodes = set(solution)
            for subset in solution:
                dominated_nodes |= set(graph.neighbors(subset))
            return len(dominated_nodes), -len(solution)

        def select(population):
            population.sort(key=lambda x: evaluate(x, graph), reverse=True)
            return population[:population_size]

        def crossover(parent1, parent2):
            child1 = parent1.union(parent2)
            child2 = parent1.intersection(parent2)
            return [child1, child2]

        def mutate(solution, graph, mutation_rate=0.1):
            if random.random() < mutation_rate:
                available_nodes = set(graph.nodes()) - set(solution)
                if available_nodes:
                    if random.random() > 0.5 and solution:
                        solution.remove(random.choice(list(solution)))
                    else:
                        solution.add(random.choice(list(available_nodes)))
            return solution

        population = initialize_population(population_size, graph)
        progress = []

        for _ in range(generations):
            new_generation = []
            selected = select(population)

            for i in range(0, len(selected), 2):
                for child in crossover(selected[i], selected[(i + 1) % len(selected)]):
                    new_generation.append(mutate(child, graph))

            population = select(population + new_generation)
            best_solution = select(population)[0]
            progress.append(-evaluate(best_solution, graph)[1])

        return best_solution, progress

class GAVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.genetic_algo = GeneticAlgorithm()
        self.immune_algo = ImmuneAlgorithm()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        # Создание виджетов для параметров алгоритма
        grid.addWidget(QLabel('Кол-во вершин:'), 0, 0)
        self.vertex_count_input = QLineEdit('20')
        grid.addWidget(self.vertex_count_input, 0, 1)

        grid.addWidget(QLabel('Граничная вероятность:'), 1, 0)
        self.probability_input = QLineEdit('0.14')
        grid.addWidget(self.probability_input, 1, 1)

        grid.addWidget(QLabel('Кол-во популяций:'), 2, 0)
        self.population_size_input = QLineEdit('100')
        grid.addWidget(self.population_size_input, 2, 1)

        grid.addWidget(QLabel('Кол-во поколений:'), 3, 0)
        self.generations_input = QLineEdit('100')
        grid.addWidget(self.generations_input, 3, 1)

        self.run_button_genetic = QPushButton('Запустить ГА')
        self.run_button_genetic.clicked.connect(self.run_genetic_algorithm)
        grid.addWidget(self.run_button_genetic, 4, 0, 1, 2)

        self.run_button_immune = QPushButton('Запустить ИА')
        self.run_button_immune.clicked.connect(self.run_immune_algorithm)
        grid.addWidget(self.run_button_immune, 5, 0, 1, 2)

        # Поле для отображения графиков
        self.figure = plt.figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        grid.addWidget(self.canvas, 6, 0, 1, 2)

        self.setWindowTitle('Визуализатор генетических и иммунных алгоритмов')
        self.setGeometry(300, 300, 600, 400)

    def run_genetic_algorithm(self):
        start_time = time.time()  # Начало замера времени

        vertex_count = int(self.vertex_count_input.text())
        probability = float(self.probability_input.text())
        population_size = int(self.population_size_input.text())
        generations = int(self.generations_input.text())

        G = nx.erdos_renyi_graph(vertex_count, probability)
        pos = nx.spring_layout(G)
        dominating_set, progress = self.genetic_algo.genetic_algorithm_with_progress(G, population_size, generations)

        end_time = time.time()  # Окончание замера времени
        elapsed_time = end_time - start_time  # Вычисление затраченного времени

        print(f"Генетический алгоритм выполнился за {elapsed_time:.2f} секунд")  # Вывод времени выполнения в консоль

        self.plot_results(G, pos, dominating_set, progress)

    def run_immune_algorithm(self):
        start_time = time.time()  # Начало замера времени

        vertex_count = int(self.vertex_count_input.text())
        probability = float(self.probability_input.text())
        population_size = int(self.population_size_input.text())
        generations = int(self.generations_input.text())

        G = nx.erdos_renyi_graph(vertex_count, probability)
        pos = nx.spring_layout(G)
        dominating_set, progress = self.immune_algo.immune_algorithm_with_progress(G, population_size, generations)

        end_time = time.time()  # Окончание замера времени
        elapsed_time = end_time - start_time  # Вычисление затраченного времени

        print(f"Иммунный алгоритм выполнился за {elapsed_time:.2f} секунд")  # Вывод времени выполнения в консоль

        self.plot_results(G, pos, dominating_set, progress)

    def plot_results(self, graph, pos, dominating_set, progress):
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        nx.draw(graph, pos, ax=ax1, with_labels=True, node_color=['red' if node in dominating_set else 'blue' for node in graph.nodes()])
        ax1.set_title("График с минимальным доминирующим множеством")

        ax2 = self.figure.add_subplot(122)
        ax2.plot(progress)
        ax2.set_title("Прогресс алгоритма на протяжении многих поколений")
        ax2.set_xlabel("Поколения")
        ax2.set_ylabel("Размер минимального доминирующего множества")
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = GAVisualizer()
    ex.show()
    sys.exit(app.exec_())

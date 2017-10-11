import time
import numpy as np
import os

class Comparator:
    def __init__(self, fitness_functions, algorithms):
        self.algorithms = algorithms
        self.fitness_functions = fitness_functions
    def run(self, num_executions, num_iterations, output_path=''):
        root_directory_name = '{}_{}x{}/'.format(time.time(), num_executions, num_iterations)
        root_directory = output_path + root_directory_name
        os.mkdir(root_directory)
        for algorithm in self.algorithms:
            algorithm_directory = root_directory + algorithm.name + '/'
            os.mkdir(algorithm_directory)
            for fitness_function in self.fitness_functions:
                file_path = algorithm_directory+'results_'+fitness_function.name + '.csv'
                best_fitness = []
                times = []
                solutions = []
                for execution in range(num_executions):
                    best_fitness.append([])
                    start_time = time.time()
                    for _ in range(num_iterations):
                        algorithm.optimize(fitness_function, 1)
                        best_fitness[execution].append(algorithm.best_fitness)
                    end_time = time.time()
                    times.append(end_time-start_time)
                    solutions.append(algorithm.solution)
                with open(file_path, mode='a') as f:
                    for line in best_fitness:
                        for fitness in line:
                            f.write(str(fitness)+',')
                        f.write('\n')
                    for t, s in zip(times, solutions):
                        f.write('{},{}\n'.format(t, s))

class OptmizationAlgorithm:
    def __init__(self, name):
        self.solution = None
        self.best_fitness = -np.inf
        self.name = name
    def reset(self):
        raise NotImplementedError()
    def optimize(self, fitness_function, num_iterations):
        raise NotImplementedError()
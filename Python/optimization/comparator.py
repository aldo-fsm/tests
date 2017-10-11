import time
import numpy as np

class Comparator:
    def __init__(self, fitness_functions, algorithms):
        self.algorithms = algorithms
        self.fitness_functions = fitness_functions
        self.data = [{
            a.name: {
                'fitness': [], #best fitness in each iteration for each exectution
                'solutions': [], #optimum point found in each execution
                'time': [] #time spent in each execution
            }
            for a in self.algorithms
        } for _ in fitness_functions]
    def __optimize(self, num_iterations, ff_index):
        fitness_function = self.fitness_functions[ff_index]
        data = self.data[ff_index]
        for a in self.algorithms:
            a_data = data[a.name]
            a_data['fitness'].append([])

            start_time = time.time()
            for _ in range(num_iterations):
                a.optimize(fitness_function, 1)
                a_data['fitness'][self.execution].append(a.best_fitness)
            end_time = time.time()

            a_data['solutions'].append(a.solution)
            a_data['time'].append(end_time - start_time)

    def __execute(self, ff_index, num_executions, num_iterations):
        self.execution = 0
        while self.execution < num_executions:
            for a in self.algorithms:
                a.reset()
            self.__optimize(num_iterations, ff_index)
            print(str(100*(self.execution+1)/num_executions)+' %')
            self.execution += 1

    def start(self, num_executions, num_iterations):
        for ff_index in range(len(self.fitness_functions)):
            print('============ Fitness function {}/{} ============'
                  .format(ff_index+1, len(self.fitness_functions)))
            self.__execute(ff_index, num_executions, num_iterations)

class OptmizationAlgorithm:
    def __init__(self, name):
        self.solution = None
        self.best_fitness = -np.inf
        self.name = name
    def reset(self):
        raise NotImplementedError()
    def optimize(self, fitness_function, num_iterations):
        raise NotImplementedError()
    
import time

class Comparator:
    def __init__(self, *algorithms):
        self.algorithms = algorithms
        self.data = {
            a.name: {
                'fitness': [], #best fitness of each exectution
                'solutions': [], #maximum found in each execution
                'time': [] #time spent in each execution
            }
            for a in self.algorithms
        }
        self.fitness_list = [[] for _ in algorithms]
    def optimize(self, num_iterations):
        for a in self.algorithms:
            start_time = time.time()
            a.optimize(num_iterations)
            end_time = time.time()
            a_data = self.data[a.name]
            a_data['fitness'].append(a.best_fitness)
            a_data['solutions'].append(a.solution)
            a_data['time'].append(end_time - start_time)

    def execute(self, num_executions, num_iterations):
        for i in range(num_executions):
            for a in self.algorithms:
                a.reset()
            self.optimize(num_iterations)
            print(str(100*(i+1)/num_executions)+' %')
            
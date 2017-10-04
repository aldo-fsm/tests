import numpy as np

def gbest_multipoint_test(fitnessFunction, popSize, nVar, c1, c2, initBounds, inertia=1, convergenceVelocity=0):
    iteration = 0
    positions = np.random.uniform(initBounds[0], initBounds[1], (popSize, nVar))
    velocities = np.zeros([popSize, nVar])
    convergencePoints = []
    converged = False
    while True:
        fitness = np.array([fitnessFunction(p) for p in positions])
        gbestIndex = np.argmax(fitness)
        gbest = positions[gbestIndex]

        yield {
            'positions':positions,
            'velocities': velocities,
            'gbestIndex': gbestIndex,
            'fitness':fitness,
            'convergencePoints': convergencePoints
            }

        print('Iteration: {}'.format(iteration))
        print('gbest: {0} ( {1} )'.format(fitness[gbestIndex], gbest))
        for (p, f) in zip(positions, fitness):
            print("  {0} ..... {1}".format(p, f))

        if iteration == 0:
            bestPositions = positions
            bestFitness = fitness
        else:
            bestPositions = np.array([positions[i] if fitness[i] > bestFitness[i]
                                    else bestPositions[i] for i in range(len(positions))])
            bestFitness = np.array([fitness[i] if fitness[i] > bestFitness[i]
                                    else bestFitness[i] for i in range(len(fitness))])
            if np.mean([np.linalg.norm(v) for v in velocities]) <= convergenceVelocity:
                converged = True
                convergencePoints.append(np.array(positions[gbestIndex]))
        r1 = np.random.uniform(0,1,(popSize, nVar))
        r2 = np.random.uniform(0,1,(popSize, nVar))
        velocities = inertia*velocities + c1*r1*(bestPositions-positions)+c2*r2*(gbest-positions)
        positions += velocities
        if converged: 
            velocities += np.random.normal(positions, 10, (popSize, nVar))-positions
            positions += velocities
            bestFitness = np.zeros(popSize)-np.inf
            converged = False
        iteration+=1
        
def gbest(fitnessFunction, popSize, nVar, c1, c2, initBounds, inertia=1):
    iteration = 0
    positions = np.random.uniform(initBounds[0], initBounds[1], (popSize, nVar))
    velocities = np.zeros([popSize, nVar])
    while True:
        fitness = np.array([fitnessFunction(p) for p in positions])
        gbestIndex = np.argmax(fitness)
        gbest = positions[gbestIndex]

        yield {
            'positions':positions,
            'velocities': velocities,
            'gbestIndex': gbestIndex,
            'fitness':fitness
            }

        print('Iteration: {}'.format(iteration))
        print('gbest: {0} ( {1} )'.format(fitness[gbestIndex], gbest))
        for (p, f) in zip(positions, fitness):
            print("  {0} ..... {1}".format(p, f))

        if iteration == 0:
            bestPositions = positions
            bestFitness = fitness
        else:
            bestPositions = np.array([positions[i] if fitness[i] > bestFitness[i]
                                    else bestPositions[i] for i in range(len(positions))])
            bestFitness = np.array([fitness[i] if fitness[i] > bestFitness[i]
                                    else bestFitness[i] for i in range(len(fitness))])
        r1 = np.random.uniform(0,1,(popSize, nVar))
        r2 = np.random.uniform(0,1,(popSize, nVar))
        velocities = inertia*velocities + c1*r1*(bestPositions-positions)+c2*r2*(gbest-positions)
        positions += velocities
        iteration+=1

class Pso:
    def __init__(self, fitness_function, num_var, c1, c2, inertia):
        self.iteration = 0
        self.fitness_function = fitness_function
        self.num_var = num_var
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.population = []
        self.swarms = []
    def init_particles(self, num_particles, local_to_global_ratio, normal_mean=0, normal_sd=1):
        self.init_particles_args = num_particles, local_to_global_ratio, normal_mean, normal_sd
        swarm_size = round(num_particles*local_to_global_ratio)
        self.swarms = [Swarm(swarm_size, self.num_var, normal_mean, normal_sd)
                           for _ in range(int(np.floor(num_particles/swarm_size)))]
        if swarm_size*len(self.swarms) < num_particles:
            self.swarms.append(Swarm(num_particles%swarm_size, self.num_var, normal_mean, normal_sd))
        self.population = np.concatenate([s.particles for s in self.swarms])
    def reset(self):
        self.iteration = 0
        self.init_particles(*self.init_particles_args)

    def optimize(self, iterations=1):
        self.iteration += iterations
        for _ in range(iterations):
            # update pbest
            for p in self.population:
                p.update_pbest(self.fitness_function)
            # update lbest
            for s in self.swarms:
                s.update_gbest()
            # update velocity and position 
            for s in self.swarms:
                lbest = s.gbest[0].position
                for p in s.particles:
                    pbest = p.pbest[0]
                    r1, r2 = np.random.rand(2, self.num_var)
                    p.velocity = p.velocity*self.inertia + \
                                 self.c1*r1*(pbest-p.position) + \
                                 self.c2*r2*(lbest-p.position)
                    p.position = p.position + p.velocity
        self.solution, self.best_fitness = self.gbest()
    def gbest(self):
        fitness = [p.pbest[1] for p in self.population]
        bestIndex = np.argmax(fitness)
        return self.population[bestIndex].pbest
class Swarm:
    def __init__(self, num_particles, num_var, normal_mean=0, normal_sd=1):
        positions = np.random.normal(normal_mean, normal_sd, (num_particles, num_var))
        velocities = np.random.normal(normal_mean, normal_sd, (num_particles, num_var))
        self.particles = np.array([Particle(p, v) for p, v in zip(positions, velocities)])
        self.gbest = (positions[0], -np.inf)
    def update_gbest(self):
        fitness = [p.fitness for p in self.particles]
        bestIndex = np.argmax(fitness)
        self.gbest = self.particles[bestIndex], fitness[bestIndex]
class Particle:
    def __init__(self, initial_position, initial_velocity):
        self.position = initial_position
        self.velocity = initial_velocity
        self.fitness = -np.inf
        self.pbest = (initial_position, -np.inf)
    def update_pbest(self, fitness_function):
        self.fitness = fitness_function(self.position)
        if self.fitness > self.pbest[1]:
            self.pbest = self.position, self.fitness

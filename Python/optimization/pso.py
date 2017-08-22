import numpy as np

def gbest(fitnessFunction, popSize, nVar, c1, c2, bounds, inertia=1):
    iteration = 0
    positions = np.random.uniform(bounds[0], bounds[1], (popSize, nVar))
    velocities = np.zeros([popSize, nVar])    
    while True:
        fitness = np.array([fitnessFunction(p) for p in positions])
        gbestIndex = np.argmax(fitness)
        gbest = positions[gbestIndex]

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
        r1 = np.random.uniform(0,1,(popSize, 2))
        r2 = np.random.uniform(0,1,(popSize, 2))
        velocities = inertia*velocities + c1*r1*(bestPositions-positions)+c2*r2*(gbest-positions)
        positions += velocities
        yield positions
        iteration+=1
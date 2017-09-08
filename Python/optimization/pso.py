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
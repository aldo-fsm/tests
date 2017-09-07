from sympy import *
from matplotlib import pyplot as plt
import numpy as np
import itertools as it


def gbest_pso(fitnessFunction, popSize, nVar, c1, c2, initBounds, error_list, inertia=1):
    iterations = 50
    positions = np.random.uniform(initBounds[0], initBounds[1], (popSize, nVar))
    velocities = np.zeros([popSize, nVar])    
    for i in range(iterations):
        fitness = np.array([fitnessFunction(p) for p in positions])
        gbestIndex = np.argmax(fitness)
        gbest = positions[gbestIndex]
        error_list.append(-fitness[gbestIndex])
        print('Iteration: {}'.format(i))
        print('gbest: {0} ( {1} )'.format(fitness[gbestIndex], gbest))
        for (p, f) in zip(positions, fitness):
            print("  {0} ..... {1}".format(p, f))

        if i == 0:
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
    return positions[np.argmax(fitness)]

def gradient_descent(gradient, errorL, number_params, data, error_list):
    iterations = 600
    delta = 0.01
    threshold = 5
    solution = np.random.randn(number_params)
    inputs, targets = data
    for _ in range(iterations):
        current_params = solution
        e = np.mean(errorL(inputs, targets, *solution))
        gradient_vector = np.array([np.mean(g(inputs, targets, *current_params)) for g in gradient])
        gradient_norm = np.linalg.norm(gradient_vector)
        if(gradient_norm > threshold):
            gradient_vector *= threshold/gradient_norm
        error_list.append(e)
        print(e)
        for i in range(len(solution)):
            solution[i] -= delta*gradient_vector[i]
    return solution
def fitness_function(particle):
    f=-np.mean(errorL(inputs, targets, *particle))
    return f

x, t = symbols('x t')
number_params = 3
a = list(it.islice(numbered_symbols('a'),0,number_params))

y = a[2]*sin(x*a[1]) + a[0]
# y = a[2]*x**2 + a[1]*x + a[0]

inputs = np.linspace(-4,4,10)
# targets = np.array([4.5, 0.9, 0.3, 1.2, 3.9])
targets = np.sin(inputs)

error = 0.5*(y-t)**2
errorL = np.vectorize(lambdify([x,t,*a],error,np))
error_list = []
# gradient = map(lambda a : diff(error, a), a)
# gradient = list(map(lambda f : np.vectorize(lambdify([x, t, *a], f, np)), gradient))

# solution = gradient_descent(gradient, errorL, len(a), (inputs, targets), error_list)
# print(solution)

solution = gbest_pso(fitness_function, 30, len(a), 1, 1, [-1,1], error_list, inertia=0.3)
print(solution)
f = np.vectorize(lambdify(x, y.subs(zip(a, solution))))
t = np.linspace(-5,5,1000)
plt.plot(error_list)
plt.show()
plt.plot(t,f(t))
plt.scatter(inputs,targets)
plt.show()
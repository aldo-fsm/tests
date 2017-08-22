from sympy import *
from matplotlib import pyplot as plt
import numpy as np
import itertools as it

x, t = symbols('x t')
number_params = 3
a = list(it.islice(numbered_symbols('a'),0,number_params))

y = a[2]*x**2 + a[1]*x + a[0]

iterations = 600
delta = 0.01
threshold = 5

inputs = np.array([-2, -1, 0, 1, 2])
targets = np.array([4.5, 0.9, 0.3, 1.2, 3.9])

error = 0.5*(y-t)**2
errorL = np.vectorize(lambdify([x,t,*a],error,np))
gradient = map(lambda a : diff(error, a), a)
gradient = list(map(lambda f : np.vectorize(lambdify([x, t, *a], f, np)), gradient))

solution = np.random.randn(len(a))
error_list = []
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
print(solution)
f = np.vectorize(lambdify(x, y.subs(zip(a, solution))))
t = np.linspace(-5,5,1000)
plt.plot(error_list)
plt.show()
plt.plot(t,f(t))
plt.scatter(inputs,targets)
plt.show()
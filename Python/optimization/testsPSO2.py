from comparator import Comparator
from pso import Pso
import numpy as np
from matplotlib import pyplot as plt

f = lambda x, y: ((y+47)*np.sin(np.sqrt(np.abs(x/2+y+47)))+x*np.sin(np.sqrt(np.abs(x-y-47))))*(1-np.sign(x-512))*(1+np.sign(x+512))*(1-np.sign(y-512))*(1+np.sign(y+512))

pso1 = Pso(lambda x: f(*x), 2, 2, 2, 0.8)
pso1.init_particles(100, 1, 0, 1)
pso1.name = 'pso1'

pso2 = Pso(lambda x: f(*x), 2, 2, 2, 0.8)
pso2.init_particles(100, 0.1, 0, 1)
pso2.name = 'pso2'

comp = Comparator(pso1, pso2)
comp.execute(100,100)

pso1_data, pso2_data = comp.data['pso1'], comp.data['pso2']
plt.grid()
plt.boxplot([pso1_data['fitness'], pso2_data['fitness']])
plt.show()

plt.boxplot([pso1_data['time'], pso2_data['time']])
plt.show()

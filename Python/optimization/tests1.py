import re
from matplotlib import animation
import time
import pso
import geneticAlgorithms as ga
import numpy as np
from matplotlib import pyplot as plt

f = lambda x,y: x**2*np.sin(y)+y**2*np.sin(x)+x**2+y**2

numBits = 8
numPossibleValues = 2**numBits
xBounds = [-10, 10]
yBounds = [-10, 10]

xq = np.arange(xBounds[0], xBounds[1],(xBounds[1]-xBounds[0])/numPossibleValues)
yq = np.arange(yBounds[0], yBounds[1], (yBounds[1]-yBounds[0])/numPossibleValues)

def costFunction(chromossome):
    return f(*decode(chromossome))

# pylint: disable=unbalanced-tuple-unpacking
def decode(chromossome):
    xBits, yBits = np.split(chromossome, [numBits])
    xBinStr = re.sub(r'\D', '', str(xBits))
    yBinStr = re.sub(r'\D', '', str(yBits))
    i, j = int(xBinStr, 2), int(yBinStr, 2)
    return xq[i], yq[j]

def update(i):
    txtIt.set_text('iteration: {}'.format(i))
    
    popPSO = next(psoGen)
    offsetsPSO = [p for p in popPSO]
    sctPSO.set_offsets(list(zip(offsetsPSO)))
    
    popGA = next(gaGen)
    offsetsGA = []
    for chrom in popGA:
        offsetsGA.append(decode(chrom))
    sctGA.set_offsets(list(zip(offsetsGA)))
    
    # return sctPSO,

gaGen = ga.optimize(costFunction, 2*numBits, 30, 0.3, 0.1, pairing='rank_weighting', crossover='double_point')
psoGen = pso.gbest(lambda x: -f(*x), 30, 2, 1, 1, [-10,10], inertia=0.6)

x, y = np.meshgrid(xq, yq)
fig, ax = plt.subplots()
ax.pcolormesh(x, y, f(x, y))
sctGA = ax.scatter([],[],color='g')
sctPSO = ax.scatter([],[],color='r')
txtIt = fig.text(0.15,0.03,'')

anim = animation.FuncAnimation(fig, update, interval=10, repeat=False)
# anim.save('animations/PSO-{}.mp4'.format(time.ctime()), bitrate=7200, dpi=200, fps=24)
plt.show()
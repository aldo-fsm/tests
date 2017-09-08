import numpy as np
import pso
import time
from matplotlib import pyplot as plt, animation

gbestList = []
meanList = []
meanVelocityList = []
def update(i):
    global gbestList, meanList, ax2
    txtIt.set_text('iteration: {}'.format(i))
    results = next(psoGen)
    gbestIndex = results['gbestIndex']
    fitnessMean = np.mean(results['fitness'])
    bestFitness = results['fitness'][gbestIndex]
    gbestList.append(bestFitness)
    meanList.append(fitnessMean)
    meanVelocityList.append(np.mean([np.linalg.norm(v) for v in results['velocities']]))
    meanLines.set_data(range(len(meanList)), meanList)
    gbestLines.set_data(range(len(gbestList)), gbestList)
    meanVelocityLines.set_data(range(len(gbestList)), meanVelocityList)
    sctPSO.set_offsets(results['positions'])
    sctFound.set_offsets(results['convergencePoints'])
    print(results['convergencePoints'])
    ax2.relim()
    ax2.autoscale_view()
f = lambda x,y: (np.sin(x+y)+np.sin(x-y))*(1-np.sign(x-10))*(1+np.sign(x+10))*(1-np.sign(y-10))*(1+np.sign(y+10))

psoGen = pso.gbest_multipoint_test(lambda x: f(*x), 30, 2, 1, 1, [-10,10], inertia=0.6, convergenceVelocity=0.7)

x, y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.pcolormesh(x, y, f(x, y))
sctPSO = ax1.scatter([],[], color='b')
sctFound = ax1.scatter([], [], color='y', marker='x')
meanLines, gbestLines, meanVelocityLines = ax2.plot([],[],[],[], [], [])
ax2.margins(x=0.1,y=0.1)
ax2.grid()
txtIt = fig.text(0.15,0.03,'')

anim = animation.FuncAnimation(fig, update, interval=10, repeat=False, frames=2000)
# anim.save('animations/PSO-multitarget-test{}.mp4'.format(time.ctime()), bitrate=7200, dpi=200, fps=24)
plt.show()
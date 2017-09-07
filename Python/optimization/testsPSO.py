import numpy as np
import pso
from matplotlib import pyplot as plt, animation

gbestList = []
meanList = []
def update(i):
    global gbestList, meanList, ax2
    txtIt.set_text('iteration: {}'.format(i))
    results = next(psoGen)
    gbestIndex = results['gbestIndex']
    fitnessMean = np.mean(results['fitness'])
    bestFitness = results['fitness'][gbestIndex]
    
    gbestList.append(bestFitness)
    meanList.append(fitnessMean)

    meanLines.set_data(range(len(meanList)), meanList)
    gbestLines.set_data(range(len(gbestList)), gbestList)
    sctPSO.set_offsets(results['positions'])
    ax2.relim()
    ax2.autoscale_view()
f = lambda x,y: np.sin(x+y)+np.sin(x-y)

psoGen = pso.gbest(lambda x: f(*x), 30, 2, 1, 1, [-10,10], inertia=0.6)

x, y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
fig, (ax1, ax2) = plt.subplots(1,2)
ax1.pcolormesh(x, y, f(x, y))
sctPSO = ax1.scatter([],[],color='b')
meanLines, gbestLines = ax2.plot([],[],[],[])
ax2.margins(x=0.1,y=0.1)
ax2.grid()
txtIt = fig.text(0.15,0.03,'')

anim = animation.FuncAnimation(fig, update, interval=10, repeat=False)
# anim.save('animations/PSO-{}.mp4'.format(time.ctime()), bitrate=7200, dpi=200, fps=24)
plt.show()
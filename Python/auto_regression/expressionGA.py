import numpy as np
from sympy import *
from random import choice
from collections import deque

nextSymbolId = 0

def evolveExpr(fitnessFunction, popSize, numVar, initialTreeSize, selRate, mutRate):
    funcPool = [(Add, 2), (Mul, 2), (Pow, 2), (sin, 1), (cos, 1), (log, 1)]
    valuesPool = [None]#[None, pi, E]
    valuesPool.extend(symbols(['x'+str(i) for i in range(numVar)]))
    
    popKeepSize = round(selRate*popSize)
    numPairs = np.ceil((popSize-popKeepSize)/2)
    numMutation = np.ceil(mutRate*popSize)
    pop = [createExpr(initialTreeSize, funcPool, valuesPool) for _ in range(popSize)]

    generation = 0
    while True:
        fitness = np.array([fitnessFunction(expr) for expr in pop])
        sortIndexes = np.argsort(fitness*-1)
        pop = np.array(pop)[sortIndexes]
        fitness = fitness[sortIndexes]
        popKeep = pop[:popKeepSize]
        print('\n------------- Generation: {} -------------'.format(generation))
        for e, f in zip(pop, fitness):
            print('{0} .............. {1}'.format(e, f))
        parents1, parents2 = selectPairs(popKeep, fitness, numPairs)
        children = [crossover(p1, p2) for p1, p2 in zip(parents1,parents2)]
        pop = np.concatenate([popKeep, *children])[:popSize]
        yield pop
        pop = mutation(pop, numMutation, initialTreeSize, funcPool, valuesPool)
        generation+=1

def selectPairs(popKeep, fitness, numPairs):
    popKeepSize = len(popKeep)
    numPairs = int(numPairs)
    ranks = list(range(popKeepSize))
    ranksSum = np.sum(ranks)
    prob = [(popKeepSize-i-1)/ranksSum for i in ranks]
    cumProb = np.cumsum(prob)
    random1 = np.random.rand(numPairs)
    random2 = np.random.rand(numPairs)
    p1 = [argfirst(lambda n : n > r, cumProb) for r in random1]
    p2 = [argfirst(lambda n : n > r, cumProb) for r in random2]
    for i in range(len(p1)):
        if p1[i] == p2[i] :
            p2[i] = np.random.choice(popKeepSize)
    return (popKeep[p1], popKeep[p2])
def crossover(p1, p2):
    crossoverPoint1 = argRandomBranch(p1)
    crossoverPoint2 = argRandomBranch(p2)
    child1 = setNode(p1, crossoverPoint1, randomBranch(p2))
    child2 = setNode(p2, crossoverPoint2, randomBranch(p1))
    return child1, child2

def mutation(pop, numMutations, maxSize, funcPool, valuesPool):
    popSize = len(pop)
    for i in np.random.choice(popSize-1, int(numMutations))+1:
        mutationPoint = argRandomBranch(pop[i])
        pop[i] = setNode(pop[i], mutationPoint, createExpr(maxSize, funcPool, valuesPool))
    return pop

def createExpr(maxSize, funcPool, valuesPool):
    if np.random.choice(2) == 0 or maxSize <= 1:  
        value = choice(valuesPool)
        if value:
            return value
        else :
            global nextSymbolId
            newValue = Symbol('v'+str(nextSymbolId))
            valuesPool.append(newValue)
            nextSymbolId += 1
            return newValue
    else:
        func, numArgs = choice(funcPool)
        return func(*[createExpr(maxSize-1, funcPool, valuesPool)
                      for _ in range(numArgs)])

def randomBranch(node):
    aux = np.arange(height(node)+1)+1
    prob = aux/np.sum(aux)
    cumProb = np.cumsum(prob)
    r = np.random.rand()
    maxHeight = argfirst(lambda p: p > r, cumProb)
    while maxHeight > 0 and node.args:
        node = choice(node.args)
        maxHeight -= 1
    return node

def argRandomBranch(node):
    aux = np.arange(height(node)+1)+1
    prob = aux/np.sum(aux)
    cumProb = np.cumsum(prob)
    r = np.random.rand()
    maxHeight = argfirst(lambda p: p > r, cumProb)
    indexList = []
    while maxHeight > 0 and node.args:
        nodeIndex = np.random.choice(len(node.args))
        node = node.args[nodeIndex]
        indexList.append(nodeIndex)
        maxHeight -= 1
    return indexList

def getNode(expr, indexList):
    if type(indexList) != deque:
        indexList = deque(indexList)
    node = expr
    while indexList:
        node = node.args[indexList.popleft()]
    return node

def setNode(expr, indexList, node):
    if type(indexList) != deque:
        indexList = deque(indexList)
    if indexList and expr.args:
        index = indexList.popleft()
        args = expr.args
        return expr.func(*[setNode(args[i], indexList, node) if i == index 
                        else args[i] for i in range(len(args))])
    else:
        return node

def height(node):
    if node.args:
        return 1+max([height(n) for n in node.args])
    else:
        return 0

def argfirst(condition, iterable):
    return next(x[0] for x in enumerate(iterable) if condition(x[1]))


# init_printing()
# for e in evolveExpr(10,2,5,0,0):
#     print(e)
x,y,z = symbols('x y z')
e = x+y*sin(z)
print(e)
print(srepr(e))
# print(height(e))
# print(argRandomBranch(e))
print(getNode(e,[1,1,0]))

#1ºteste - aproximar targetValue
def testFitness(expr):
    targetValue = np.pi
    for a in expr.free_symbols:
        expr = expr.subs(a,1)
    try:
        return np.float(-(expr.evalf()-targetValue)**2)
    except:
        return -np.inf
ga = evolveExpr(testFitness, 30, 2, 3, 0.5, 0.2)
for _ in range(200):
    pop = next(ga)

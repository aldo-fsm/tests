import numpy as np
from sympy import *
from random import choice
from collections import deque

nextSymbolId = 0

def evolveExpr(popSize, numVar, initialTreeSize, selRate, mutRate):
    funcPool = [(Add, 2), (Mul, 2), (Pow, 2), (sin, 1), (cos, 1), (log, 1)]
    valuesPool = [None, pi, E]
    valuesPool.extend(symbols(['x'+str(i) for i in range(numVar)]))
    pop = [createExpr(initialTreeSize, funcPool, valuesPool) for _ in range(popSize)]
    # 1) fitness
    # 2) selection
    # 3) crossover
    # 4) mutation
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

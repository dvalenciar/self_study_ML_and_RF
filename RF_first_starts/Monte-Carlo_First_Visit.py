
import numpy as np
import random
'''
             matrix 4x4

       | ///|    1|    2|    3|
       |   4|    5|    6|    7|
       |   8|    9|   10|   11|
       |  12|   13|   14|  ///|

                states

       | 0,0|   0,1|   0,2|  0,3|
       | 1,0|   1,1|   1,2|  1,3|
       | 2,0|   2,1|   2,2|  2,3|
       | 3,0|   3,1|   3,2|  3,3|

                   Up=[-1,0]
                       | 
       Left=[0,-1]<--- |  ---> Right=[0,1]
                       |
                  Down = [1,0]
                  
                  
        EL RESULTADO DE ESTO ES V ..DE VALUE FUNCTION FOR EACH STATE

       | V(0,0)|   V(0,1)|   V0,2)|  V(0,3)|
       | V(1,0)|   V(1,1)|   V1,2)|  V(1,3)|
       | V(2,0)|   V(2,1)|   V2,2)|  V(2,3)|
       | V(3,0)|   V(3,1)|   V3,2)|  V(3,3)|

Reward = -1 for each transitions
action = up, down, left, right
'''

# parameters
gamma = 0.6  # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [[0, 0], [3, 3]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000

# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j):list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


def generateEpisode():

    initState = random.choice(states[1:-1])
    episode = []

    while True:
        if list(initState) in terminationStates:
            return episode
        action = random.choice(actions)
        finalState = np.array(initState)+np.array(action)
        if -1 in list(finalState) or gridSize in list(finalState):
            finalState = initState
        episode.append([list(initState), action, rewardSize, list(finalState)])
        initState = finalState


for it in range(numIterations):
    episode = generateEpisode()
    G = 0
    for i, step in enumerate(episode[::-1]):
        G = gamma*G + step[2]   # G --> gamma*G + Rt+1

        if step[0] not in [x[0] for x in episode[::-1][len(episode)-i:]]:
            idx = (step[0][0], step[0][1])
            returns[idx].append(G)
            newValue = np.average(returns[idx])
            deltas[idx[0], idx[1]].append(np.abs(V[idx[0], idx[1]]-newValue))
            V[idx[0], idx[1]] = newValue

print(V)
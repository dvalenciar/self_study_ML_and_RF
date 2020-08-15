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

'''

# parameters
gamma = 0.5  # discounting rate
rewardSize = -1
gridSize = 4
alpha = 0.5  # (0,1] // stepSize
terminationStates = [[0, 0], [3, 3]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 10000

# initialization
V = np.zeros((gridSize, gridSize))
returns = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
deltas = {(i, j): list() for i in range(gridSize) for j in range(gridSize)}
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


def generateInitialState():
    initState = random.choice(states[1:-1])
    return initState

def generateNextAction():
    return random.choice(actions)

def takeAction(state, action):
    if list(state) in terminationStates:
        return 0, None
    finalState = np.array(state)+np.array(action)
    # if robot crosses wall
    if -1 in list(finalState) or gridSize in list(finalState):
        finalState = state
    return rewardSize, list(finalState)


for it in range(numIterations):

    state = generateInitialState()
    while True:
        action = generateNextAction()
        reward, finalState = takeAction(state, action)
        # we reached the end
        if finalState is None:
            break

        # modify Value function
        before = V[state[0], state[1]]
        V[state[0], state[1]] += alpha * (reward + gamma * V[finalState[0], finalState[1]] - V[state[0], state[1]])
        deltas[state[0], state[1]].append(float(np.abs(before - V[state[0], state[1]])))

        state = finalState

print(V)
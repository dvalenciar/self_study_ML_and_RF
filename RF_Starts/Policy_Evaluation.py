import numpy as np


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

Reward = -1 for each transitions
action = up, down, left, right
policy = 0.25 uniform random policy     
'''
gamma = 1  # discounting rate
rewardSize = -1
gridSize = 4
terminationStates = [[0, 0], [3, 3]]
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
numIterations = 1000

# Initialization
valueMap = np.zeros((gridSize, gridSize))
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


def actionRewardFunction(initialPosition, action):

    if initialPosition in terminationStates:
        return initialPosition, 0

    reward = rewardSize
    finalPosition = np.array(initialPosition) + np.array(action)
    if -1 in finalPosition or 4 in finalPosition:
        finalPosition = initialPosition
    return finalPosition, reward


deltas = []
for it in range(numIterations):
    copyValueMap = np.copy(valueMap)
    deltaState = []
    for state in states:
        weightedRewards = 0
        for action in actions:
            finalPosition, reward = actionRewardFunction(state, action)
            #print("state", state, "action", action, "next_state", finalPosition, "reward", reward)
            weightedRewards += 0.25 * (reward+(gamma*valueMap[finalPosition[0], finalPosition[1]]))
        deltaState.append(np.abs(copyValueMap[state[0], state[1]]-weightedRewards))
        copyValueMap[state[0], state[1]] = weightedRewards
    deltas.append(deltaState)
    valueMap = copyValueMap

    if it in [0, 1, 2, 9, 99, 999, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")



"""
Aqui se resuelve con Q-Learning el problema del
Taxi V3.

----------Environment Notes----------

There are 6 discrete actions
Action space is [0,1,2,3,4,5]
0= move south  1= move north
2= move east   3= move west
4= pick up passenger   5= dropoff passenger

Observation space
There are 500 discretes states

Passenger Location:
0:R    2:Y   4:in taxi
1:G    3:B

Destionation:
0:R    2:Y
1:G    3:B

There is a reward of -1 for each action
Reward of 20 for delivery the passenger
Reward of -10 for illegal pickup or dropoff


Once the passenger is dropped off, the episode ends
Also after 200 steps the episode ends
"""

import gym
import math
import  numpy as np
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

#--------- Q Learning settings -----#

DISCOUNT  = 0.99 # gamma
min_alpha = 0.01
min_eps   = 0.01
eps_decay_rate = 0.001

EPISODES  = 100_000

Q_table = np.random.uniform(low=-1, high=1, size=(env.observation_space.n, env.action_space.n)) #matriz 500X6


#--------------- Functions to get alpha and epsilon--------------------#

def get_alpha(t):
    change = max ( min_alpha, min(1.0, 1.0 - math.log10((t + 1) /25)))
    return change

def get_epsilon(t):
    change2 = max ( min_eps, min(1.0, 1.0 - math.log((t+1) /25)))
    return change2

def get_epsilon2(t):
    change3 = min_eps + (1 - min_eps) * np.exp(-eps_decay_rate * t)
    return change3


total_reward = []
total_steps  = []
total_penalities = []

total_epsilon = []
total_steps_grap = []
total_reward_grap = []
total_penalities_grap = []

ave_steps = []
ave_reward = []
ave_penalities = []


for i in range (1, EPISODES):

    state = env.reset()

    episilon = get_epsilon(i)
    #episilon = get_epsilon2(i)
    LEARNING_RATE = get_alpha(i)


    total_epsilon.append(episilon) # just to plot

    done = False

    steps, penalities, reward_per_episode = 0, 0, 0

    while not done:

        if np.random.random() <= episilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[state])

        next_state, reward, done, info = env.step(action)

        oldvalue = Q_table[state, action]
        next_max = np.max(Q_table[next_state])
        new_value = (1-LEARNING_RATE) * oldvalue + LEARNING_RATE * (reward + DISCOUNT * next_max)
        Q_table[state, action] = new_value
        state = next_state

        if reward == -10:
           penalities += 1

        steps += 1
        reward_per_episode += reward

    total_steps.append(steps)
    total_penalities.append(penalities)
    total_reward.append(reward_per_episode)

    # solo con propositos de graficar
    total_steps_grap.append(steps)
    total_penalities_grap.append(penalities)
    total_reward_grap.append(reward_per_episode)

    if (i) % 100 == 0:
        print("Training...", f"Episode: {i}\n")

    if (i+1) % 100 == 0:
        # cada 100 episodios hago eso para poder graficas mejor
        step_average = np.mean(total_steps_grap)
        ave_steps.append((step_average))
        total_steps_grap = []

        rew_average = np.mean(total_reward_grap)
        ave_reward.append((rew_average))
        total_reward_grap = []

        penal_average = np.mean(total_penalities_grap)
        ave_penalities.append((penal_average))
        total_penalities_grap = []

        #env.render()

np.save('Q_table_Taxi_V3.npy', Q_table)

print ("Training Finished.\n")
print ("====+====+====+====+===+====+====+====+====+====+====+.\n")
print (f"Results After {EPISODES} Episodes:.\n", f"Average timesteps  per episode: {np.mean(total_steps)}\n", f"Average penalities per episode: {np.mean(total_penalities)}")
print (f"Average Reward per episode: {np.mean(total_reward)}")


env.close()

#Plot information:
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(total_steps)), total_steps)
plt.ylabel('Steps')
plt.title(' Taxi Environment  Using Q- Learning')

plt.subplot(3, 1, 2)
plt.plot(np.arange(len(total_penalities)), total_penalities)
plt.ylabel('Penalties')

plt.subplot(3, 1, 3)
plt.plot(np.arange(len(total_reward)), total_reward)
plt.ylabel('Rewards')
plt.xlabel('Episodes')


plt.figure(2)
plt.subplot(3, 1, 1)
plt.plot(100*(np.arange(len(ave_steps)) + 1), ave_steps)
plt.ylabel('Average Steps')
plt.title(' Taxi Environment  Using Q- Learning')

plt.subplot(3, 1, 2)
plt.plot(100*(np.arange(len(ave_penalities))+1), ave_penalities)
plt.ylabel('Average Penalties')

plt.subplot(3, 1, 3)
plt.plot(100*(np.arange(len(ave_reward)) +1), ave_reward)
plt.ylabel('Average Rewards')
plt.xlabel('Episodes')

'''
plt.figure(3)
plt.plot(np.arange(len(total_epsilon)), total_epsilon)
plt.xlabel('Epsides')
plt.ylabel('Epsilon')
plt.title(' Epsilon Decay')
'''
plt.show()


# Despues de entrenar el agente usando Q-Learning generamos la Q-table que es usada aqui
# El archivo donde se genera la Q-table se llama QL_CartPole_v0.py

import gym
import math
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make("CartPole-v0")
Q_table = np.load('Q_table_CartPole_v0.npy')

EPISODES = 100
total_reward = []


def discretize(obs):
   buckets = (1, 1, 6, 12)
   upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
   lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
   ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
   new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
   new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
   return tuple(new_obs)


for i in range(EPISODES):
    state = env.reset()
    discrete_state = discretize(state)
    done = False
    reward_per_episode = 0

    while not done:

        action = np.argmax(Q_table[discrete_state])
        next_state, reward, done, info = env.step(action)

        new_discrete_state = discretize(next_state)
        discrete_state = new_discrete_state
        reward_per_episode += reward

        env.render()
        #time.sleep(1)

    total_reward.append(reward_per_episode)

print ("====+====+====+====+===+====+====+====+====+====+====+.\n")
print (f"Results After {EPISODES} Episodes:.\n", f"Average Reward per episode: {np.mean(total_reward)}")

env.close()

#  Plot information:

plt.figure(1)
plt.ylabel('Steps')
plt.title(' CartPole Environment  Using Q- Learning')
plt.plot(np.arange(len(total_reward)), total_reward)
plt.ylabel('Rewards')
plt.xlabel('Episodes')
plt.show()
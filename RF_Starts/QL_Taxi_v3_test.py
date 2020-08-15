# Despues de entrenar el agente usando Q-Learning generamos la Q-table que es usada aqui
# El archivo donde se genera la Q-table se llama QL_Taxi_V3.py

import gym
import numpy as np
import matplotlib.pyplot as plt
import time

env = gym.make("Taxi-v3")

Q_table  = np.load('Q_table_Taxi_V3.npy')

EPISODES = 100
total_reward = []
total_steps  = []
total_penalties = []

for i in range(EPISODES):
    state = env.reset()
    done = False
    steps, penalties, reward_per_episode = 0, 0, 0
    while not done:
        action = np.argmax(Q_table[state])
        state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1
        steps += 1
        reward_per_episode += reward
        #env.render()
        #time.sleep(1)

    total_steps.append(steps)
    total_penalties.append(penalties)
    total_reward.append(reward_per_episode)

print (f"Results After {EPISODES} Episodes:.\n", f"Average timesteps  per episode: {np.mean(total_steps)}\n", f"Average penalties per episode: {np.mean(total_penalties)}")
print (f"Average Reward per episode: {np.mean(total_reward)}")

#Plot information:
plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(np.arange(len(total_steps)), total_steps)
plt.ylabel('Steps')
plt.title(' Taxi Environment  Using Q- Learning')

plt.subplot(3, 1, 2)
plt.plot(np.arange(len(total_penalties)), total_penalties)
plt.ylabel('Penalties')

plt.subplot(3, 1, 3)
plt.plot(np.arange(len(total_reward)), total_reward)
plt.ylabel('Rewards')
plt.xlabel('Episodes')

plt.show()
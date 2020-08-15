"""
Author = David Valencia
Date   = 22 - March - 2020
Notes  = Here we solve the CartPole v0 with Q-Learning
         In this file I train the model and crete the Q-Table
         Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials

----------Environment Notes----------
observations come in a four-dim vector (x, y, teta, w)

        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf

Action space is discrete [0, 1]

        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

Reward:
        Reward is 1 for every step taken, including the termination step

Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

import  gym
import math
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v0")
#env = gym.make("CartPole-v1")  # In this version the episode  ends after 500 steps


#--------- Q Learning settings -----#

DISCOUNT = 0.99  # gamma
min_alpha = 0.1
min_eps = 0.1
eps_decay_rate = 0.001
EPISODES = 1_000

total_reward = []
ave_reward_list = []
total_reward_grap = []

#Q_table = np.random.uniform(low=-1, high=1, size=([1, 1, 6, 12] + [env.action_space.n]))
Q_table = np.zeros(shape=([1, 1, 6, 12] + [env.action_space.n]))


#--------------- Functions to get alpha and epsilon--------------------#


def get_alpha(t):
    change = max(min_alpha, min(1.0, 1.0 - math.log10((t + 1) / 25)))
    return change


def get_epsilon(t):
    change2 = max(min_eps, min(1.0, 1.0 - math.log((t+1) / 25)))
    return change2


def get_epsilon2(t):
    change3 = min_eps + (1 - min_eps) * np.exp(-eps_decay_rate * t)
    return change3

# --------------- This space state is continuous is necessary discretizacion-------------------#


def discretize(obs):

   buckets = (1, 1, 6, 12)
   upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
   lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
   ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
   new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
   new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
   return tuple(new_obs)


for i in range(1, EPISODES):

    state = env.reset()
    discrete_state = discretize(state)

    epsilon = get_epsilon(i)
    LEARNING_RATE = get_alpha(i)

    done = False
    reward_per_episode = 0

    while not done:
        if np.random.random() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[discrete_state])

        next_state, reward, done, info = env.step(action)

        new_discrete_state = discretize(next_state)

        old_value = Q_table[discrete_state + (action,)]
        next_max = np.max(Q_table[new_discrete_state])
        new_Q_Value = (1-LEARNING_RATE) * old_value + LEARNING_RATE * (reward + DISCOUNT * next_max)

        Q_table[discrete_state + (action,)] = new_Q_Value

        discrete_state = new_discrete_state

        #  Reward is 1 for every step taken
        reward_per_episode += reward

        #env.render()

    total_reward.append(reward_per_episode)
    total_reward_grap.append(reward_per_episode)

    if (i+1) % 100 == 0:

        print("Training...", f"Episode: {i+1}")
        rew_average = np.mean(total_reward_grap)  # each 100 steps calculate the average
        ave_reward_list.append(rew_average)
        total_reward_grap = []

        print('Episode {} Average Reward: {}\n'.format(i + 1, rew_average))

np.save('Q_table_CartPole_v0.npy', Q_table)


print ("Training Finished.\n")
print ("====+====+====+====+===+====+====+====+====+====+====+.\n")
print (f"Results After {EPISODES} Episodes:.\n", f"Average Reward per episode: {np.mean(total_reward)}")


env.close()


#  Plot information:

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(total_reward)), total_reward)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.title(' CartPole Environment  Using Q- Learning')

plt.subplot(2, 1, 2)
plt.plot(100*(np.arange(len(ave_reward_list)) + 1), ave_reward_list, 'o-')
plt.ylabel('Average Rewards')

plt.show()



"""
Author : David Valencia
Date: 23 - March - 2020

Note:  Here we solve the CartPole v0 with Deep-Q-Learning
       Using DQL for this problem  a discretization is not necessary
       Considered solved when the average reward is greater than or equal to 200.0 over 100 consecutive trials
       in the version CartPole V1 when the average reward is greater or equal to 500

       Here I used one NN, same for prediction and for target

"""

import gym
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

EPISODE = 1000
min_eps = 0.01
eps_decay_rate = 0.995


class DQNAgent:
    def __init__(self, environment):

        self.env = environment

        # get the size of state and action
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.batch_size = 64
        self.memory = deque(maxlen=100000)
        self.train_start = 500
        self.gamma = 0.95
        self.learning_rate = 0.01

        #  Initialize the model
        self.model = self.build_nn_model()

    def build_nn_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(48, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(self.action_size, activation=tf.keras.activations.linear))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, decay=0.01), loss=tf.keras.losses.mse)
        model.summary()
        return model

    def memorize(self, s, a, r, n_s, d):
        self.memory.append((s, a, r, n_s, d))

    def get_action(self, obs, eps):

        if np.random.random() <= eps:
            acti = env.action_space.sample()
            return acti
        else:
            acti = np.argmax(self.model.predict(obs))
            return acti

    def replay(self):

        if len(self.memory) < self.train_start:
            return
        # Randomly sample mini_batch from the memory
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in mini_batch:

            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = (reward + self.gamma * np.max(self.model.predict(next_state)[0]))
            x_batch.append(state[0])
            y_batch.append(y_target[0])

            self.model.fit(np.array(x_batch), np.array(y_batch), epochs=1, verbose=0, batch_size=len(x_batch))

    def save_model(self):

        self.model.save('CartPole_model_v0.h5')


def get_epsilon(t):
    change_eps = max(min_eps, min(1.0, 1.0 - math.log((t + 1) / 25)))
    return change_eps


if __name__ == "__main__":

    env = gym.make("CartPole-v0")
    observation_space = env.observation_space.shape[0]
    agent = DQNAgent(env)

    epsilon = 1.0
    total_score = []
    total_score_list = []
    total_score_grap = []
    total_epsilon = []

    for i in range(1, EPISODE):

        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        done = False

        epsilon = get_epsilon(i)
        #if epsilon >= min_eps:
            #epsilon *= eps_decay_rate

        total_epsilon.append(epsilon)

        score = 0

        while not done:
            action = agent.get_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, observation_space])
            agent.memorize(state, action, reward, next_state, done)
            agent.replay()  # train the model
            state = next_state
            score += 1

        total_score.append(score)
        total_score_grap.append(score)

        print(f"Episode: {i}", f"Score: {score}")

        if i % 100 == 0:
            average_score = np.mean(total_score_grap)
            total_score_list.append(average_score)
            total_score_grap = []
            print(f"Episode: {i}", f" Mean survival time over last 100 episodes was {average_score}")

    agent.save_model()


    print("Training Finished.\n")
    print("====+====+====+====+===+====+====+====+====+====+====+.\n")
    print(f"Results After {EPISODE} Episodes---", f"Average Score per Episode: {np.mean(total_score)}")
    env.close()

    plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(total_score)), total_score)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(' CartPole Environment  Using Deep-Q-Learning')

    plt.subplot(2, 1, 2)
    plt.plot(100 * (np.arange(len(total_score_list)) + 1), total_score_list, 'o-')
    plt.ylabel('Average Score')

    plt.show()




















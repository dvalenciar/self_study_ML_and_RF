"""
Author : David Valencia
Date: 24 - April - 2020


Deep-Q learning version version 2,
Here I fix some parameters, also I used two identical  NN, one for prediction and the other for target

Note:  Here we solve the CartPole v0 with Deep-Q-Learning
       Using DQL for this problem  a discretization is not necessary
       Considered solved when the average reward is greater than or equal to 200.0 over 100 consecutive trials
       in the version CartPole V1 when the average reward is greater or equal to 500

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


class DQNAgent:
    def __init__(self, state_size, action_size):

        # get the size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # Hyper_parameters  for the DQN

        self.gamma = 0.98  # discount factor
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 1000

        # create a replay memory necessary for DQN
        self.memory = deque(maxlen=3_000)

        #  main model  (predict)
        self.model = self.build_nn_model()

        # target model  (target)
        self.target_model = self.build_nn_model()

        # Update the target model to be the same with the model
        self.update_target_model()

    def build_nn_model(self):
        # State is the input for the NN, the output is the Q value of each action
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(48, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(self.action_size, activation=tf.keras.activations.linear))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), loss='mean_squared_error')
        # model.summary()
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state, eps):
        # get action from model using epsilon-greedy-policy
        if np.random.random() <= eps:
            acti = random.randrange(self.action_size)
            return acti
        else:
            q_value = self.model.predict(state)
            acti = np.argmax(q_value[0])
            return acti

    def memorize(self, s, a, r, n_s, d):
        # save samples (state, action, reward, next_state, done) to the replay memory
        self.memory.append((s, a, r, n_s, d))

    def train_model_1(self):
        if len(self.memory) < self.train_start:
            return
        # Randomly sample mini_batch from the memory
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_batch, y_batch = [], []
        for state, action, reward, next_state, done in mini_batch:
            prediction = self.model.predict(state)
            if done:
                prediction[0][action] = reward
            else:
                Q_Value_next_state = self.target_model(next_state)
                target = (reward + self.gamma * np.max(Q_Value_next_state[0]))
                prediction[0][action] = target
            x_batch.append(state[0])
            y_batch.append(prediction[0])
            self.model.fit(np.array(x_batch), np.array(y_batch), epochs=1, verbose=0, batch_size=len(x_batch))

    def train_model_2(self):

        if len(self.memory) < self.train_start:
            return

        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state_update = np.zeros((min(self.batch_size, len(self.memory)), self.state_size))
        next_state_update = np.zeros((min(self.batch_size, len(self.memory)), self.state_size))

        action, reward, done = [], [], []

        for i in range(self.batch_size):
            state_update[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            next_state_update[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(state_update)
        target_val = self.target_model.predict(next_state_update)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(state_update, target, batch_size=self.batch_size, epochs=1, verbose=0)

    def save_model(self):
        self.model.save('CartPole_model_better_version.h5')


def get_epsilon(t):
    # epsilon - greedy - policy
    change_eps = max(min_eps, min(1.0, 1.0 - math.log((t + 1) / 25)))
    return change_eps


if __name__ == "__main__":

    env = gym.make("CartPole-v0")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    total_score = []

    for i in range(1, EPISODE):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        epsilon = get_epsilon(i)

        while not done:

            action = agent.get_action(state, epsilon)  # get an action
            next_state, reward, done, info = env.step(action)  # go one step in environment
            next_state = np.reshape(next_state, [1, state_size])

            # save the sample to the replay memory
            agent.memorize(state, action, reward, next_state, done)

            # every time step do the training
            agent.train_model_2()

            score += 1
            state = next_state

            if done:
                agent.update_target_model()
                total_score.append(score)
                print(f"Episode: {i}", f"Score: {score}")

        # save the model
        if i % 20 == 0:
            agent.save_model()

    print("Training Finished.\n")
    print("====+====+====+====+===+====+====+====+====+====+====+.\n")
    print(f"Results After {EPISODE} Episodes      ", f"Average Score per Episode   : {np.mean(total_score)}")
    env.close()

    plt.figure(1)
    plt.plot(np.arange(len(total_score)), total_score)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(' CartPole Environment Deep-Q-Learning(predict and target NN)')
    plt.show()

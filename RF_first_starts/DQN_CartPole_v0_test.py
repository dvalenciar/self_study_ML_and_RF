"""
 Despues de entrenar el agente usando Deep Q-Learning generamos el modelo que es usado aqui
 Los mejores resultados se alcanzaron con el modelo de double DQN
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym

env = gym.make("CartPole-v0")

model = tf.keras.models.load_model('CartPole_model_v0.h5')
#model = tf.keras.models.load_model('CartPole_model_better_version.h5')
#model = tf.keras.models.load_model('CartPole_model_double_DQN.h5')  # best performance

EPISODE = 100
total_score = []

for i in range(1, EPISODE):

    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    score = 0
    done = False

    while not done:

        action = np.argmax(model.predict(state))
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        state = next_state
        score += 1
        env.render()

    print(f"Episode:{i}", f"Score:{score}")
    total_score.append(score)

print("====+====+====+====+===+====+====+====+====+====+====+.\n")
print(f"Results After {EPISODE} Episodes:.\n", f"Average Score per episode: {np.mean(total_score)}")

env.close()

plt.figure(1)
plt.ylabel('Score')
plt.title(' CartPole Environment  Using Deep Q- Learning After Train')
plt.plot(np.arange(len(total_score)), total_score)
plt.ylabel('Score')
plt.xlabel('Episodes')
plt.show()
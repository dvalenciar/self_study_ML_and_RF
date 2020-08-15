import gym

env = gym.make("FrozenLake-v0", is_slippery=False)

number_max_steps = 100
number_episodes = 3

for i in range(number_episodes):
    state = env.reset()
    env.render()

    for t in range(number_max_steps):
        action = env.action_space.sample()  # take a accion randomly
        state, reward, done, info = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
env.close()

env.close()

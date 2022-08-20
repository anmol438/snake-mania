import gym
import time

env_name = 'snake_mania:snakemania-v0'
env = gym.make(env_name)

for episode in range(1000):
    env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        img, reward, done, info = env.step(action)
        total_reward += reward
        env.render(mode='human')
        if done:
            break
        time.sleep(0.05)

    print(f'Episode : {episode}, Total reward : {total_reward},  {info}, steps: {env.steps}')

env.close()


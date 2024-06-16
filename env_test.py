import gym
import snake_mania # need to load the environment

env_name = 'snakemania-v0'
env = gym.make(env_name)
env.metadata['render_fps'] = 20 # default is 30, can be changed like this

for episode in range(10):
    env.reset()
    total_reward = 0
    while True:
        action = env.action_space.sample()
        img, reward, done, info = env.step(action)
        total_reward += reward
        env.render(mode='human')
        if done:
            break

    print(f'Episode : {episode}, Total reward : {total_reward},  {info}')
env.close()

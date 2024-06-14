from importlib.metadata import entry_points
from gym import register

register(id='snakemania-v0', entry_point = 'snake_mania.envs:SnakeEnv')
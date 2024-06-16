import pygame, random
from pygame.surfarray import pixels3d
import gym
from gym import spaces
import numpy as np
from collections import deque

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]} # allowed rendering modes

    def __init__(self):
        
        # working in coordinate system : X axis increasing toward right, Y axis increasing toward down
        # shape (rows, cols) = coordinate (y, x)
        self._x = 150 # number of cols
        self._y = 200 # number of rows
        self._node_size = 10 # size of each node in snake body
        self.action_space = spaces.Discrete(4) # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Box(0, 255, (self._y, self._x, 3), np.uint8)
        
        self._collide_punish = -100
        self._step_punish = -1
        self._reward = 50

        self._render_mode = None # it will be pass in render fn
        self._display = None # keep it None until human render is not required
        self._surface = pygame.Surface((self._x,self._y)) # surface to draw observations and rendering

        self.reset()

    def _get_obs(self):
        self._surface.fill(BLACK)
        j=0
        for i in self._snake_body:
            if j==0:
                pygame.draw.rect(self._surface,RED,pygame.Rect(i[0],i[1],self._node_size,self._node_size))
                j+=1
            else:
                pygame.draw.rect(self._surface,WHITE,pygame.Rect(i[0],i[1],self._node_size,self._node_size))
        
        pygame.draw.circle(self._surface, GREEN, (self._food_pos[0] + 5, self._food_pos[1] + 5), self._node_size/2)

        return np.transpose( # 3D rgb array
                np.array(pixels3d(self._surface)), axes=(1, 0, 2)
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._snake_pos = [80,100]
        self._snake_body = deque([[80,100],[70,100],[60,100]])
        self._food_pos = self._spawn_food()

        self._action = 3
        self._direction = "RIGHT"
        
        self._score = 0
        self._steps = 0

        return self._get_obs()

    def _spawn_food(self):
        return [random.randrange(1,self._x//self._node_size)*self._node_size, random.randrange(1,self._y//self._node_size)*self._node_size]

    def _eat_check(self):
        if self._food_pos == self._snake_pos:
            self._score += self._reward
            self._snake_body.append(list(self._snake_body[-1])) # increase a extra dummy node at the end
            self._food_pos = self._spawn_food()
            return self._reward
        else:
            return self._step_punish

    def _change_direction(self):
        if self._action==0 and self._direction != 'DOWN':
            self._direction = 'UP'
        if self._action==1 and self._direction != 'UP':
            self._direction = 'DOWN'
        if self._action==2 and self._direction != 'RIGHT':
            self._direction = 'LEFT'
        if self._action==3 and self._direction != 'LEFT':
            self._direction = 'RIGHT'

        self._move()

    def _move(self):
        if self._direction == 'UP':
            self._snake_pos[1] -= self._node_size
        if self._direction == 'DOWN':
            self._snake_pos[1] += self._node_size
        if self._direction == 'LEFT':
            self._snake_pos[0] -= self._node_size
        if self._direction == 'RIGHT':
            self._snake_pos[0] += self._node_size
            
        self._snake_body.appendleft(list(self._snake_pos))
        self._snake_body.pop()
    
    def step(self,action):
            self._action = action
            self._change_direction()
            reward, done = self._game_over_check()

            if not done:
                reward = self._eat_check()
            
            self._steps += 1
            obs = self._get_obs()
            info = {'Score':self._score, 'NumSteps':self._steps}

            return obs, reward, done, info

    def _game_over_check(self):
        # out of boundary
        if  self._snake_pos[0] < 0 or self._snake_pos[0] > self._x - self._node_size:
            return self._collide_punish, True
        if  self._snake_pos[1] < 0 or self._snake_pos[1] > self._y - self._node_size:
            return self._collide_punish, True

        # collide with itself
        j=0
        for i in self._snake_body:
            if j>0:
                if i == self._snake_pos:
                    return self._collide_punish, True
            j+=1
        
        return self._step_punish, False

    def render(self, mode='None'):

        if not (mode is None or mode in self.metadata["render_modes"]): # not supported render mode
            raise ValueError("SnakeEnv only support 'rgb_array' and 'human' render modes.")
        self._render_mode = mode
            
        obs = self._get_obs()

        # initialize display for first time human mode
        if self._render_mode == 'human' and self._display == None:
            pygame.init()
            pygame.display.init()
            self._display = pygame.display.set_mode((self._x,self._y))

        # update the display for human mode
        if self._render_mode == 'human':
            self._display.blit(self._surface, self._surface.get_rect())   
            pygame.event.pump()
            pygame.display.update()
        
        return obs          

    def close(self):
        if self._display != None:
            self._display = None
            pygame.display.quit()
            pygame.quit()

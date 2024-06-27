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
    metadata = {"render_modes": ["human", "rgb_array"], 'render_fps':30} # allowed rendering modes

    def __init__(self):
        
        # working in coordinate system : X axis increasing toward right, Y axis increasing toward down
        # shape (rows, cols) = coordinate (y, x)
        self._x = 150 # number of cols
        self._y = 200 # number of rows
        self._node_size = 10 # size of each node in snake body
        self.action_space = spaces.Discrete(4) # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Box(0, 255, (self._y, self._x, 3), np.uint8)
        
        self._eaten = False
        self._collide_punish = -100
        self._step_punish = -1
        self._reward = 50
        self._empty_grid = []
        self._snake_body_ind = np.zeros((self._x//self._node_size, self._y//self._node_size))

        self._render_mode = None # it will be pass in render fn
        self._display = None # keep it None until human render is not required
        self._surface = pygame.Surface((self._x,self._y)) # surface to draw observations and rendering
        self._clock = None

        self.reset()

    def _get_obs(self):
        self._surface.fill(BLACK)
        pygame.draw.circle(self._surface, GREEN, (self._food_pos[0] + 5, self._food_pos[1] + 5), self._node_size/2)

        j=0
        for i in self._snake_body:
            if j==0:
                pygame.draw.rect(self._surface,RED,pygame.Rect(i[0],i[1],self._node_size,self._node_size))
                j+=1
            else:
                pygame.draw.rect(self._surface,WHITE,pygame.Rect(i[0],i[1],self._node_size,self._node_size))
        
        return np.transpose( # 3D rgb array
                np.array(pixels3d(self._surface)), axes=(1, 0, 2)
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._snake_body_ind.fill(0)
        self._snake_pos = [80,100]
        self._snake_body = deque([[80,100],[70,100],[60,100]])
        for node in self._snake_body:
            self._snake_body_ind[node[0]//self._node_size][node[1]//self._node_size] += 1
        
        self._food_pos = self._spawn_food()

        self._action = 3
        self._direction = "RIGHT"
        
        self._score = 0
        self._steps = 0

        return self._get_obs()

    def _spawn_food(self):
        available_node = []
        for i in range(0,self._x,10):
            for j in range(0, self._y, 10):
                if not self._snake_body_ind[i//self._node_size][j//self._node_size]:
                    available_node.append([i,j])

        return random.choice(available_node)

    def _eat_check(self):
        if self._food_pos == self._snake_pos:
            self._score += self._reward
            self._eaten = True
            if len(self._snake_body) < ((self._x//self._node_size)*(self._y//self._node_size)):
                self._food_pos = self._spawn_food()
            return self._reward
        else:
            return self._step_punish
        
    def get_action_meanings(self):
        return ['UP', 'DOWN', 'LEFT', 'RIGHT']

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
        if self._in_bounds():
            self._snake_body_ind[self._snake_pos[0]//self._node_size][self._snake_pos[1]//self._node_size] += 1
        if not self._eaten:
            popped = self._snake_body.pop()
            self._snake_body_ind[popped[0]//self._node_size][popped[1]//self._node_size] -= 1
        self._eaten = False
    
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

    def _in_bounds(self):
        if  self._snake_pos[0] < 0 or self._snake_pos[0] > self._x - self._node_size:
            return False
        
        if  self._snake_pos[1] < 0 or self._snake_pos[1] > self._y - self._node_size:
            return False
        
        return True

    def _game_over_check(self):
        
        if not self._in_bounds():
            return self._collide_punish, True

        # collide with itself
        if self._snake_body_ind[self._snake_pos[0]//self._node_size][self._snake_pos[1]//self._node_size] > 1:
            return self._collide_punish, True
        
        return self._step_punish, False

    def render(self, mode='rgb_array'):

        if not (mode in self.metadata["render_modes"]): # not supported render mode
            raise ValueError("SnakeEnv only support 'rgb_array' and 'human' render modes.")
        self._render_mode = mode
            
        obs = self._get_obs()

        # initialize display and clock for first time human mode
        if self._render_mode == 'human' and self._display == None:
            pygame.init()
            pygame.display.init()
            self._display = pygame.display.set_mode((self._x,self._y))
            self._clock = pygame.time.Clock()

        # update the display for human mode
        if self._render_mode == 'human':
            self._display.blit(self._surface, self._surface.get_rect())   
            pygame.event.pump()
            pygame.display.update()
            self._clock.tick(self.metadata['render_fps'])
        
        return obs          

    def close(self):
        if self._display != None:
            self._display = None
            self._clock = None
            pygame.display.quit()
            pygame.quit()

import pygame, random
from pygame.surfarray import pixels3d
import gym
from gym import spaces
import numpy as np

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]} # allowed rendering modes

    def __init__(self):
        
        self.x = 200
        self.y = 200
        self.action_space = spaces.Discrete(4) # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        self.observation_space = spaces.Box(0, 255, (self.x, self.y, 3), np.uint8)
        self.reward_range = (-1, 1)

        self.render_mode = None # it will be pass in render fn
        self._display = None # keep it None until human render is not required
        self._surface = pygame.Surface((self.x,self.y)) # surface to draw observations and rendering

        self.reset()

    def _get_obs(self):
        self._surface.fill(BLACK)
        j=0
        for i in self.snake_body:
            if j==0:
                pygame.draw.rect(self._surface,RED,pygame.Rect(i[0],i[1],11,11))
                j+=1
            else:
                pygame.draw.rect(self._surface,GREEN,pygame.Rect(i[0],i[1],10,10))
        
        pygame.draw.rect(self._surface,WHITE,pygame.Rect(self.food_pos[0],self.food_pos[1],10,10))

        return np.transpose( # 3D rgb array
                np.array(pixels3d(self._surface)), axes=(1, 0, 2)
            )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.snake_pos = [100,50]
        self.snake_body = [[100,50],[90,50],[80,50]]
        self.food_pos = self.spawn_food()

        self.action = 3
        self.direction = "RIGHT"
        
        self.score = 0
        self.steps = 0

        return self._get_obs()

    def spawn_food(self):
        return [random.randrange(1,self.x//10)*10, random.randrange(1,self.y//10)*10]

    def eat_check(self):
        if self.food_pos == self.snake_pos:
            self.score += 1
            self.snake_body.append(list(self.snake_body[-1])) # increase a extra dummy node
            self.food_pos = self.spawn_food()
            return 1
        else:
            return 0

    def change_direction(self):
        if self.action==0 and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.action==1 and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.action==2 and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.action==3 and self.direction != 'LEFT':
            self.direction = 'RIGHT'

        self.move()

    def move(self):
        if self.direction == 'UP':
            self.snake_pos[1] -= 10
        if self.direction == 'DOWN':
            self.snake_pos[1] += 10
        if self.direction == 'LEFT':
            self.snake_pos[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_pos[0] += 10
            
        self.snake_body.insert(0,list(self.snake_pos))
        self.snake_body.pop()
    
    def step(self,action):
            self.action = action
            self.change_direction()
            reward, done = self.game_over_check()

            if not done:
                reward = self.eat_check()
            
            self.steps += 1
            obs = self._get_obs()
            info = {'Score':self.score}

            return obs, reward, done, info

    def game_over_check(self):
        # out of boundary
        if  self.snake_pos[0] < 0 or self.snake_pos[0] > self.x - 10:
            return -1, True
        if  self.snake_pos[1] < 0 or self.snake_pos[1] > self.y - 10:
            return -1, True

        # collide with itself
        for i in self.snake_body[1:]:
            if i == self.snake_pos:
                return -1, True
        
        return 0, False

    def render(self, mode='None'):

        if not (mode is None or mode in self.metadata["render_modes"]): # not supported render mode
            raise ValueError("SnakeEnv only support 'rgb_array' and 'human' render modes.")
        self.render_mode = mode
            
        obs = self._get_obs()

        # initialize display for first time human mode
        if self.render_mode == 'human' and self._display == None:
            pygame.init()
            pygame.display.init()
            self._display = pygame.display.set_mode((self.x,self.y))

        # update the display for human mode
        if self.render_mode == 'human':
            self._display.blit(self._surface, self._surface.get_rect())   
            pygame.event.pump()
            pygame.display.update()
        
        return obs          

    def close(self):
        if self._display != None:
            self._display = None
            pygame.display.quit()
            pygame.quit()

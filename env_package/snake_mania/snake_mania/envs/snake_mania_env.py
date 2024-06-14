import pygame, sys, random
from pygame.surfarray import array3d
import gym
from gym import spaces
import numpy as np

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255,0,0)
GREEN = pygame.Color(0,255,0)


class SnakeEnv(gym.Env):
    metadata = {'render.modes':['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.x = 200
        self.y = 200
        self.game_window = pygame.display.set_mode((self.x,self.y))
        self.LIMIT_STEP = 1000
        self.reset()

    
    def reset(self):
        self.game_window.fill(BLACK)
        self.snake_pos = [100,50]
        self.snake_body = [[100,50],[90,50],[80,50]]
        self.food_pos = self.spawn_food()

        self.action = 3
        self.direction = "RIGHT"
        
        self.score = 0
        self.steps = 0

        img = array3d(pygame.display.get_surface())
        img = np.swapaxes(img,0,1)
        return img

    def spawn_food(self):
        return [random.randrange(1,self.x//10)*10, random.randrange(1,self.y//10)*10]

    def eat_check(self):
        if self.food_pos == self.snake_pos:
            self.score += 1
            self.snake_body.append(list(self.snake_body[-1]))
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
            img = array3d(pygame.display.get_surface())
            img = np.swapaxes(img,0,1)
            info = {'Score':self.score}

            return img, reward, done, info

    def game_over_check(self):
        if  self.snake_pos[0] < 0 or self.snake_pos[0] > self.x - 10:
            return -1, True
        if  self.snake_pos[1] < 0 or self.snake_pos[1] > self.y - 10:
            return -1, True

        for i in self.snake_body[1:]:
            if i == self.snake_pos:
                return -1, True
        
        if self.steps>=self.LIMIT_STEP:
            return 0, True
        
        return 0, False

    def render(self, mode = 'human'):
        self.game_window.fill(BLACK)
        j=0
        for i in self.snake_body:
            if j==0:
                pygame.draw.rect(self.game_window,RED,pygame.Rect(i[0],i[1],11,11))
                j+=1
            else:
                pygame.draw.rect(self.game_window,GREEN,pygame.Rect(i[0],i[1],10,10))
        
        pygame.draw.rect(self.game_window,WHITE,pygame.Rect(self.food_pos[0],self.food_pos[1],10,10))

        if mode == 'human':
            pygame.display.update()

        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def close(self):
        pygame.quit()
        sys.exit()
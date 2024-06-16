
# Environment: snake-mania
A custom OpenAI Gym based environment for training a snake to eat food.

# Installing environment
Create a python environment with python version >= 3.8  
Run command <code>pip install -e snake-mania</code> in <code>./env_package/</code> folder to install the snakemania gym environment.

# Creating Environment Instance
After installating the package, you can create the instance of the evironment as:
```python
import gym
import snake_mania

env_name = 'snakemania-v0'
env = gym.make(env_name)
```

# Running the snake mania game

Run the <code>env_test.py</code> file to explore the environment.  

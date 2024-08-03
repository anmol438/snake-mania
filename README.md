
# Environment: snake-mania
A custom OpenAI Gym based environment for training a snake to eat food.

# Installing environment
Create a python environment with python version >= 3.8  (preferably py3.10)
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

https://github.com/anmol438/snake-mania/assets/50985412/20d1e248-50da-4f03-b3ba-99a8d2532cef


# Train the Agent on Snake-Mania

# Installing Dependencies
Install **Tf-Agents** version 0.19.0 : <code>pip install tf-agents==0.19.0</code>  
Install **Tensorflow** version 2.15.0 : <code>pip install tensorflow==2.15.0</code>  
Install **Imageio** versio 2.4.0 : <code>pip install imageio==2.4.0</code>  

# Agent Training
The agent has been trained for few million steps in the environment to make the snake eat the food without any collision. It has shown good performance.

# Load Model and Continue Training
To directly explore the trained model or continue the training for much better performance, create the RL components with same architecture as in <code>./training_env/snake-mania_training.py</code> and load the checkpointer.  
The <code>agent.policy.action(time_step).action</code> can be used after loading the model to chose action and explore the trained model.

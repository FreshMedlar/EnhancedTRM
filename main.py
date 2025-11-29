from world_model import WorldModel
from dynamic_model import DynamicModel
from memory import ModernHopfieldNetwork
from action_model import ActionModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
env = gym.make("LunarLander-v3")
# observation = (lander_x, lander_y, velocity_x, velocity_y, angle, ang_velocity, leg1, leg2)
observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    

    # create models and utils
    input_dim = 10
    output_dim = 1

    world_model = WorldModel(input_dim, 32, output_dim)
    action_model = ActionModel(input_dim, 32, output_dim)
    dynamic_model = DynamicModel(input_dim, 32, output_dim)
    memory_model = ModernHopfieldNetwork(20.0)
    
    # 2. The Loss Function (Criterion)
    # MSELoss is standard for regression (predicting a number)
    criterion = nn.MSELoss()

    # 3. The Optimizer
    # SGD (Stochastic Gradient Descent) with a learning rate of 0.01
    w_optimizer = optim.AdamW(world_model.parameters(), lr=1e-4, weight_decay=0.01)
    a_optimizer = optim.AdamW(action_model.parameters(), lr=1e-4, weight_decay=0.01)
    d_optimizer = optim.AdamW(dynamic_model.parameters(), lr=1e-4, weight_decay=0.01)











    if terminated or truncated:
        observation, info = env.reset()
env.close()


from .reward_model import RewardModel
from .world_model import WorldModel
from .action_model import ActionModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym

# Setup environment and device
device = torch.device("cpu") # Forcing CPU for this environment
env = gym.make("LunarLander-v3")

obs_dims = env.observation_space.shape[0]
act_dims = env.action_space.n

# Create models
world_model = WorldModel(obs_dims, 1, 128, obs_dims).to(device)
action_model = ActionModel(obs_dims, 128, act_dims).to(device)
reward_model = RewardModel(obs_dims, 128, 1).to(device)

# Optimizers
w_optimizer = optim.AdamW(world_model.parameters(), lr=1e-4)
a_optimizer = optim.AdamW(action_model.parameters(), lr=1e-3)
r_optimizer = optim.AdamW(reward_model.parameters(), lr=1e-4)

mse_criterion = nn.MSELoss()


def select_action(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    logits = action_model(obs_t)
    dist = Categorical(logits=logits)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(), log_prob  # Return tensor, not .item()


num_episodes = 1500 # Reduced for quicker demonstration
gamma = 0.99

print("Starting training...")
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    ep_obs, ep_actions, ep_log_probs, ep_rewards, ep_next_obs = [], [], [], [], []
    
    reward, pred_rewards = [], []

    while not done:
        act_pred, log_prob = select_action(obs)

        # Train World Model
        # input: world model, action
        # output: future world model
        # how do we make this model check at different points in the future? we pass a variable along to signal when? how does it learn then?
        # this model learns by checking if predictions are corrects (HOW? delayed proof of correctness)       w_optimizer.zero_grad()
        w_optimizer.zero_grad()
        pred_next_obs = world_model(obs, act_pred)

        # Train Reward Model
        # input: world_model(), objective() 
        # output: reward ()
        # This reward model simply judges an hypotetical world model, and use it to decide what to do next. It probably needs to recurrently check
        # learn by checking the reward of the current world model paired with the objective
        r_optimizer.zero_grad()
        pred_rewards = reward_model(pred_next_obs).squeeze()
        

        if (pred_rewards > 0):
            action = act_pred 
        else:
            action = []

        # Train Action Model (REINFORCE)
        # input: world model, objective
        # output: action
        # this model gives an hypotetical action
        # again this learning suffers from delayed rewards. What could give a loss to learn? the outcome can be theorized and loss can be estimated
        # eat candy -> fat (here we use world model to infer future) 
        # fat -> bad (high loss) (here we use reward model to infer loss)
        # the cycle could be action - world - reward
        



        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        
        # world learning
        world_loss = mse_criterion(pred_next_obs, obs)
        world_loss.backward()
        w_optimizer.step()
        # reward learning
        reward_loss = mse_criterion(pred_rewards, reward)
        reward_loss.backward()
        r_optimizer.step()

# The code from here on was old code, that I need to remove after finishing restructuring
# before the strategy was to perform a whole episode and store it in these lists and then train on them

        # Prepare Tensors - stack instead of converting from numpy
        obs_t = torch.tensor(np.array(ep_obs), dtype=torch.float32, device=device)
        act_t = torch.tensor(np.array(ep_actions), dtype=torch.float32, device=device).unsqueeze(1)
        log_probs_t = torch.stack(ep_log_probs).squeeze()  # Stack tensors to preserve gradients
        rew_t = torch.tensor(ep_rewards, dtype=torch.float32, device=device)
        next_obs_t = torch.tensor(np.array(ep_next_obs), dtype=torch.float32, device=device)

        returns = []
        discounted_return = 0
        for r in reversed(ep_rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)
        
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        a_optimizer.zero_grad()
        policy_loss = (-log_probs_t * returns_t).sum()
        policy_loss.backward()
        a_optimizer.step()

# this is the end of the old code


        ep_reward += reward
        

    # if (episode + 1) % 50 == 0:
    #     print(f"Episode {episode+1:4d} | Total Reward: {ep_reward:7.2f} | Policy Loss: {policy_loss.item():7.2f} | World Loss: {world_loss.item():7.4f}")

env.close()
print("Training finished.")






















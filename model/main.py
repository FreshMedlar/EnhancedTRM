from reward_model import RewardModel
from world_model import WorldModel
from action_model import ActionModel

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym
import copy

# Setup environment and device
device = torch.device("cpu") # Forcing CPU for this environment
env = gym.make("LunarLander-v3")

obs_dims = env.observation_space.shape[0]
act_dims = env.action_space.n

# Create models
world_model = WorldModel(obs_dims, 1, 128, 128, obs_dims).to(device)
action_model = ActionModel(obs_dims, 128, 128, act_dims).to(device)
reward_model = RewardModel(obs_dims, 128, 128, 1).to(device)

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
    return action, log_prob  # Return tensor, not .item()

# -------------------------------------------------------------
# THE CONCEPT: BACKPROPAGATION THROUGH TIME (BPTT)
# -------------------------------------------------------------
# instead of guessing actions, we let the gradients tell the 
# Action Model how to adjust its weights to maximize value.

def train_policy_in_imagination(world_model, reward_model, action_model, a_optimizer, obs_batch):
    """
    The Policy 'dreams' a future, sees the result, and updates itself
    to make that result better next time.
    """
    
    # 1. Freeze World and Reward Models
    # We don't want the Policy to 'break' the physics engine to get rewards.
    # We want it to learn to act INSIDE the physics engine.
    world_model.requires_grad_(False)
    reward_model.requires_grad_(False)
    
    horizon = 15
    traj_states = []
    traj_rewards = []
    
    # Start thinking from the current real observations
    state = obs_batch
    
    # 2. Rollout (Imagine the future)
    for t in range(horizon):
        # The Policy chooses an action based on the imagined state
        # CRITICAL: This action must allow gradients to pass through!
        # In Pytorch distributions, .rsample() allows gradients (reparameterization trick)
        # If using deterministic outputs, just direct forward pass is fine.
        action_logits = action_model(state)
        dist = Categorical(logits=action_logits)
        
        # We can't use simple sampling for discrete actions in standard backprop 
        # without Gumbel-Softmax, but for simplicity, let's assume
        # we use the logits directly or a continuous approximation.
        # For LunarLander (Discrete), Dreamer uses "Straight-Through Estimator" or Gumbel.
        # Here is a simplified continuous-like trick for demonstration:
        action_probs = F.softmax(action_logits, dim=-1)
        
        # We need to turn probs into an input for World Model (one-hot like)
        # This is the tricky part in Discrete spaces. 
        # Let's assume we pass the probabilities directly to a modified World Model
        # or use Gumbel-Softmax:
        action = F.gumbel_softmax(action_logits, hard=False) 
        
        # Predict next state using the chosen action
        next_state = world_model(state, action.unsqueeze(1))
        
        # Predict value of that state
        value = reward_model(next_state)
        
        traj_rewards.append(value)
        state = next_state

    # 3. Calculate Loss
    # The loss is negative value (we want to maximize value)
    # We sum the value of all future states in the dream
    total_imagined_value = torch.stack(traj_rewards).sum()
    policy_loss = -total_imagined_value
    
    # 4. Update Policy
    a_optimizer.zero_grad()
    policy_loss.backward()
    a_optimizer.step()
    
    # Unfreeze models for their own training steps later
    world_model.requires_grad_(True)
    reward_model.requires_grad_(True)
    
    return policy_loss.item()


num_episodes = 1500 # Reduced for quicker demonstration
gamma = 0.99
print("Starting training...")
buffer = []
MAX_BUFFER = 10000
for episode in range(num_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    pred_ep_rewards = 0.0 

    while not done:
        # FIND OBJECTIVE
        # we add noise to ensure the model explore, but this is solved with a "parent model"
        if len(buffer) < 500:
            action = env.action_space.sample()
        else:
            if np.random.rand() < 0.1:
                action = env.action_space.sample()
            else:
                action = plan_action(obs, world_model, reward_model)

        # EXECUTE 
        next_obs, real_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # Store transition
        buffer.append((obs, action, real_reward, next_obs, done))
        if len(buffer) > MAX_BUFFER: buffer.pop(0)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        # action selection
        action, log_prob = select_action(obs)
        action_t = torch.tensor([action.item()], dtype=torch.float32, device=device).unsqueeze(1)
        pred_next_obs = world_model(obs_t, action_t)
        pred_rewards = reward_model(pred_next_obs.detach()).view(-1)
        a_optimizer.zero_grad()
        action_loss = -log_prob * pred_rewards.detach()
        action_loss.backward()
        a_optimizer.step()

        # env update
        next_obs, real_reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
        real_reward_t = torch.tensor([real_reward], dtype=torch.float32, device=device)
        
        # world learning
        w_optimizer.zero_grad()
        world_loss = mse_criterion(pred_next_obs.squeeze(), next_obs_t)
        world_loss.backward()
        w_optimizer.step()

        # reward learning
        r_optimizer.zero_grad()
        reward_loss = mse_criterion(pred_rewards, real_reward_t)
        reward_loss.backward()
        r_optimizer.step()

        obs = next_obs
        ep_reward += real_reward
        pred_ep_rewards += pred_rewards.item()
        

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1:4d} | Real Reward: {ep_reward:7.2f} | Reward: {pred_ep_rewards:7.2f} | Action Loss: {action_loss.item():7.2f} | world Loss: {world_loss.item():7.4f}")

env.close()
print("Training finished.")






















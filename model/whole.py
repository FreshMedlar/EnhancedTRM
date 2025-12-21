import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
from PIL import Image
import wandb
import random
from collections import deque
import math

import sys
import os
from dataclasses import dataclass

# Constants
IMG_SIZE = 128
NUM_ACTIONS = 4
LATENT_DIM = 128 # d_model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

FRAME_STACK = 4
def preprocess(obs):
    """
    Resize observation to 128x128, convert to grayscale, and return as tensor.
    """
    img = Image.fromarray(obs).convert("L") # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST) # Nearest for small objects
    
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, 128, 128)

def stack_frames(frames_deque):
    # frames_deque contains tensors of shape (1, 1, 128, 128)
    return torch.cat(list(frames_deque), dim=1) # (1, FRAME_STACK, 128, 128)

from networks import Actor, Critic
from jepa import JEPAWorldModel, SIGReg

class MBRLAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        
        self.wm = JEPAWorldModel(LATENT_DIM, NUM_ACTIONS, in_channels=FRAME_STACK).to(device)
        self.sigreg = SIGReg().to(device)
        self.goal_memory = [] # Store latent embeddings of successful states
        self.max_goals = 1000
        
        self.actor = Actor(LATENT_DIM, NUM_ACTIONS).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        self.wm_optim = AdamW(self.wm.parameters(), lr=1e-4) 
        self.ac_optim = AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=1e-4
        )

    def get_action(self, obs, deterministic=False):
        """
        Get action, log_prob, value, and state embedding from current observation.
        """
        with torch.no_grad():
            # Obs: (1, 3, 128, 128)
            _, state = self.wm.encoder(obs) # (1, LATENT_DIM)
            
            logits = self.actor(state)
            value = self.critic(state)
            
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
                
        return action, log_prob, value, state

    def get_intrinsic_reward(self, state):
        """
        Calculate max cosine similarity between state and goal memory.
        state: (1, LATENT_DIM)
        """
        if not self.goal_memory:
            return 0.0
        
        # Stack goals: (N, LATENT_DIM)
        # Check if we have enough goals to batch?
        # Ideally keep goals as a tensor buffer
        goals = torch.stack(self.goal_memory) # (N, D)
        
        # Normalize state and goals
        state_norm = F.normalize(state, p=2, dim=1)
        goals_norm = F.normalize(goals, p=2, dim=1)
        
        # Similarity: (1, D) @ (D, N) -> (1, N)
        sim_scores = torch.mm(state_norm, goals_norm.t())
        
        max_sim = sim_scores.max().item()
        return max_sim

    def train_world_model(self, batch):
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        # obs_b: (B, 3, 128, 128)
        
        # 1. Encode current and next observations
        _, proj_t = self.wm.encoder(obs_b) # h_t
        _, proj_next = self.wm.encoder(next_obs_b) # h_{t+1} (Target)
        
        # 2. Predict next latent
        act_emb = self.wm.act_embedding(act_b.squeeze(1)) # (B, D)
        pred_proj_next = self.wm.predictor_net(proj_t, act_emb) # \hat{h}_{t+1}
        
        # 3. Losses
        
        # Prediction Loss (Latent Space)
        # Prevent collapse? SIGReg handles regularization.
        # Minimal example: 
        # loss = sigreg(proj) * lamb + inv_loss * (1-lamb)
        # Here we have dynamics prediction.
        
        # Invariance Loss (MSE between prediction and target)
        # Note: In standard JEPA, we might stop gradient on target, but LeJEPA paper says "no stop-gradients"
        loss_pred = F.mse_loss(pred_proj_next, proj_next) # | \hat{h}_{t+1} - h_{t+1} |^2
        
        # Regularization on projected states (to prevent collapse to 0)
        # Apply SIGReg to both h_t and h_{t+1}? Or just h_t?
        # Minimal example applies it to `proj` (output of encoder).
        loss_sigreg = self.sigreg(proj_t) + self.sigreg(proj_next)
        
        # Total Loss
        # Reverting to the simple weighted sum from MINIMAL.md
        # lejepa_loss = sigreg_loss * lamb + inv_loss * (1 - lamb)
        # Here pred_loss is our "invariance" or performance goal.
        
        loss_sig = self.sigreg(proj_t) + self.sigreg(proj_next)
        lamb = 0.02 # Starting with paper recommended value
        
        total_loss = (1 - lamb) * loss_pred + lamb * loss_sig
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        self.wm_optim.step()
        
        return {
            "loss_wm": loss_pred.item(),
            "loss_sigreg": loss_sig.item(),
            "rep/std": proj_t.std(dim=0).mean().item(),
            "rep/avg_norm": proj_t.norm(p=2, dim=1).mean().item(),
            "rep/pairwise_dist": F.pdist(proj_t).mean().item() if proj_t.size(0) > 1 else 0.0
        }

    def update_ac(self, rollouts):
        """
        Standard PPO Update on collected rollouts.
        """
        states, actions, rewards, values, log_probs = rollouts
        if not states: return {}

        # Stack lists
        # T steps, B batch
        T = len(rewards)
        B = rewards[0].size(0)
        
        # Flatten time and batch for processing
        rewards = torch.stack(rewards).view(T, B) # (T, B)
        values = torch.stack(values).view(T, B) # (T, B)
        log_probs = torch.stack(log_probs).view(T, B) # (T, B)
        
        # Calculate Advantages (GAE)
        gamma = 0.99
        lam = 0.95
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        last_gae = 0
        last_val = 0 # Assume 0 value at trajectory end
        
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * last_val - values[t]
            last_gae = delta + gamma * lam * last_gae
            advantages[t] = last_gae
            last_val = values[t]
            
        returns = advantages + values
        
        # Flatten
        b_states = torch.stack(states).view(T*B, LATENT_DIM)
        b_actions = torch.stack(actions).view(T*B)
        b_log_probs = log_probs.view(T*B)
        b_returns = returns.view(T*B)
        b_advantages = advantages.view(T*B)
        
        # PPO Epochs
        for _ in range(3):
            new_logits = self.actor(b_states)
            dist = torch.distributions.Categorical(logits=new_logits)
            new_log_probs = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()
            
            new_values = self.critic(b_states).squeeze(-1)
            
            ratio = (new_log_probs - b_log_probs).exp()
            surr1 = ratio * b_advantages.detach()
            surr2 = torch.clamp(ratio, 0.8, 1.2) * b_advantages.detach()
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            critic_loss = F.mse_loss(new_values, b_returns.detach())
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.ac_optim.zero_grad()
            loss.backward()
            self.ac_optim.step()
            
        return {"loss_ac": loss.item(), "val_est": new_values.mean().item()}

def train():
    print(f"Using device: {DEVICE}")
    wandb.init(project="EnhancedTRM")
    
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    agent = MBRLAgent()
    
    replay_buffer = deque(maxlen=5000) # Increased capacity
    
    print("Pre-filling buffer...")
    obs, _ = env.reset()
    
    # Initialize frame stack
    frame_stack_deque = deque([preprocess(obs) for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
    curr_stacked_obs = stack_frames(frame_stack_deque)
    
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # New frame
        next_frame = preprocess(next_obs)
        frame_stack_deque.append(next_frame)
        next_stacked_obs = stack_frames(frame_stack_deque)
        
        replay_buffer.append((curr_stacked_obs, action, reward, next_stacked_obs, done))
        
        curr_stacked_obs = next_stacked_obs
        if done:
            obs, _ = env.reset()
            frame_stack_deque = deque([preprocess(obs) for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
            curr_stacked_obs = stack_frames(frame_stack_deque)
            
    print("Starting Training...")
    bs = 32
    
    rollout_len = 128
    states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf = [], [], [], [], []

    # Trackers
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    frames = [] # To store frames for video logging
    action_counts = {i: 0 for i in range(NUM_ACTIONS)}

    for step in range(50000): # Increased steps
        # 1. Interact
        action, log_prob, value, state = agent.get_action(curr_stacked_obs.to(DEVICE))
        
        act_val = action.item()
        action_counts[act_val] += 1
        next_obs, reward, term, trunc, info = env.step(act_val)
        done = term or trunc
        
        episode_reward += reward
        episode_steps += 1
        
        # New frame and stack
        next_frame = preprocess(next_obs)
        frame_stack_deque.append(next_frame)
        next_stacked_obs = stack_frames(frame_stack_deque)
        
        # Goal-Conditioned Logic
        if reward > 0.1: # Hit a block
            if len(agent.goal_memory) < agent.max_goals:
                agent.goal_memory.append(state.detach().squeeze(0)) # (D,)
            else:
                 agent.goal_memory.pop(0)
                 agent.goal_memory.append(state.detach().squeeze(0))
        
        # Intrinsic Reward
        with torch.no_grad():
             _, next_state_emb = agent.wm.encoder(next_stacked_obs.to(DEVICE))
        
        intrinsic_reward = agent.get_intrinsic_reward(next_state_emb)
        alpha = 0.5 
        total_reward = reward + alpha * intrinsic_reward
        
        # Visualization
        if episode_count % 10 == 0 and len(frames) < 300:
            small_frame = Image.fromarray(next_obs).resize((160, 210))
            frames.append(np.array(small_frame).transpose(2, 0, 1))
        
        # Store for AC update
        states_buf.append(state)
        actions_buf.append(action)
        rewards_buf.append(torch.tensor([total_reward], device=DEVICE))
        values_buf.append(value)
        log_probs_buf.append(log_prob)
        
        replay_buffer.append((curr_stacked_obs, act_val, reward, next_stacked_obs, done))
        curr_stacked_obs = next_stacked_obs
        
        if done:
            # Log episode metrics
            log_dict = {
                "env/episode_reward": episode_reward,
                "env/episode_steps": episode_steps,
                "env/episode_count": episode_count,
                "env/max_intrinsic": intrinsic_reward,
                "env/goals_stored": len(agent.goal_memory)
            }
            
            if frames:
                video_array = np.stack(frames)
                log_dict["env/gameplay"] = wandb.Video(video_array, fps=30, format="gif")
                frames = []
            
            for a in range(NUM_ACTIONS):
                log_dict[f"actions/act_{a}"] = action_counts[a]
                action_counts[a] = 0
                
            wandb.log(log_dict)
            
            print(f"Episode {episode_count} finished. Reward: {episode_reward}, Steps: {episode_steps}")
            
            episode_reward = 0
            episode_steps = 0
            episode_count += 1
            
            obs, _ = env.reset()
            frame_stack_deque = deque([preprocess(obs) for _ in range(FRAME_STACK)], maxlen=FRAME_STACK)
            curr_stacked_obs = stack_frames(frame_stack_deque)
            
        # 2. Update AC periodically
        if len(states_buf) >= rollout_len:
            rollouts = (states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf)
            ac_metrics = agent.update_ac(rollouts)
            wandb.log(ac_metrics)
            states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf = [], [], [], [], []

        # 3. Update World Model
        if len(replay_buffer) > bs:
            batch = random.sample(replay_buffer, bs)
            
            b_obs = torch.cat([b[0] for b in batch]).to(DEVICE)
            b_act = torch.tensor([b[1] for b in batch], device=DEVICE).unsqueeze(1)
            b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            b_next = torch.cat([b[3] for b in batch]).to(DEVICE)
            b_done = torch.tensor([b[4] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            
            wm_metrics = agent.train_world_model((b_obs, b_act, b_rew, b_next, b_done))
            wandb.log(wm_metrics)
            
        if step % 100 == 0:
            print(f"Step {step}")
            
    env.close()

def test():
    print(f"Using device: {DEVICE}")
    print("Testing MBRLAgent (Verification)...")
    agent = MBRLAgent()
    
    # 1. Test Forward Pass
    print("1. Testing World Model Forward Pass...")
    dummy_obs = torch.rand((2, FRAME_STACK, 128, 128)).to(DEVICE) # Float
    dummy_act = torch.zeros((2, 1), dtype=torch.long).to(DEVICE)
    dummy_next = torch.rand((2, FRAME_STACK, 128, 128)).to(DEVICE)
    
    # Updated batch for Goal-Conditioned (no rew/done needed in WM update, but function signature expects them unpacked)
    # train_world_model unpacks 5 items. We pass dummies.
    dummy_rew = torch.zeros((2, 1)).to(DEVICE)
    dummy_done = torch.zeros((2, 1)).to(DEVICE)
    
    batch = (dummy_obs, dummy_act, dummy_rew, dummy_next, dummy_done)
    metrics = agent.train_world_model(batch)
    print(f"WM Loss: {metrics['loss_wm']:.4f}")
    
    # 3. Test Action
    print("3. Testing Actor...")
    action, log_prob, value, state = agent.get_action(dummy_obs[0:1])
    print(f"Action selected: {action}")
    
    print("Test Passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        test()

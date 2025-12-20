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

# Hyperparameters
IMG_SIZE = 16
PIXELS = IMG_SIZE * IMG_SIZE # 256
NUM_ACTIONS = 4
VOCAB_SIZE = 256 + NUM_ACTIONS + 2 # 0-255 pixels, 4 actions, 2 specials (SOS, PAD/EOS if needed)
# Token mapping
# 0-255: Pixels
# 256-259: Actions (Action 0 -> 256, etc.)
# 260: SOS (Start of Sequence) / Separator if needed
ACTION_OFFSET = 256
SOS_TOKEN = 260
TOTAL_VOCAB_SIZE = 261

# Seq definition: [Obs (256), Action (1), NextObs (256)]
# Context length needed: 256 + 1 + 256 = 513
MAX_SEQ_LEN = 513 
LATENT_DIM = 128 # d_model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess(obs):
    """
    Resize observation to 32x32 and convert to grayscale.
    Returns: torch.Tensor of shape (1, 1024) with values 0-255 (Long)
    """
    img = Image.fromarray(obs)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img = img.convert("L") # Grayscale
    
    # Flatten array to (1024,)
    arr = np.array(img).flatten()
    
    # Add batch dim -> (1, 1024)
    return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

from rwkv_model import RWKVConfig, RWKVWorldModel
from networks import Actor, Critic

class MBRLAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        
        # RWKV World Model
        self.rwconfig = RWKVConfig(
            vocab_size=TOTAL_VOCAB_SIZE,
            block_size=MAX_SEQ_LEN,
            n_embd=LATENT_DIM,
            n_layer=2, 
            use_customized_cuda_kernel=False
        )
        self.world_model = RWKVWorldModel(self.rwconfig).to(device)

        self.actor = Actor(LATENT_DIM, NUM_ACTIONS).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        self.wm_optim = AdamW(self.world_model.parameters(), lr=3e-4)
        self.ac_optim = AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=1e-4
        )

    def get_action(self, obs_seq, deterministic=False):
        """
        Get action, log_prob, value, and state embedding from current observation.
        """
        with torch.no_grad():
            _, embeddings, _ = self.world_model(obs_seq)
            state = embeddings[:, -1, :] 
            
            logits = self.actor(state)
            value = self.critic(state)
            
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
                
        return action, log_prob, value, state

    def train_world_model(self, batch):
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        B = obs_b.size(0)
        
        act_tokens = act_b + ACTION_OFFSET
        full_seq = torch.cat([obs_b, act_tokens, next_obs_b], dim=1) # (B, 513)
        
        inputs = full_seq[:, :-1] 
        targets = full_seq[:, 1:] 
        
        logits, _, (rew_pred, done_pred) = self.world_model(inputs)
        
        # 1. Pixel/Token Loss (Dynamics)
        loss_tokens = F.cross_entropy(logits.reshape(-1, TOTAL_VOCAB_SIZE), targets.reshape(-1))
        
        # 2. Reward and Done Loss (Objectives)
        # We want to predict the reward/done based on the state after the action is taken.
        # Action is at index 256 in the full_seq (0-indexed).
        # So in 'inputs' (length 512), index 256 is the action token.
        # Its embedding has seen (Obs + Action). This is when we can predict Reward and NextObs.
        pred_idx = PIXELS # The index of the action token
        
        target_rew = rew_b.float() # (B, 1)
        target_done = done_b.float() # (B, 1)
        
        loss_rew = F.mse_loss(rew_pred[:, pred_idx, :], target_rew)
        loss_done = F.binary_cross_entropy_with_logits(done_pred[:, pred_idx, :], target_done)
        
        total_loss = loss_tokens + 10.0 * loss_rew + 5.0 * loss_done
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        self.wm_optim.step()
        
        return {
            "loss_wm": loss_tokens.item(),
            "loss_rew": loss_rew.item(),
            "loss_done": loss_done.item(),
            "pred_logits": logits.detach()
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
    wandb.init(project="EnhancedTRM")
    
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    agent = MBRLAgent()
    
    replay_buffer = deque(maxlen=2000)
    
    print("Pre-filling buffer...")
    obs, _ = env.reset()
    curr_seq = preprocess(obs) 
    
    for i in range(50):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        next_seq = preprocess(next_obs)
        replay_buffer.append((curr_seq, action, reward, next_seq, done))
        
        curr_seq = next_seq
        if done:
            obs, _ = env.reset()
            curr_seq = preprocess(obs)
            
    print("Starting Training...")
    bs = 8
    
    
    rollout_len = 128
    states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf = [], [], [], [], []

    for step in range(10000):
        # 1. Interact
        action, log_prob, value, state = agent.get_action(curr_seq.to(DEVICE))
        
        act_val = action.item()
        next_obs, reward, term, trunc, info = env.step(act_val)
        done = term or trunc
        
        # Store for AC update
        states_buf.append(state) # (1, 256)
        actions_buf.append(action) # (1,)
        rewards_buf.append(torch.tensor([reward], device=DEVICE))
        values_buf.append(value) # (1, 1)
        log_probs_buf.append(log_prob) # (1,)
        
        next_seq = preprocess(next_obs)
        replay_buffer.append((curr_seq, act_val, reward, next_seq, done))
        curr_seq = next_seq
        
        if done:
            obs, _ = env.reset()
            curr_seq = preprocess(obs)
            
        # 2. Update AC periodically (Direct Learning)
        if len(states_buf) >= rollout_len:
            rollouts = (states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf)
            ac_metrics = agent.update_ac(rollouts)
            wandb.log(ac_metrics)
            # Clear buffers
            states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf = [], [], [], [], []

        # 3. Update World Model (Representation Learning)
        if len(replay_buffer) > bs:
            batch = random.sample(replay_buffer, bs)
            
            b_obs = torch.cat([b[0] for b in batch]).to(DEVICE)
            b_act = torch.tensor([b[1] for b in batch], device=DEVICE).unsqueeze(1)
            b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            b_next = torch.cat([b[3] for b in batch]).to(DEVICE)
            b_done = torch.tensor([b[4] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            
            wm_metrics = agent.train_world_model((b_obs, b_act, b_rew, b_next, b_done))
            
            if step % 500 == 0:
                pred_logits = wm_metrics["pred_logits"]
                pred_pixels_logits = pred_logits[0, PIXELS:2*PIXELS, :]
                pred_pixels = pred_pixels_logits.argmax(dim=-1).cpu().numpy().astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
                target_pixels = b_next[0].cpu().numpy().astype(np.uint8).reshape(IMG_SIZE, IMG_SIZE)
                
                vis_img = np.hstack([target_pixels, pred_pixels])
                wm_metrics["reconstruction"] = wandb.Image(vis_img, caption="Actual vs Predicted")
                
            del wm_metrics["pred_logits"]
            wandb.log(wm_metrics)
            
        if step % 100 == 0:
            print(f"Step {step}")
            
    env.close()

def test():
    print("Testing MBRLAgent (Verification)...")
    agent = MBRLAgent()
    
    # 1. Test Forward Pass
    print("1. Testing World Model Forward Pass...")
    dummy_obs = torch.randint(0, 255, (2, PIXELS)).to(DEVICE)
    dummy_act = torch.zeros((2, 1), dtype=torch.long).to(DEVICE)
    dummy_next = torch.randint(0, 255, (2, PIXELS)).to(DEVICE)
    
    dummy_rew = torch.zeros((2, 1), dtype=torch.float).to(DEVICE)
    dummy_done = torch.zeros((2, 1), dtype=torch.float).to(DEVICE)
    
    batch = (dummy_obs, dummy_act, dummy_rew, dummy_next, dummy_done)
    metrics = agent.train_world_model(batch)
    print(f"WM Loss: {metrics['loss_wm']:.4f}")
    
    # 3. Test Action
    print("3. Testing Actor...")
    action, log_prob, value, state = agent.get_action(dummy_obs)
    print(f"Action selected: {action}")
    
    print("Test Passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        test()
        # train() # Uncomment to run training after test manually

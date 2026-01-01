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
from typing import Optional, Tuple, List, Union

# --- Constants and Configuration ---
IMG_SIZE = 84
NUM_ACTIONS = 4
LATENT_DIM = 128 # d_model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 8 
FRAME_STACK = 1
TRAIN_FREQ = 50
TRAIN_STEPS = 5

# --- TOPAS Components (Adapted for Atari) ---

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, num_channels, emb_dim):
        super().__init__()
        self.num_channels = num_channels
        self.norm = nn.LayerNorm(num_channels, elementwise_affine=False)
        self.emb_proj = nn.Linear(emb_dim, num_channels * 2)
        nn.init.zeros_(self.emb_proj.weight)
        nn.init.zeros_(self.emb_proj.bias)

    def forward(self, x, emb):
        # x: [B, C, H, W], emb: [B, D]
        x_perm = x.permute(0, 2, 3, 1) # [B, H, W, C]
        x_norm = self.norm(x_perm)
        emb_out = self.emb_proj(emb).unsqueeze(1).unsqueeze(1) # [B, 1, 1, 2*C]
        scale, shift = emb_out.chunk(2, dim=-1)
        out = x_norm * (1 + scale) + shift
        return out.permute(0, 3, 1, 2)

class CanvasCoreLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout, img_size):
        super().__init__()
        self.img_size = img_size
        self.conv1 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.adagn1 = AdaptiveLayerNorm(d_model, emb_dim=d_model)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.adagn2 = AdaptiveLayerNorm(d_model, emb_dim=d_model)
        self.cross_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_tokens, logic_context, instruction_vec):
        # x_tokens: [B, H*W, D], logic_context: [B, L_len, D], instruction_vec: [B, D]
        B, HW, D = x_tokens.shape
        H = W = int(math.sqrt(HW))
        
        # 1. Spatial Mixing
        x_norm1 = self.norm1(x_tokens)
        x_grid = x_norm1.transpose(1, 2).view(B, D, H, W)
        h = self.act1(self.adagn1(self.conv1(x_grid), instruction_vec))
        h = self.adagn2(self.conv2(h), instruction_vec)
        local_out = h.view(B, D, HW).transpose(1, 2)
        x_tokens = x_tokens + self.dropout(local_out)

        # 2. Cross-Attention
        x_norm2 = self.norm2(x_tokens)
        cross_out, _ = self.cross_att(x_norm2, logic_context, logic_context)
        x_tokens = x_tokens + self.dropout(self.gate * cross_out)

        # 3. FFN
        x_norm3 = self.norm3(x_tokens)
        ff_out = self.linear2(F.relu(self.linear1(x_norm3)))
        x_tokens = x_tokens + self.dropout(ff_out)
        return x_tokens

class LogicCoreLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.self_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_att = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, ff_dim)
        self.linear2 = nn.Linear(ff_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_logic, canvas_context):
        # x_logic: [B, L_len, D], canvas_context: [B, HW, D]
        x_norm1 = self.norm1(x_logic)
        self_out, _ = self.self_att(x_norm1, x_norm1, x_norm1)
        x_logic = x_logic + self.dropout(self_out)

        x_norm2 = self.norm2(x_logic)
        cross_out, _ = self.cross_att(x_norm2, canvas_context, canvas_context)
        x_logic = x_logic + self.dropout(self.gate * cross_out)

        x_norm3 = self.norm3(x_logic)
        ff_out = self.linear2(F.relu(self.linear1(x_norm3)))
        x_logic = x_logic + self.dropout(ff_out)
        return x_logic

# --- TOPAS World Model ---

class TOPASWorldModel(nn.Module):
    def __init__(self, latent_dim, num_actions, img_size=84, h_cycles=2, l_cycles=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles
        self.img_size = img_size
        
        # Encoder (CNN based)
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), # 42x42
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 21x21
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, 3, stride=1, padding=1),
            LayerNorm2d(latent_dim)
        )
        # We'll use a 21x21 feature map for the canvas core
        self.canvas_hw = 21
        
        # Logic state initialization (learned tokens)
        self.num_logic_tokens = 8
        self.logic_token_init = nn.Parameter(torch.randn(1, self.num_logic_tokens, latent_dim) * 0.02)
        
        # Action embedding
        self.act_embedding = nn.Embedding(num_actions, latent_dim)
        
        # Cores (Simplified for speed)
        self.canvas_core = CanvasCoreLayer(latent_dim, 4, latent_dim*2, 0.1, self.canvas_hw)
        self.logic_core = LogicCoreLayer(latent_dim, 4, latent_dim*2, 0.1)
        
        # Predictors
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1)
        )
        
        # Decoder for reconstruction (Auxiliary task to prevent collapse)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 3, stride=1, padding=1), # 21x21 -> 21x21
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 21x21 -> 42x42
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), # 42x42 -> 84x84
            nn.Sigmoid() # Normalize to [0,1]
        )
        
    def _recursive_refinement(self, z_C_init, z_L_init):
        B = z_C_init.shape[0]
        z_C = z_C_init
        z_L = z_L_init
        
        for h in range(self.h_cycles):
            for l in range(self.l_cycles):
                # 1. Logic thoughts
                z_L = self.logic_core(z_L, z_C)
                
                # 2. Extract instruction (mean of logic tokens)
                # Apply LayerNorm or Tanh to stabilize the feedback loop
                # If instruction grows unbounded, AdaLN scales explode
                instruction = torch.tanh(z_L.mean(dim=1)) 
                
                # 3. Canvas refinement
                z_C = self.canvas_core(z_C, z_L, instruction)
                
        return z_C, z_L

    def forward(self, x, actions=None):
        """
        x: [B, T, 1, 84, 84]
        actions: [B, T] indices
        """
        if x.dim() == 4: # [B, 1, 84, 84]
            x = x.unsqueeze(1) # [B, 1, 1, 84, 84]
            
        B, T, C, H, W = x.shape
        device = x.device
        
        # Encode sequence
        x_flat = x.view(B*T, C, H, W)
        z_C_all = self.encoder_stem(x_flat) # [B*T, D, 21, 21]
        z_C_all = z_C_all.view(B*T, self.latent_dim, -1).transpose(1, 2) # [B*T, 21*21, D]
        
        # Initial logic tokens
        z_L_init = self.logic_token_init.expand(B*T, -1, -1)
        
        # If we have actions, inject them into logic tokens
        if actions is not None:
            act_emb = self.act_embedding(actions.view(-1)) # [B*T, D]
            # Replace first logic token with action embedding
            z_L_init = torch.cat([act_emb.unsqueeze(1), z_L_init[:, 1:]], dim=1)
            
        # Recursive pass
        z_C_ref, z_L_ref = self._recursive_refinement(z_C_all, z_L_init)
        
        # Reshape back to sequence
        z_C_ref = z_C_ref.view(B, T, -1, self.latent_dim)
        z_L_ref = z_L_ref.view(B, T, -1, self.latent_dim)
        
        return z_C_ref.squeeze(1), z_L_ref.squeeze(1)

    def predict_next(self, z_C, z_L, action):
        """
        Simple predictor on top of recursive states.
        """
        B = z_C.shape[0]
        act_emb = self.act_embedding(action) # [B, D]
        
        # Inject action into logic state
        z_L_next_init = torch.cat([act_emb.unsqueeze(1), z_L[:, 1:]], dim=1)
        
        # Run refinement again for "prediction" (thinking about the future)
        z_C_pred, z_L_pred = self._recursive_refinement(z_C, z_L_next_init)
        
        return z_C_pred, z_L_pred

# --- Agent and Training (Adapted from whole.py) ---

def preprocess(obs):
    img = Image.fromarray(obs).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

from networks import Actor, Critic

class TOPASAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        self.wm = TOPASWorldModel(LATENT_DIM, NUM_ACTIONS).to(device)
        self.actor = Actor(LATENT_DIM, NUM_ACTIONS).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        self.wm_optim = AdamW(self.wm.parameters(), lr=1e-4)
        self.ac_optim = AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=1e-4
        )
        self.state = None

    def reset_state(self):
        self.state = None

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
            z_seq, _ = self.wm(obs.to(self.device))
            # z_seq is [B, T, HW, D] or [B, HW, D] if T=1
            # For actor, we'll pool spatial tokens
            if z_seq.dim() == 4:
                state = z_seq[:, -1].mean(dim=1)
            else:
                state = z_seq.mean(dim=1)
            
            logits = self.actor(state)
            value = self.critic(state).squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
                
        return action, log_prob, value, state

    def train_world_model(self, batch):
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        B, T, C, H, W = obs_b.shape
        
        z_C, z_L = self.wm(obs_b, act_b.squeeze(-1).long())
        with torch.no_grad():
            z_C_next_target, _ = self.wm(next_obs_b)
            
        loss_wm = F.mse_loss(z_C, z_C_next_target)
        
        # Reward Prediction
        z_L_flat = z_L.view(B*T, -1, LATENT_DIM).mean(dim=1)
        pred_reward = self.wm.reward_predictor(z_L_flat)
        loss_reward = F.mse_loss(pred_reward, rew_b.view(-1, 1))

        # Reconstruction (Prevents collapse to zero)
        # z_C is [B, T, HW, D]. Need [B*T, D, H, W] for decoder
        B, T, HW, D = z_C.shape
        H_small = int(math.sqrt(HW)) # 21
        z_C_spatial = z_C.view(B*T, HW, D).transpose(1, 2).view(B*T, D, H_small, H_small)
        
        recon_imgs = self.wm.decoder(z_C_spatial)
        target_imgs = obs_b.view(B*T, 1, 84, 84)
        loss_recon = F.mse_loss(recon_imgs, target_imgs)
        
        total_loss = loss_wm + loss_reward + loss_recon
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.wm_optim.step()
        
        return {
            "loss_wm": loss_wm.item(), 
            "loss_reward": loss_reward.item(),
            "loss_recon": loss_recon.item(),
            "z_C_norm": z_C.norm().item() / (B*T),
            "z_L_norm": z_L.norm().item() / (B*T)
        }

    def update_ac(self, rollouts):
        states, actions, rewards, values, log_probs, dones = rollouts
        if not states: return {}
        T = len(rewards)
        B = rewards[0].size(0)
        rewards = torch.stack(rewards).view(T, B)
        values = torch.stack(values).view(T, B)
        log_probs = torch.stack(log_probs).view(T, B)
        
        gamma, lam = 0.99, 0.95
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(T)):
            next_val = 0 if t == T - 1 else values[t+1]
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_val * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        b_states = torch.stack(states).view(T*B, LATENT_DIM)
        b_actions = torch.stack(actions).view(T*B)
        b_log_probs = log_probs.view(T*B)
        b_returns = returns.view(T*B)
        b_advantages = advantages.view(T*B)
        
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
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.ac_optim.step()
            
        return {"loss_ac": loss.item(), "val_est": new_values.mean().item()}

def train():
    print(f"Using device: {DEVICE}")
    wandb.init(project="EnhancedTRM-TOPAS")
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    agent = TOPASAgent()
    replay_buffer = [] 
    current_episode = []
    
    obs, _ = env.reset()
    curr_frame = preprocess(obs)
    
    bs = 8
    rollout_len = 128
    max_episode_steps = 1000 # Force done after N steps to prevent infinite loops
    
    states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []
    episode_reward, episode_steps, episode_count = 0, 0, 0

    for step in range(50000):
        action, log_prob, value, state = agent.get_action(curr_frame.to(DEVICE))
        act_val = action.item()
        next_obs, reward, term, trunc, _ = env.step(act_val)
        
        episode_steps += 1
        done = term or trunc or (episode_steps >= max_episode_steps)
        if done and episode_steps >= max_episode_steps:
            trunc = True # Mark as truncated if timeout
        
        episode_reward += reward
        next_frame = preprocess(next_obs)
        
        current_episode.append((curr_frame, act_val, reward, next_frame, done))
        states_buf.append(state)
        actions_buf.append(action)
        rewards_buf.append(torch.tensor([reward], device=DEVICE))
        values_buf.append(value)
        log_probs_buf.append(log_prob)
        dones_buf.append(torch.tensor([done], device=DEVICE))
        
        curr_frame = next_frame
        if done:
            replay_buffer.append(current_episode)
            current_episode = []
            if len(replay_buffer) > 500: replay_buffer.pop(0)
            agent.reset_state()
            wandb.log({"env/episode_reward": episode_reward, "env/episode_steps": episode_steps}, step=step)
            print(f"Episode {episode_count} finished. Reward: {episode_reward}, Steps: {episode_steps}")
            sys.stdout.flush()
            episode_reward, episode_steps, episode_count = 0, 0, episode_count + 1
            obs, _ = env.reset()
            curr_frame = preprocess(obs)
            
        if len(states_buf) >= rollout_len:
            ac_metrics = agent.update_ac((states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf))
            wandb.log(ac_metrics, step=step)
            states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []

        if len(replay_buffer) > 2 and step % TRAIN_FREQ == 0:
            for _ in range(TRAIN_STEPS):
                b_obs, b_act, b_rew, b_next, b_done = [], [], [], [], []
                for _ in range(bs):
                    ep = random.choice(replay_buffer)
                    # Sample sequence
                    if len(ep) < SEQ_LEN:
                        # Improved padding
                        seq = ep + [ep[-1]] * (SEQ_LEN - len(ep))
                    else:
                        start_idx = random.randint(0, len(ep) - SEQ_LEN)
                        seq = ep[start_idx:start_idx + SEQ_LEN]
                        
                    b_obs.append(torch.stack([s[0] for s in seq]))
                    b_act.append(torch.tensor([s[1] for s in seq], dtype=torch.long).unsqueeze(1))
                    b_rew.append(torch.tensor([s[2] for s in seq], dtype=torch.float).unsqueeze(1))
                    b_next.append(torch.stack([s[3] for s in seq]))
                    b_done.append(torch.tensor([s[4] for s in seq], dtype=torch.float).unsqueeze(1))
                
                batch = (torch.stack(b_obs).to(DEVICE), torch.stack(b_act).to(DEVICE), 
                         torch.stack(b_rew).to(DEVICE), torch.stack(b_next).to(DEVICE), 
                         torch.stack(b_done).to(DEVICE))
                
                try:
                    wm_metrics = agent.train_world_model(batch)
                    wandb.log(wm_metrics, step=step)
                except RuntimeError as e:
                    print(f"Error in WM train: {e}")
                    # Skip step if error (e.g. NaN)
                    
    env.close()
def test():
    print(f"Using device: {DEVICE}")
    print("Testing TOPASWorldModel...")
    model = TOPASWorldModel(LATENT_DIM, NUM_ACTIONS).to(DEVICE)
    dummy_obs = torch.rand((2, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    dummy_act = torch.zeros((2, SEQ_LEN), dtype=torch.long).to(DEVICE)
    z_C, z_L = model(dummy_obs, dummy_act)
    print(f"z_C shape: {z_C.shape}, z_L shape: {z_L.shape}")
    print("Testing Agent Action...")
    agent = TOPASAgent(device=DEVICE)
    action, _, _, state = agent.get_action(dummy_obs[:, 0])
    print(f"Action: {action}, State shape: {state.shape}")
    print("Test Passed!")
    
    print("3. Testing Training Step...")
    # Create dummy batch
    # obs: [B, T, C, H, W]
    obs_b = torch.rand((4, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    act_b = torch.randint(0, NUM_ACTIONS, (4, SEQ_LEN, 1)).to(DEVICE)
    rew_b = torch.rand((4, SEQ_LEN, 1)).to(DEVICE)
    next_obs_b = torch.rand((4, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    done_b = torch.zeros((4, SEQ_LEN, 1)).to(DEVICE)
    
    batch = (obs_b, act_b, rew_b, next_obs_b, done_b)
    metrics = agent.train_world_model(batch)
    print(f"Training Metrics: {metrics}")
    print("Training Test Passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    else:
        test()

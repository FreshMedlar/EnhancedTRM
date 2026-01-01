import os
import sys
import math
import random
import warnings
from dataclasses import dataclass
from collections import deque

import gymnasium as gym
import ale_py
import numpy as np
import wandb
import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision.ops import MLP 
# Add base_models to sys.path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from base_models.rwkv import Block, LayerNorm
from base_models.lejepa import SIGReg, ViTEncoder

# --- Constants & Config ---
# --- Constants & Config ---
IMG_SIZE = 64
CHANNELS = 1
NUM_ACTIONS = 6 # Breakout-v5 has 6 actions (NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE)
LATENT_DIM = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 16
BATCH_SIZE = 16
TRAIN_FREQ = 10
TRAIN_STEPS = 1
GAMMA = 0.99
GAE_LAMBDA = 0.95  # Renamed to avoid confusion with LeJEPA lambda
LEJEPA_LAMBDA = 0.1

@dataclass
class RWKVConfig:
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    bias: bool = True
    dropout: float = 0.0

# --- World Model & Predictor ---

class RWKVPredictor(nn.Module):
    def __init__(self, proj_dim, n_layer=4, n_head=4):
        super().__init__()
        self.config = RWKVConfig(n_embd=proj_dim, n_layer=n_layer, n_head=n_head)
        self.blocks = nn.ModuleList([Block(self.config, i) for i in range(n_layer)])
        self.ln_out = LayerNorm(proj_dim, bias=True)

    def forward(self, z, actions):
        # z: (N, V, D)
        # actions: (N, V, D) - action embeddings
        x = z + actions
        for block in self.blocks:
            x = block(x)
        return self.ln_out(x)

class PixelRWKV(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder: 128x128 -> 16x16 (SEQ_LEN=256)
        # Encoder: 64x64   -> 8x8   (SEQ_LEN=64)
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> 32x32
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.GELU(),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1), # -> 8x8
            nn.GELU(),
        )
        
        # RWKV Core
        self.rwkv = RWKVPredictor(latent_dim, n_layer=2, n_head=4)
        
        # Decoder: 8x8 -> 64x64
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1), # -> 16x16
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # -> 32x32
            nn.GELU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # -> 64x64
            nn.Sigmoid() # Output pixels 0-1
        )

    def forward(self, obs, z):
        # obs: (B, 1, 64, 64)
        # z: (B, D) - latent of next state
        
        # 1. Compress Image
        feat = self.enc(obs) # (B, D, 8, 8)
        B, C, H, W = feat.shape
        
        # 2. Prepare Sequence
        seq = feat.flatten(2).transpose(1, 2) # (B, 64, D)
        
        # 3. Condition on z (Add to every token)
        seq = seq + z.unsqueeze(1)
        
        # 4. RWKV Processing (Note: dummy actions for predictor interface)
        dummy_act = torch.zeros_like(seq)
        seq_out = self.rwkv(seq, dummy_act) # (B, 64, D)
        
        # 5. Decode
        feat_out = seq_out.transpose(1, 2).reshape(B, C, H, W)
        rec = self.dec(feat_out) # (B, 1, 64, 64)
        return rec

class JEPAWorldModel(nn.Module):
    def __init__(self, proj_dim=128, num_actions=4):
        super().__init__()
        self.encoder = ViTEncoder(proj_dim, img_size=IMG_SIZE)
            
        self.predictor = RWKVPredictor(proj_dim)
        self.pixel_predictor = PixelRWKV(proj_dim)
        self.action_emb = nn.Embedding(num_actions, proj_dim)
        self.reward_predictor = nn.Linear(proj_dim, 1)
        
    def encode(self, obs):
        _, proj_emb = self.encoder(obs)
        return proj_emb.transpose(0, 1)

    def predict(self, z, actions):
        act_emb = self.action_emb(actions)
        return self.predictor(z, act_emb)

# --- Actor-Critic ---

class Actor(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
    def forward(self, x):
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x)

# --- Environment Hub ---

def preprocess(obs):
    # obs: (210, 160, 3)
    img = Image.fromarray(obs).convert('L') 
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img).astype(np.uint8) # Store as uint8 to save 4x memory
    return torch.tensor(arr, dtype=torch.uint8).unsqueeze(0)

class MBRLAgent:
    def __init__(self, num_actions, device=DEVICE):
        self.device = device
        self.wm = JEPAWorldModel(LATENT_DIM, num_actions=num_actions).to(device)
        self.sigreg = SIGReg().to(device)
        self.actor = Actor(LATENT_DIM, num_actions).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        self.wm_optim = AdamW(self.wm.parameters(), lr=1e-4)
        self.ac_optim = AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=1e-4)
        self.last_z_std = 0.0
        
    @torch.no_grad()
    def get_action(self, obs, deterministic=False):
        self.wm.eval()
        self.actor.eval()
        self.critic.eval()
        try:
            # obs: (C, H, W) -> (1, 1, C, H, W)
            obs_seq = obs.unsqueeze(0).unsqueeze(0).to(self.device)
            z_seq = self.wm.encode(obs_seq)
            z = z_seq.squeeze(0).squeeze(0) # (D)
            
            logits = self.actor(z)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            return action, dist.log_prob(action), self.critic(z).squeeze(-1)
        finally:
            self.wm.train()
            self.actor.train()
            self.critic.train()

    def train_world_model(self, batch):
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        # obs_b/next_obs_b: (B, T, 1, H, W) float32
        B, T, _, _, _ = obs_b.shape
        
        # 1. Faster Encoding: Encode all unique frames in one pass
        full_seq = torch.cat([obs_b, next_obs_b[:, -1:]], dim=1) # (B, T+1, 1, H, W)
        all_z = self.wm.encode(full_seq) # (B, T+1, D)
        
        z = all_z[:, :-1]    # (B, T, D)
        z_next_target = all_z[:, 1:] # (B, T, D)
            
        # 2. Predict
        z_next_pred = self.wm.predict(z, act_b.squeeze(-1)) # (B, T, D)
        
        # 3. Reward Prediction
        rew_pred = self.wm.reward_predictor(z) # (B, T, 1)
        
        # 4. Pixel Prediction (PixelRWKV)
        # Flatten obs/next_obs: (B, T, 1, H, W) -> (B*T, 1, H, W)
        obs_flat = obs_b.flatten(0, 1) # (B*T, 1, H, W)
        z_next_flat = z_next_pred.flatten(0, 1) # (B*T, D)
        next_obs_flat = next_obs_b.flatten(0, 1) # (B*T, 1, H, W)
        
        rec_next = self.wm.pixel_predictor(obs_flat, z_next_flat) # (B*T, 1, H, W)
        
        # 5. Losses
        loss_pred = F.mse_loss(z_next_pred, z_next_target)
        loss_reward = F.mse_loss(rew_pred, rew_b)
        loss_pixel = F.mse_loss(rec_next, next_obs_flat)
        
        # SIGReg (on student latents)
        loss_sig = self.sigreg(z.reshape(B*T, -1))
        
        # LeJEPA style weighted loss: lamb * SIGReg + (1-lamb) * Invariance(Prediction)
        lejepa_loss = loss_sig * LEJEPA_LAMBDA + loss_pred * (1 - LEJEPA_LAMBDA)
        
        total_loss = lejepa_loss + loss_reward + loss_pixel # Add pixel loss
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        self.wm_optim.step()
        
        with torch.no_grad():
            self.last_z_std = z.std(dim=(0, 1)).mean().item()
            z_target_std = z_next_target.std(dim=(0, 1)).mean().item()

        return {
            "wm/loss_total": total_loss.item(),
            "wm/loss_lejepa": lejepa_loss.item(),
            "wm/loss_pred": loss_pred.item(),
            "wm/loss_reward": loss_reward.item(),
            "wm/loss_pixel": loss_pixel.item(),
            "wm/loss_sig": loss_sig.item(),
            "wm/z_std": self.last_z_std,
            "wm/z_target_std": z_target_std,
        }

    def update_ac(self, rollouts):
        states, actions, rewards, values, log_probs, dones = rollouts
        T = len(rewards)
        B = rewards[0].size(0)
        
        rewards = torch.stack(rewards).view(T, B)
        values = torch.stack(values).view(T, B)
        log_probs = torch.stack(log_probs).view(T, B)
        dones = torch.stack(dones).view(T, B).float()
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            next_val = values[t+1] if t < T - 1 else 0
            delta = rewards[t] + GAMMA * next_val * mask - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * mask * last_gae
            advantages[t] = last_gae
            
        returns = advantages + values
        
        b_states = torch.stack(states).view(T*B, -1)
        b_actions = torch.stack(actions).view(T*B)
        b_log_probs = log_probs.view(T*B)
        b_returns = returns.view(T*B)
        b_advantages = (advantages.view(T*B) - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(3):
            logits = self.actor(b_states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(b_actions)
            new_values = self.critic(b_states).squeeze(-1)
            
            ratio = (new_log_probs - b_log_probs).exp()
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * b_advantages
            
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist.entropy().mean()
            critic_loss = F.mse_loss(new_values, b_returns)
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.ac_optim.zero_grad()
            loss.backward()
            self.ac_optim.step()
            
        return {"ac/loss": loss.item(), "ac/val": values.mean().item()}

def main():
    print(f"Starting JEPA-RWKV Training on {DEVICE}", flush=True)
    wandb.init(project="jepa-rwkv", name="vit-rwkv-breakout")
    
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    num_actions = env.action_space.n
    agent = MBRLAgent(num_actions=num_actions)
    
    replay_buffer = []
    current_episode = []
    
    obs, _ = env.reset()
    curr_frame = preprocess(obs)
    
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    
    states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []
    
    pbar = tqdm.tqdm(range(50000))
    for step in pbar:
        # 1. Action
        action, log_prob, val = agent.get_action(curr_frame.float() / 255.0)
        act_val = action.item()
        
        next_obs, reward, term, trunc, _ = env.step(act_val)
        done = term or trunc
        next_frame = preprocess(next_obs)
        
        # Store for RL
        with torch.no_grad():
            agent.wm.eval()
            z = agent.wm.encode(curr_frame.unsqueeze(0).unsqueeze(0).to(DEVICE).float() / 255.0).squeeze(0).squeeze(0)
            agent.wm.train()
            
        states_buf.append(z)
        actions_buf.append(action)
        rewards_buf.append(torch.tensor([reward], device=DEVICE))
        values_buf.append(val)
        log_probs_buf.append(log_prob)
        dones_buf.append(torch.tensor([done], device=DEVICE))
        
        # Store for WM (Store only curr_frame to save memory, next_frame is redundant)
        current_episode.append((curr_frame, act_val, reward, done))
        
        curr_frame = next_frame
        episode_reward += reward
        episode_steps += 1
        
        if done:
            # Store terminal frame so next_obs lookup works for the final step
            # We use 0 actions/rewards as dummies for the terminal state
            current_episode.append((curr_frame, 0, 0.0, 1.0))
            
            replay_buffer.append(current_episode)
            if len(replay_buffer) > 100: replay_buffer.pop(0)
            current_episode = []
            
            wandb.log({
                "env/reward": episode_reward,
                "env/steps": episode_steps,
                "env/episode": episode_count
            }, step=step)
            
            pbar.set_description(f"Ep {episode_count} Rew {episode_reward} z_std {agent.last_z_std:.2f}")
            
            episode_reward = 0
            episode_steps = 0
            episode_count += 1
            obs, _ = env.reset()
            curr_frame = preprocess(obs)
            
        # 2. Train AC
        if len(states_buf) >= 128:
            ac_metrics = agent.update_ac((states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf))
            wandb.log(ac_metrics, step=step)
            states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []
            
        # 3. Train WM
        if step % TRAIN_FREQ == 0 and len(replay_buffer) > 2:
            wm_metrics_accum = {}
            for _ in range(TRAIN_STEPS):
                b_obs, b_act, b_rew, b_next, b_done = [], [], [], [], []
                for _ in range(BATCH_SIZE):
                    ep = random.choice(replay_buffer)
                    if len(ep) < SEQ_LEN:
                        seq = ep + [ep[-1]] * (SEQ_LEN - len(ep))
                    else:
                        start = random.randint(0, len(ep) - SEQ_LEN)
                        seq = ep[start:start+SEQ_LEN]
                    
                    b_obs.append(torch.stack([s[0] for s in seq]))
                    b_act.append(torch.tensor([s[1] for s in seq], dtype=torch.long).unsqueeze(-1))
                    b_rew.append(torch.tensor([s[2] for s in seq], dtype=torch.float32).unsqueeze(-1))
                    # next_obs is obs of next step
                    # To get next_obs for the last step in seq, we need the following step in 'ep'
                    # We ensure ep is long enough by picking start < len(ep) - SEQ_LEN
                    # Or we compute it during construction
                    
                    # More robust sequence building
                    obs_seq = torch.stack([s[0] for s in seq])
                    
                    # Getting the actual next observations (next step in episode)
                    # We need the 's[0]' from the NEXT index in the episode
                    next_obs_tensors = []
                    for i in range(start, start + SEQ_LEN):
                        # ep[i+1][0] is the next frame. If it's the end, repeat last.
                        next_idx = min(i + 1, len(ep) - 1)
                        next_obs_tensors.append(ep[next_idx][0])
                    next_obs_seq = torch.stack(next_obs_tensors)

                    b_next.append(next_obs_seq)
                    b_done.append(torch.tensor([s[3] for s in seq], dtype=torch.float32).unsqueeze(-1))
                
                batch = (
                    torch.stack(b_obs).to(DEVICE).float() / 255.0,
                    torch.stack(b_act).to(DEVICE),
                    torch.stack(b_rew).to(DEVICE),
                    torch.stack(b_next).to(DEVICE).float() / 255.0,
                    torch.stack(b_done).to(DEVICE)
                )
                metrics = agent.train_world_model(batch)
                for k, v in metrics.items():
                    wm_metrics_accum[k] = wm_metrics_accum.get(k, 0) + v / TRAIN_STEPS
            
            wandb.log(wm_metrics_accum, step=step)

    env.close()

if __name__ == "__main__":
    main()

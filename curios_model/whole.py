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

from networks import Actor, Critic
from jepa import JEPAWorldModel, SIGReg

# Constants
IMG_SIZE = 84
NUM_ACTIONS = 4
LATENT_DIM = 128 # d_model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LEN = 8 # Training sequence length (short for test speed, aim higher for real training)

FRAME_STACK = 1
TRAIN_FREQ = 50 # Train every N steps
TRAIN_STEPS = 5 # How many batches per train step
def preprocess(obs):
    """
    Resize observation to 84x84, convert to grayscale, and return as tensor.
    """
    img = Image.fromarray(obs).convert("L") # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST) # Nearest for small objects
    
    arr = np.array(img).astype(np.float32) / 255.0
    # No batch dim here, just (1, H, W)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0) # (1, 84, 84)

def stack_frames(frames_deque):
    # Just return the latest frame if stack is 1
    return frames_deque[-1].unsqueeze(0) # (1, 1, 84, 84)

class CuriosityEnsemble(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_models=5):
        """
        LEXA-style Ensemble for calculating Model Disagreement.
        
        Args:
            state_dim (int): Dimension of the latent state (z).
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Size of hidden layers.
            num_models (int): Number of models in the ensemble (usually 5 or 10).
        """
        super().__init__()
        self.models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim) # Predicts next latent state
            ) for _ in range(num_models)
        ])
    
    def forward(self, state, action):
        """
        Returns predictions from all models in the ensemble.
        Shape: [num_models, batch_size, state_dim]
        """
        # Concatenate state and action - action needs to be one-hot or embedding? 
        # In this codebase, action is usually an index passed to an embedding layer, 
        # but here the input is usually expected to be the raw features.
        # However, the JEPAWorldModel uses an embedding layer for actions.
        # We should probably use the action embedding here too to match dimensions if passed, 
        # OR we can just one-hot encode it. 
        # Looking at lexa.md, it assumes `action` is concatenated.
        # To make it compatible with the rest of the file which uses embeddings, 
        # we might need to handle action embeddings outside or pass them in.
        # Let's assume we pass in the action EMBEDDING or ONE-HOT. 
        # For simplicity and consistency with JEPA, let's pass ONE-HOT since this is a separate module.
        # Actually, let's stick to the plan: follow lexa.md but adapt. 
        # MBRLAgent uses action embeddings. Let's change the input to expect action embeddings or do the embedding inside.
        # Since this is a separate module, let's pass the Action Embedding to it? 
        # Or just one-hot it. One-hot is safer for "raw" action input.
        
        x = torch.cat([state, action], dim=-1)
        
        # Stack predictions from all models
        predictions = torch.stack([model(x) for model in self.models])
        return predictions

    def get_intrinsic_reward(self, state, action):
        """
        THE EXPLORER REWARD: Calculates variance across ensemble predictions.
        High Variance = High Curiosity Reward.
        """
        with torch.no_grad():
            predictions = self.forward(state, action)
            # Calculate variance across the 'num_models' dimension (dim 0)
            # We average the variance over the feature dimension to get a single scalar per batch item
            disagreement = torch.var(predictions, dim=0).mean(dim=-1)
            
        return disagreement.unsqueeze(-1) # Shape: [batch_size, 1]

    def get_goal_reward(self, current_state, goal_state):
        """
        THE ACHIEVER REWARD: Negative distance to the goal.
        You can use Euclidean or Cosine distance.
        """
        # Negative Euclidean distance (minimize distance = maximize reward)
        distance = torch.norm(current_state - goal_state, p=2, dim=-1)
        return -distance.unsqueeze(-1)

    def train_ensemble(self, state, action, next_state_target, optimizer):
        """
        Trains the ensemble to predict the next state correctly.
        Standard MSE Loss.
        """
        predictions = self.forward(state, action)
        
        # Expand target to match ensemble size: [num_models, batch, state_dim]
        target = next_state_target.unsqueeze(0).expand_as(predictions)
        
        # MSE Loss summed over all models
        loss = F.mse_loss(predictions, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()


class MBRLAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        
        self.wm = JEPAWorldModel(LATENT_DIM, NUM_ACTIONS, in_channels=FRAME_STACK, img_size=IMG_SIZE).to(device)
        self.sigreg = SIGReg().to(device)
        
        self.actor = Actor(LATENT_DIM, NUM_ACTIONS).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        # Lexa Curiosity
        self.curiosity = CuriosityEnsemble(LATENT_DIM, LATENT_DIM).to(device)
        self.ensemble_opt = AdamW(self.curiosity.parameters(), lr=1e-4)

        self.wm_optim = AdamW(self.wm.parameters(), lr=1e-4) 
        self.ac_optim = AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=1e-4
        )
        self.state = None # Recurrent state
    
    def reset_state(self):
        self.state = None
    
    def get_intrinsic_reward(self, state, action):
        """
        Calculate intrinsic reward given latent state and action index.
        """
        # One-hot encode action
        # action can be a single int or a tensor
        if isinstance(action, int):
            action = torch.tensor([action], device=self.device)
        
        if action.dim() == 0:
            action = action.unsqueeze(0)
            
        with torch.no_grad():
            act_emb = self.wm.act_embedding(action.long())
            # state might be (D,) if from single obs, or (B, D)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if act_emb.dim() == 1:
                act_emb = act_emb.unsqueeze(0)
        
        return self.curiosity.get_intrinsic_reward(state, act_emb)

    def train_ensemble(self, batch):
        """
        Train the curiosity ensemble on a batch.
        """
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        
        with torch.no_grad():
            # For ensemble, we need (B, D) states, not (B, T, D)
            # So we take the last state of the sequence for prediction
            z_t_seq, _ = self.wm.encode(obs_b, state=None)
            z_t = z_t_seq[:, -1, :] # (B, D)
            
            z_next_seq, _ = self.wm.encode(next_obs_b, state=None)
            z_next = z_next_seq[:, -1, :] # (B, D)
            
                
        # act_b is (B, T, 1) indices. Take the last action for prediction.
        act_indices = act_b[:, -1, 0].long()
        act_emb = self.wm.act_embedding(act_indices)
        
        loss = self.curiosity.train_ensemble(z_t, act_emb, z_next, self.ensemble_opt)
        return {"loss_ensemble": loss}

    def get_action(self, obs, deterministic=False):
        with torch.no_grad():
            # Encode observation to latent state
            # obs is (1, C, H, W)
            # wm.encode expects (B, T, C, H, W) or (B, C, H, W)
            obs_seq = obs.unsqueeze(0).unsqueeze(0) # (1, 1, C, H, W)
            z_seq, self.state = self.wm.encode(obs_seq, state=self.state)
            state = z_seq.squeeze(0).squeeze(0) # (1, 1, D) -> (D)
            
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
        # Batch is sequences: (B, T, C, H, W) and (B, T, 1)
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        B, T, C, H, W = obs_b.shape
        
        # 1. Encode Sequence
        # Since we train on random sequences, we don't have the context (state) from before T=0.
        # We start with state=None (Learned initial state or Zero).
        proj_seq, _ = self.wm.encode(obs_b, state=None) # (B, T, D)
        
        # For targets, we can run encoder on next_obs_b. 
        # Ideally we shift target, but here we have explicit next_obs.
        with torch.no_grad():
             proj_next_seq, _ = self.wm.encode(next_obs_b, state=None) # (B, T, D)
        
        # 2. Predict next latent
        act_emb = self.wm.act_embedding(act_b.squeeze(-1)) # (B, T, D)
        pred_proj_next = self.wm.predictor_net(proj_seq, act_emb) # (B, T, D)
        
        # 3. Losses
        loss_pred = F.mse_loss(pred_proj_next, proj_next_seq)
        
        # Reward Prediction
        pred_reward = self.wm.reward_predictor(proj_seq, act_emb)
        loss_reward = F.mse_loss(pred_reward, rew_b) # rew_b is (B, T, 1)

        # Reshape for SIGReg (B*T, D)
        # Assuming SIGReg implementation handles scaling correctly.
        # Our SIGReg in jepa.py scales by proj.size(-2) which is T here if we pass (B, T, D).
        # But lexa.md / jepa.py expect (B, D). Let's maximize entropy over the batch*time?
        # Or just Batch? Let's use Batch.
        flat_proj = proj_seq.reshape(B*T, -1)
        loss_sig = self.sigreg(flat_proj)

        lamb = 0.02 
        total_loss = (1 - lamb) * loss_pred + lamb * loss_sig + loss_reward
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        self.wm_optim.step()
        
        return {
            "loss_wm": loss_pred.item(),
            "loss_reward": loss_reward.item(),
            "loss_sigreg": loss_sig.item(),
            "rep/std": flat_proj.std(dim=0).mean().item(),
            "rep/avg_norm": flat_proj.norm(p=2, dim=1).mean().item(),
            "rep/pairwise_dist": torch.pdist(flat_proj).mean().item()
        }

    def update_ac(self, rollouts):
        """
        Standard PPO Update on collected rollouts.
        """
        states, actions, rewards, values, log_probs, dones = rollouts
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
            # If t is the last step in rollout, or next step is done
            if t == T - 1:
                next_val = 0 # Or bootstrap from critic(next_state) if we had it
            else:
                next_val = values[t+1]

            # Mask delta and GAE if next step is done
            # reward[t] leads to next_obs which might be 'done'
            # Typically: GAE_{t} = delta_t + gamma * lam * (1-done_{t+1}) * GAE_{t+1}
            mask = 1.0 - dones[t].float()
            delta = rewards[t] + gamma * next_val * mask - values[t]
            last_gae = delta + gamma * lam * mask * last_gae
            advantages[t] = last_gae
            
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
    
    # Replay Buffer: List of Episodes
    # Each episode is a list of (obs, act, rew, next_obs, done)
    replay_buffer = [] 
    current_episode = []
    
    print("Pre-filling buffer...")
    obs, _ = env.reset()
    
    # Preprocess returns (1, 84, 84)
    curr_frame = preprocess(obs)
    
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        next_frame = preprocess(next_obs)
        
        current_episode.append((curr_frame, action, reward, next_frame, done))
        
        curr_frame = next_frame
        if done:
            replay_buffer.append(current_episode)
            current_episode = []
            obs, _ = env.reset()
            curr_frame = preprocess(obs)
            
    print("Starting Training...")
    bs = 8  # Reduced from 32 to fit in 8GB VRAM
    
    rollout_len = 128
    states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []

    # Trackers
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    frames = [] # To store frames for video logging
    action_counts = {i: 0 for i in range(NUM_ACTIONS)}

    for step in range(50000): # Increased steps
        # 1. Interact
        # Ensure state is reset if starting new episode (usually handled by done check below)
        action, log_prob, value, state = agent.get_action(curr_frame.to(DEVICE))
        
        # Intrinsic Reward
        intr_reward = agent.get_intrinsic_reward(state, action)
        
        act_val = action.item()
        action_counts[act_val] += 1
        next_obs, reward, term, trunc, info = env.step(act_val)
        done = term or trunc
        
        episode_reward += reward
        episode_steps += 1
        
        # Combined Reward for Agent
        total_reward = reward + 10 * intr_reward.item()
        
        # New frame
        next_frame = preprocess(next_obs)
        
        # Store in current episode
        current_episode.append((curr_frame, act_val, reward, next_frame, done))
        
        # Store for AC update
        states_buf.append(state)
        actions_buf.append(action)
        rewards_buf.append(torch.tensor([total_reward], device=DEVICE))
        values_buf.append(value)
        log_probs_buf.append(log_prob)
        dones_buf.append(torch.tensor([done], device=DEVICE))
        
        # Capture frame for video
        if episode_count % 10 == 0: # Log video every 10 episodes to save space/time
            frames.append(next_obs)
        
        curr_frame = next_frame
        
        if done:
            # Store episode
            replay_buffer.append(current_episode)
            current_episode = []
            if len(replay_buffer) > 500: # Limit buffer size by episodes
                 replay_buffer.pop(0)

            # Reset Agent State for new episode
            agent.reset_state()

            # Log episode metrics
            log_dict = {
                "env/episode_reward": episode_reward,
                "env/episode_steps": episode_steps,
                "env/episode_count": episode_count,
            }
            
            if frames:
                video_array = np.stack(frames)
                log_dict["env/gameplay"] = wandb.Video(video_array, fps=30, format="gif")
                frames = []
            
            for a in range(NUM_ACTIONS):
                log_dict[f"actions/act_{a}"] = action_counts[a]
                action_counts[a] = 0
                
            wandb.log(log_dict, step=step)
            
            print(f"Episode {episode_count} finished. Reward: {episode_reward}, Steps: {episode_steps}")
            
            episode_reward = 0
            episode_steps = 0
            episode_count += 1
            
            obs, _ = env.reset()
            curr_frame = preprocess(obs)
            
        # 2. Update AC periodically
        if len(states_buf) >= rollout_len:
            rollouts = (states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf)
            ac_metrics = agent.update_ac(rollouts)
            wandb.log(ac_metrics, step=step)
            states_buf, actions_buf, rewards_buf, values_buf, log_probs_buf, dones_buf = [], [], [], [], [], []

        # 3. Update World Model (Sequence Training)
        if len(replay_buffer) > 2 and step % TRAIN_FREQ == 0:
            for _ in range(TRAIN_STEPS):
                # Sample Batch of Sequences
                b_obs, b_act, b_rew, b_next, b_done = [], [], [], [], []
            
                for _ in range(bs):
                    # Sample random episode
                    ep = random.choice(replay_buffer)
                    # Ensure episode is long enough or handle short ones
                    if len(ep) < SEQ_LEN:
                        # Pad if necessary, but simple way is to just repeat last frame
                        seq = ep[:]
                        while len(seq) < SEQ_LEN:
                            seq.append(ep[-1])
                    else:
                        max_start = len(ep) - SEQ_LEN
                        start_idx = random.randint(0, max_start)
                        seq = ep[start_idx:start_idx + SEQ_LEN]
                    
                    # Unzip sequence
                    # s: (1, 84, 84), a: int, r: float, sn: (1, 84, 84), d: bool
                    # We need to stack them into (T, ...)
                    t_obs = torch.stack([s[0] for s in seq]) # (T, 1, 84, 84)
                    t_act = torch.tensor([s[1] for s in seq], dtype=torch.long).unsqueeze(1) # (T, 1)
                    t_rew = torch.tensor([s[2] for s in seq], dtype=torch.float).unsqueeze(1) # (T, 1)
                    t_next = torch.stack([s[3] for s in seq]) # (T, 1, 84, 84)
                    t_done = torch.tensor([s[4] for s in seq], dtype=torch.float).unsqueeze(1) # (T, 1)
                    
                    b_obs.append(t_obs)
                    b_act.append(t_act)
                    b_rew.append(t_rew)
                    b_next.append(t_next)
                    b_done.append(t_done)
                
                # Stack batch: (B, T, ...)
                b_obs = torch.stack(b_obs).to(DEVICE)
                b_act = torch.stack(b_act).to(DEVICE)
                b_rew = torch.stack(b_rew).to(DEVICE)
                b_next = torch.stack(b_next).to(DEVICE)
                b_done = torch.stack(b_done).to(DEVICE)
                
                wm_metrics = agent.train_world_model((b_obs, b_act, b_rew, b_next, b_done))
                ens_metrics = agent.train_ensemble((b_obs, b_act, b_rew, b_next, b_done))
                wm_metrics.update(ens_metrics)
                wandb.log(wm_metrics, step=step)
                
                # Free up GPU memory
                del b_obs, b_act, b_rew, b_next, b_done
                torch.cuda.empty_cache()
            
        if step % 100 == 0:
            print(f"Step {step}, Intr: {intr_reward.item():.4f}")
            
    env.close()

def test():
    print(f"Using device: {DEVICE}")
    print("Testing MBRLAgent (Verification)...")
    agent = MBRLAgent()
    
    # 1. Test Forward Pass
    print("1. Testing World Model Forward Pass...")
    # Batch of sequences (B, T, C, H, W)
    # T=SEQ_LEN=8
    dummy_obs = torch.rand((2, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE) # Float
    dummy_act = torch.zeros((2, SEQ_LEN, 1), dtype=torch.long).to(DEVICE)
    dummy_next = torch.rand((2, SEQ_LEN, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
    
    # Updated batch for Goal-Conditioned (no rew/done needed in WM update, but function signature expects them unpacked)
    dummy_rew = torch.zeros((2, SEQ_LEN, 1)).to(DEVICE)
    dummy_done = torch.zeros((2, SEQ_LEN, 1)).to(DEVICE)
    
    batch = (dummy_obs, dummy_act, dummy_rew, dummy_next, dummy_done)
    metrics = agent.train_world_model(batch)
    print(f"WM Loss: {metrics['loss_wm']:.4f}")
    
    # 3. Test Action
    print("3. Testing Actor...")
    # Single frame inference
    single_obs = torch.rand((1, 84, 84)).to(DEVICE)
    action, log_prob, value, state = agent.get_action(single_obs)
    print(f"Action selected: {action}")
    print(f"State shape: {state[0].shape if isinstance(state, tuple) else state.shape}") # Check if state is returned
    
    # 4. Test Curiosity
    print("4. Testing Curiosity...")
    intr = agent.get_intrinsic_reward(state, action)
    print(f"Intrinsic Reward: {intr.item()}")
    
    # Train Ensemble needs (B, D) tensors, extracted from sequence in train_world_model, 
    # but train_ensemble expects batch tuple.
    ens_metrics = agent.train_ensemble(batch)
    print(f"Ensemble Loss: {ens_metrics['loss_ensemble']:.4f}")

    print("Test Passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train()
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        test()
    else:
        test()

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
import sys
import os

from tiny_agent import TinyRecursiveAgent

# Constants
IMG_SIZE = 84
NUM_ACTIONS = 4
HIDDEN_DIM = 64
RECURSION_ITERS = 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 3e-4
ROLLOUT_LEN = 128
PPO_EPOCHS = 3
MINI_BATCH_SIZE = 32

def preprocess(obs):
    """Resize, Grayscale, Normalize. Returns (1, 84, 84)"""
    img = Image.fromarray(obs).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.hidden_states = [] # Store hidden state used for this step

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        self.hidden_states = []

    def store(self, state, action, reward, value, log_prob, done, h):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.hidden_states.append(h)

def train_tiny():
    print(f"Using device: {DEVICE}")
    wandb.init(project="EnhancedTRM-Tiny")
    
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    
    agent = TinyRecursiveAgent(
        num_actions=NUM_ACTIONS, 
        hidden_dim=HIDDEN_DIM, 
        num_iterations=RECURSION_ITERS
    ).to(DEVICE)
    
    optimizer = AdamW(agent.parameters(), lr=LR)
    
    print(f"Agent Parameters: {agent.get_parameter_count()}")
    
    memory = PPOMemory()
    
    obs, _ = env.reset()
    curr_frame = preprocess(obs).to(DEVICE) # (1, 84, 84)
    
    # Initial hidden stat
    curr_h = torch.zeros(1, HIDDEN_DIM).to(DEVICE)
    
    step = 0
    episode_reward = 0
    episode_count = 0
    
    while step < 100000: # Training Loop
        # 1. Collect Rollout
        agent.eval()
        with torch.no_grad():
            # Add batch dim -> (1, 1, 84, 84)
            logits, value, next_h, _ = agent(curr_frame.unsqueeze(0), curr_h)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        act_val = action.item()
        next_obs, reward, term, trunc, _ = env.step(act_val)
        done = term or trunc
        
        memory.store(
            curr_frame.cpu(), 
            action.cpu(), 
            reward, 
            value.item(), 
            log_prob.cpu(), 
            done, 
            curr_h.cpu() # Store PREVIOUS h, which was used to generate this action
        )
        
        episode_reward += reward
        step += 1
        
        curr_frame = preprocess(next_obs).to(DEVICE)
        curr_h = next_h
        
        if done:
            curr_h = torch.zeros(1, HIDDEN_DIM).to(DEVICE) # Reset hidden on done
            obs, _ = env.reset()
            curr_frame = preprocess(obs).to(DEVICE)
            
            wandb.log({
                "env/episode_reward": episode_reward, 
                "env/episode_step": step,
                "env/episodes": episode_count
            }, step=step)
            print(f"Episode {episode_count} Done. Reward: {episode_reward}")
            
            episode_reward = 0
            episode_count += 1
            
        # 2. Update PPO
        if len(memory.states) >= ROLLOUT_LEN:
            agent.train()
            
            # Convert memory to tensors
            # Stack: (T, ...)
            old_states = torch.stack(memory.states).to(DEVICE) # (T, 1, 84, 84)
            old_actions = torch.stack(memory.actions).to(DEVICE).squeeze() # (T)
            old_log_probs = torch.stack(memory.log_probs).to(DEVICE).squeeze() # (T)
            old_h = torch.stack(memory.hidden_states).to(DEVICE).squeeze(1) # (T, D)
            
            # GAE Calculation
            rewards = memory.rewards
            values = memory.values
            dones = memory.dones
            
            returns = []
            advantages = []
            gae = 0
            
            # Bootstrap value for last step
            with torch.no_grad():
                 _, next_val, _, _ = agent(curr_frame.unsqueeze(0), curr_h)
                 next_val = next_val.item()
                 
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - float(dones[t]) # Should use next done?? No, dones[t] is if THIS step ended
                    # If this step ended, next val is 0. If not, use stored next_value (conceptually)
                    # Actually standard GAE:
                    v_next = next_val
                else:
                    next_non_terminal = 1.0 - float(dones[t]) 
                    v_next = values[t+1]
                
                delta = rewards[t] + 0.99 * v_next * next_non_terminal - values[t]
                gae = delta + 0.99 * 0.95 * next_non_terminal * gae
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
                
            returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(DEVICE)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO Epochs
            for _ in range(PPO_EPOCHS):
                # We need to re-run the model on the batch. 
                # Since it's recurrent, we should technically unroll, but for "Tiny" we can use stored hidden states?
                # Stored 'old_h' are the states associated with 'old_states'.
                # Wait, 'old_h' is PREVIOUS h. 
                # So we can just run one step: agent(state, old_h) -> new_logits, new_val
                # This works because we treat steps as independent given the hidden state in this simplified loop.
                # Valid for truncated BPTT with window 1 (which we sort of do here by detaching h).
                
                new_logits, new_val, _, all_logits = agent(old_states, old_h)
                
                # Main Output Loss
                dist = torch.distributions.Categorical(logits=new_logits)
                new_log_probs = dist.log_prob(old_actions)
                entropy = dist.entropy().mean()
                
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_val.squeeze(), returns)
                
                # Deep Supervision Loss
                # Apply PPO-like advantage learning to *every* iteration's output? 
                # Or just CrossEntropy if we had ground truth.
                # Since we don't have ground truth, let's just force them to match the "best" action 
                # or just regularize them. 
                # Simplest Deep Supervision for RL:
                # Apply the same Actor Loss to all intermediate logits
                deep_sup_loss = 0
                for logits_i in all_logits:
                    dist_i = torch.distributions.Categorical(logits=logits_i)
                    log_probs_i = dist_i.log_prob(old_actions)
                    ratio_i = (log_probs_i - old_log_probs).exp() # Reuse old_log_probs? Approximation.
                    # Better: just maximize prob of the action taken (which was good)? 
                    # Or reuse advantages.
                    surr1_i = ratio_i * advantages
                    surr2_i = torch.clamp(ratio_i, 0.8, 1.2) * advantages
                    deep_sup_loss += -torch.min(surr1_i, surr2_i).mean()
                
                deep_sup_loss = deep_sup_loss / RECURSION_ITERS
                
                total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + 0.5 * deep_sup_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
            memory.clear()
            
    env.close()

def test_dry_run():
    print(f"Testing Tiny Agent on {DEVICE}...")
    agent = TinyRecursiveAgent(NUM_ACTIONS, HIDDEN_DIM, RECURSION_ITERS).to(DEVICE)
    
    # Input
    dummy_obs = torch.randn(2, 1, 84, 84).to(DEVICE)
    dummy_h = torch.zeros(2, HIDDEN_DIM).to(DEVICE)
    
    # Forward
    logits, val, next_h, all_logs = agent(dummy_obs, dummy_h)
    
    print(f"Logits: {logits.shape}")
    print(f"Value: {val.shape}")
    print(f"Next H: {next_h.shape}")
    print(f"All Logits: {all_logs.shape}")
    
    assert logits.shape == (2, NUM_ACTIONS)
    assert val.shape == (2, 1)
    assert next_h.shape == (2, HIDDEN_DIM)
    assert all_logs.shape == (RECURSION_ITERS, 2, NUM_ACTIONS)
    
    print("Dry run passed!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_dry_run()
    else:
        train_tiny()

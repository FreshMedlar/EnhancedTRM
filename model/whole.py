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

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import from local project structure
from trm_lucid import TinyRecursiveModel
from mlp_mixer_1d import MLPMixer1D

# Hyperparameters
SEQ_LEN = 32 * 32 + 1  # 1024 pixels + 1 action
IMG_SIZE = 32
NUM_ACTIONS = 4
LATENT_DIM = 256
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


class RewardModel(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, latents):
        # latents: (B, Seq, Dim) -> Pooled: (B, Dim)
        # We use mean pooling over the sequence dimension
        pooled = latents.mean(dim=1)
        return self.net(pooled)

class Actor(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, latents):
        # latents: (B, Seq, Dim)
        pooled = latents.mean(dim=1)
        return self.net(pooled)

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, latents):
        # latents: (B, Seq, Dim)
        pooled = latents.mean(dim=1)
        return self.net(pooled)

class ObservationModel(nn.Module):
    def __init__(self, latent_dim, num_tokens=256):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_tokens)
        )

    def forward(self, obs, context_latents, labels=None):
        # obs: (B, 1024) - current pixels
        # context_latents: (B, 1024, latent_dim) - dynamics from EM
        obs_emb = self.embedding(obs)
        combined = torch.cat([obs_emb, context_latents], dim=-1) # (B, 1024, latent_dim * 2)
        logits = self.net(combined) # (B, 1024, 256)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 256), labels.view(-1))
            
        return logits, loss

class MBRLAgent:
    def __init__(self, device=DEVICE):
        self.device = device
        
        # Evolution Model (EM): (Obs, Action) -> Latents
        # Predicts the evolution of the world state
        em_network = MLPMixer1D(
            dim=LATENT_DIM,
            depth=2,
            seq_len=SEQ_LEN # 1025 (1024 + 1)
        )
        self.evolution_model = TinyRecursiveModel(
            dim=LATENT_DIM,
            num_tokens=256 + NUM_ACTIONS, 
            network=em_network,
            num_refinement_blocks=2,
            num_latent_refinements=2
        ).to(device)

        # Observation Model (OM): (Obs, EvolutionLatents) -> Next Obs
        # Predicts next pixels given current pixels and evolution latents
        self.observation_model = ObservationModel(LATENT_DIM).to(device)
        
        self.reward_model = RewardModel(LATENT_DIM).to(device)
        self.actor = Actor(LATENT_DIM, NUM_ACTIONS).to(device)
        self.critic = Critic(LATENT_DIM).to(device)
        
        self.wm_optim = AdamW(
            list(self.evolution_model.parameters()) + 
            list(self.observation_model.parameters()) + 
            list(self.reward_model.parameters()), 
            lr=1e-4
        )
        self.ac_optim = AdamW(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=1e-4
        )
        
        # Training state (Track Evolution Model State)
        self.em_outputs, self.em_latents = self.evolution_model.get_initial()
        self.em_outputs = self.em_outputs.to(device)
        self.em_latents = self.em_latents.to(device)

    def reset_internal_state(self):
        self.em_outputs, self.em_latents = self.evolution_model.get_initial()
        self.em_outputs = self.em_outputs.to(self.device)
        self.em_latents = self.em_latents.to(self.device)

    def get_action(self, obs_seq, deterministic=False):
        """
        Get action from Actor based on current observation and internal state.
        Uses Evolution Model to estimate state.
        """
        with torch.no_grad():
            # Create a "proposal" input with the current pixels and a dummy action
            dummy_action = torch.tensor([[256]], device=self.device) # Action 0 token (256)
            
            # input: (Obs, Action)
            input_full = torch.cat([obs_seq, dummy_action], dim=1)
            
            # Run one pass of Evolution Model
            # This gives us the "Next State" if we took dummy action (or just current state representation)
            # effectively processing the observation into the latent space
            _, prop_latents, _, _ = self.evolution_model(
                input_full, 
                self.em_outputs, 
                self.em_latents
            )
            
            logits = self.actor(prop_latents)
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                
        return action.item()

    def update_internal_state(self, obs_seq, action):
        """
        Update the persistent Evolution Model state with real (Obs, Action).
        """
        action_token = torch.tensor([[256 + action]], device=self.device)
        input_full = torch.cat([obs_seq, action_token], dim=1)
        
        with torch.no_grad():
            self.em_outputs, self.em_latents, _, _ = self.evolution_model(
                input_full,
                self.em_outputs,
                self.em_latents
            )

    def train_world_model(self, batch):
        """
        Train Evolution Model (Dynamics) + Observation Model (Reconstruction).
        """
        obs_b, act_b, rew_b, next_obs_b, done_b = batch
        
        # 1. Prepare Inputs for Evolution Model: [Obs, Act]
        act_tokens = act_b + 256
        em_input = torch.cat([obs_b, act_tokens], dim=1)
        
        # Get fresh initial states for EM
        init_outputs, init_latents = self.evolution_model.get_initial()
        
        # Expand to correct sequence length (SEQ_LEN = 1025)
        curr_outputs = init_outputs.expand(obs_b.size(0), SEQ_LEN, LATENT_DIM)
        curr_latents = init_latents.expand(obs_b.size(0), SEQ_LEN, LATENT_DIM)

        # EM Forward
        result = self.evolution_model(
            em_input,
            curr_outputs,
            curr_latents,
            labels=None
        )
        # When labels=None, returns (outputs, latents, pred, halt_prob)
        em_outputs_res, em_final_z, em_pred, _ = result
        
        # em_final_z is (B, 1025, Dim). 
        # Map EM latents to OM latents (B, 1024, Dim).
        om_context_latents = em_final_z[:, :1024, :]
        
        # Prepare OM inputs: Obs
        # OM Forward (Predict Next Obs)
        pred_logits_om, loss_om = self.observation_model(
            obs_b, # Input: Current Obs
            om_context_latents, # Latents from EM
            labels=next_obs_b # Target: Next Obs
        )
        
        # Reward Prediction
        # User: "actor/critic/value will use the predicted evolution of the world"
        pred_reward = self.reward_model(em_final_z)
        loss_reward = F.mse_loss(pred_reward, rew_b)
        
        total_loss = loss_om + loss_reward
        
        self.wm_optim.zero_grad()
        total_loss.backward()
        self.wm_optim.step()
        
        return {
            "loss_wm": loss_om.item(),
            "loss_reward": loss_reward.item(),
            "pred_obs": pred_logits_om.detach()
        }

    def imagine(self, start_obs, horizon=15):
        """
        Imagine leveraging Evolution Model.
        """
        b_size = start_obs.size(0)
        
        # Initialize EM state (SEQ_LEN = 1025)
        em_outputs = self.evolution_model.output_init_embed.expand(b_size, SEQ_LEN, LATENT_DIM)
        em_latents = self.evolution_model.latent_init_embed.expand(b_size, SEQ_LEN, LATENT_DIM)
        
        curr_obs = start_obs
        
        traj_states = []
        traj_actions = []
        traj_rewards = []
        traj_values = []
        traj_log_probs = []
        
        for t in range(horizon):
            # 1. Select Action (Policy uses EM latents)
            # Peek with dummy action
            dummy_act = torch.full((b_size, 1), 256, device=self.device)
            in_peek = torch.cat([curr_obs, dummy_act], dim=1)
            
            with torch.no_grad():
                _, lat_peek, _, _ = self.evolution_model(in_peek, em_outputs, em_latents)
                
            # Actor & Critic use lat_peek
            logits = self.actor(lat_peek)
            value = self.critic(lat_peek)
            
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # 2. Step Environment (Imagined)
            # Step EM with Real Action
            act_tokens = action.unsqueeze(1) + 256
            input_step = torch.cat([curr_obs, act_tokens], dim=1)
            
            with torch.no_grad():
                # Evolve EM
                em_outputs, em_latents, _, _ = self.evolution_model(
                    input_step,
                    em_outputs,
                    em_latents
                )
                
                # Predict Reward (from EM)
                reward = self.reward_model(em_latents)
                
                # Predict Next Obs (using OM)
                # OM Inputs: Current Obs
                # OM Latents: New EM Latents (chopped)
                om_context = em_latents[:, :1024, :]
                
                pred_logits_om, _ = self.observation_model(
                    curr_obs,
                    om_context
                )
                
                next_obs_pred = pred_logits_om.argmax(dim=-1)
                
            traj_states.append(lat_peek)
            traj_actions.append(action)
            traj_rewards.append(reward)
            traj_values.append(value.detach())
            traj_log_probs.append(log_prob.detach())
            
            curr_obs = next_obs_pred
            
        return (traj_states, traj_actions, traj_rewards, traj_values, traj_log_probs)

    def update_ac(self, rollouts):
        """
        PPO Update on imagined rollouts
        """
        states, actions, rewards, values, log_probs = rollouts
        
        # Stack lists
        # T steps, B batch
        T = len(rewards)
        B = rewards[0].size(0)
        
        # Flatten time and batch for processing
        # rewards: T x (B, 1)
        rewards = torch.stack(rewards).squeeze(-1) # (T, B)
        values = torch.stack(values).squeeze(-1) # (T, B)
        log_probs = torch.stack(log_probs) # (T, B)
        
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
        b_states = torch.stack(states).view(T*B, SEQ_LEN, LATENT_DIM)
        b_actions = torch.stack(actions).view(T*B)
        b_log_probs = log_probs.view(T*B)
        b_returns = returns.view(T*B)
        b_advantages = advantages.view(T*B)
        
        # PPO Epochs
        for _ in range(3):
            # Recalculate
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
    # Force FIRE to start the game
    obs, _, _, _, _ = env.step(1)
    curr_seq = preprocess(obs) # (1, 1024)
    
    # Collect some data with random policy
    for i in range(100):
        action = env.action_space.sample()
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        next_seq = preprocess(next_obs)
        replay_buffer.append((curr_seq, action, reward, next_seq, done))
        
        curr_seq = next_seq
        if done:
            obs, _ = env.reset()
            # Force FIRE to start the game
            obs, _, _, _, _ = env.step(1)
            curr_seq = preprocess(obs)
            
    print("Starting Training...")
    
    bs = 8
    
    for step in range(10000):
        # 1. Interact
        # Use Agent Epsilon-Greedy or Sampling
        if random.random() < 0.1:
            action = env.action_space.sample()
        else:
            action = agent.get_action(curr_seq.to(DEVICE))
            
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        # Update Internal State with Real Action
        agent.update_internal_state(curr_seq.to(DEVICE), action)
        
        next_seq = preprocess(next_obs)
        replay_buffer.append((curr_seq, action, reward, next_seq, done))
        curr_seq = next_seq
        
        if done:
            obs, _ = env.reset()
            # Force FIRE to start the game
            obs, _, _, _, _ = env.step(1)
            curr_seq = preprocess(obs)
            agent.reset_internal_state() # Important for RNN
            
        # 2. Train World Model
        if len(replay_buffer) > bs:
            batch = random.sample(replay_buffer, bs)
            
            # Collate
            b_obs = torch.cat([b[0] for b in batch]).to(DEVICE)
            b_act = torch.tensor([b[1] for b in batch], device=DEVICE).unsqueeze(1)
            b_rew = torch.tensor([b[2] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            b_next = torch.cat([b[3] for b in batch]).to(DEVICE)
            b_done = torch.tensor([b[4] for b in batch], dtype=torch.float, device=DEVICE).unsqueeze(1)
            
            metrics = agent.train_world_model((b_obs, b_act, b_rew, b_next, b_done))
            
            # Reconstruction Visualization
            if step % 50 == 0:
                # Take first sample from batch
                pred_pixels = metrics["pred_obs"][0].argmax(dim=-1).cpu().numpy().reshape(32, 32)
                target_pixels = b_next[0].cpu().numpy().reshape(32, 32)
                
                # Stack comparison
                vis_img = np.hstack([target_pixels, pred_pixels])
                metrics["reconstruction"] = wandb.Image(vis_img, caption="Actual vs Predicted")
            
            # Remove pred_obs from dict before logging to wandb
            del metrics["pred_obs"]

            # 3. Imagine & Train Actor-Critic
            # Imagine starting from real states in the batch
            rollouts = agent.imagine(b_obs, horizon=10)
            ac_metrics = agent.update_ac(rollouts)
            
            metrics.update(ac_metrics)
            wandb.log(metrics)
            
        if step % 100 == 0:
            print(f"Step {step}")
            
    torch.save(agent.evolution_model.state_dict(), "model/whole_em.pth")
    torch.save(agent.observation_model.state_dict(), "model/whole_om.pth")
    torch.save(agent.actor.state_dict(), "model/whole_actor.pth")
    env.close()

def test():
    print("Testing MBRL Agent...")
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    agent = MBRLAgent()
   
    # Load if exists (skip for now, just test shapes/forward)
    print("Running 1 episode...")
    obs, _ = env.reset()
    # Force FIRE to start the game
    obs, _, _, _, _ = env.step(1)
    curr_seq = preprocess(obs).to(DEVICE)
    done = False
    
    total_reward = 0
    t = 0
    while not done and t < 200:
        action = agent.get_action(curr_seq, deterministic=True)
        next_obs, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        
        agent.update_internal_state(curr_seq, action)
        curr_seq = preprocess(next_obs).to(DEVICE)
        
        total_reward += reward
        t += 1
        
    print(f"Test finished. Steps: {t}, Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    # To train: 
    train()
    # To test:
    # test()

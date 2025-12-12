import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import copy
from PIL import Image
from collections import deque

#
#
# This file contains the class and train function to train a teacher model for the more complex trm world model
#
#

# -------------------------------------------------------------
# HYPERPARAMETERS
# -------------------------------------------------------------
learning_rate = 0.0003
gamma         = 0.99
lmbda         = 0.95
eps_clip      = 0.2
K_epochs      = 4
T_horizon     = 2048
batch_size    = 64          # New: Update in mini-batches
entropy_coef  = 0.01        # New: Encourages exploration

device = torch.device("cpu")

# -------------------------------------------------------------
# WRAPPERS for Visual Observations
# -------------------------------------------------------------
class VisualWrapper(gym.Wrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(1, shape[0], shape[1]), 
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Return a black frame on reset to avoid potential render-on-reset crashes
        process_obs = np.zeros((1, self.shape[0], self.shape[1]), dtype=np.float32) 
        return process_obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
        
    def _process_obs(self, obs_vector):
        try:
            # print("Calling render...")
            frame = self.env.render() 
            # print(f"Render returned {type(frame)}")
            if frame is None:
                print("Error: env.render() returned None")
                return np.zeros((1, self.shape[0], self.shape[1]), dtype=np.float32)

            # print("Resizing...")
            img = Image.fromarray(frame)
            img = img.resize(self.shape, Image.BILINEAR)
            img = img.convert("L") # Grayscale
            
            obs = np.array(img, dtype=np.float32) / 255.0
            obs = np.expand_dims(obs, axis=0) # (1, 84, 84)
            return obs
        except Exception as e:
            print(f"Error in _process_obs: {e}")
            raise e

class StackFrames(gym.Wrapper):
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(k * shp[0], shp[1], shp[2]), # (k*C, H, W) -> (4, 84, 84)
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

# -------------------------------------------------------------
# PPO MODEL
# -------------------------------------------------------------

class CNNExtractor(nn.Module):
    def __init__(self, input_shape):
        super(CNNExtractor, self).__init__()
        # input_shape is (C, H, W) e.g. (4, 84, 84)
        c, h, w = input_shape
        
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute output size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.net(dummy)
            self.out_dim = out.shape[1]

    def forward(self, x):
        return self.net(x)

class PPO(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super(PPO, self).__init__()
        self.data = []
        
        # Feature Extractors
        # We can share parameters or have separate ones. 
        # For simplicity and classic PPO style, let's keep them separate but same architecture.
        # Although sharing is very common for CNNs. Let's do separate for stability first.
        self.actor_cnn = CNNExtractor(obs_shape)
        self.critic_cnn = CNNExtractor(obs_shape)
        
        # 1. ACTOR
        self.actor_head = nn.Sequential(
            nn.Linear(self.actor_cnn.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
            nn.Softmax(dim=-1)
        )
        
        # 2. CRITIC
        self.critic_head = nn.Sequential(
            nn.Linear(self.critic_cnn.out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor(self, x):
        features = self.actor_cnn(x)
        return self.actor_head(features)
        
    def critic(self, x):
        features = self.critic_cnn(x)
        return self.critic_head(features)

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        # Convert to tensors
        # Observations are numpy arrays already
        s = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        a = torch.tensor(np.array(a_lst)).to(device)
        r = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        s_prime = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        done_mask = torch.tensor(np.array(done_lst), dtype=torch.float).to(device)
        prob_a = torch.tensor(np.array(prob_a_lst), dtype=torch.float).to(device)
        
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a


    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        # -----------------------------------------------------------
        # 1. Calculate Generalized Advantage Estimation (GAE) ONCE
        # -----------------------------------------------------------
        with torch.no_grad():
            td_target = r + gamma * self.critic(s_prime) * done_mask
            delta = td_target - self.critic(s)
            delta = delta.cpu().numpy() # Process on CPU for the loop

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)
            # Normalize advantages (Critical for stability)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            # The target value for the Critic is the actual return
            target_value = advantage + self.critic(s) 

        # -----------------------------------------------------------
        # 2. Update with Mini-batches
        # -----------------------------------------------------------
        dataset_length = s.shape[0]
        
        for i in range(K_epochs):
            # Shuffle indices
            indices = torch.randperm(dataset_length)
            
            for start_idx in range(0, dataset_length, batch_size):
                idx = indices[start_idx : start_idx + batch_size]
                if len(idx) < batch_size: continue # Skip incomplete batches

                # Get mini-batch data
                s_batch = s[idx]
                a_batch = a[idx]
                old_prob_batch = prob_a[idx]
                adv_batch = advantage[idx]
                v_target_batch = target_value[idx]

                # Current Policy
                probs = self.actor(s_batch)
                dist = Categorical(probs)
                
                # Entropy (Exploration Bonus)
                entropy = dist.entropy().mean()
                
                # New log probabilities
                new_prob_a = probs.gather(1, a_batch)
                
                # Ratio
                ratio = torch.exp(torch.log(new_prob_a) - torch.log(old_prob_batch))

                # Surrogate Loss
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * adv_batch

                # Value Function Loss
                v_loss = F.smooth_l1_loss(self.critic(s_batch), v_target_batch)

                # Total Loss (Negative because we do gradient descent)
                # We want to MAXIMIZE entropy, so we MINIMIZE -entropy
                loss = -torch.min(surr1, surr2) + 0.5 * v_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.mean().backward()
                nn.utils.clip_grad_norm_(self.parameters(), 0.5) # Gradient Clipping
                self.optimizer.step()

def train_teacher_model(env_string, continue_training=True):
    # Initialize with RGB Render Mode
    print("Creating environment...")
    env = gym.make(env_string, render_mode="rgb_array")
    print("Environment created.")
    env = VisualWrapper(env)
    env = StackFrames(env)
    
    # Input shape is (4, 84, 84)
    model = PPO(env.observation_space.shape, env.action_space.n).to(device)
    
    model_path = "ppo_best_teacher.pth"
    
    if continue_training and os.path.exists(model_path):
        print(f"Found '{model_path}'. Loading weights to continue training...")
        # map_location ensures it loads correctly even if you switch between CPU/GPU
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No saved model found (or continue_training=False). Starting from scratch.")


    score = 0.0
    print_interval = 1
    
    # Track best score to save the absolute best model
    best_avg_score = -999 

    for n_epi in range(10000): # Increased max episodes just in case
        print(f"Starting episode {n_epi}")
        try:
            s, _ = env.reset()
        except Exception as e:
            print(f"CRASH in env.reset(): {e}")
            break
        done = False
        
        while not done:
            # Get action (no gradients needed during sampling)
            with torch.no_grad():
                # Add batch dimension: (1, 4, 84, 84)
                s_tensor = torch.from_numpy(s).float().unsqueeze(0).to(device)
                prob = model.actor(s_tensor)
                m = Categorical(prob)
                a = m.sample().item()

            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            model.put_data((s, a, r/100.0, s_prime, prob[0][a].item(), done))
            s = s_prime
            score += r

            if len(model.data) == T_horizon:
                model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            print(f"Episode: {n_epi}, Avg Score: {avg_score:.1f}")
            
            # Save if it's the best score so far
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                # Optional: Save checkpoint here

            if avg_score > 100: # 200 is good, 220 is very stable landing
                print(f"Solved at episode {n_epi}! Saving weights.")
                break
                
            score = 0.0

    env.close()
    
    # Save path logic
    save_path = "ppo_best_teacher.pth"
    torch.save(model.state_dict(), save_path)
    print(f"PPO Model Saved to {save_path}")
    return model

# -------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------

# -------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------


def train_model():
    # -------------------------------------------------------------
    # 1. SETUP & HYPERPARAMETERS
    # -------------------------------------------------------------
    env_string = "LunarLander-v3"
    device = torch.device("cpu")
    env = gym.make(env_string, render_mode="rgb_array")
    # Wrap locally just to get sizes if needed
    env = VisualWrapper(env)
    env = StackFrames(env)
    
    obs_dims = env.observation_space.shape # (4, 84, 84)
    act_dims = env.action_space.n
    
    # Define fixed goal state
    # x, y, vx, vy, angle, angular_vel, leg, leg
    goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    g_global = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)
    
   # 
    mse_criterion = nn.MSELoss()
    tau = 0.005 
    gamma = 0.99 
    epsilon = 0.2 
    num_episodes = 500
    buffer = []
    MAX_BUFFER = 10000
    
    #
    # TEACHER MODEL TRAINING
    #
    teacher_model = train_teacher_model(env_string)

    env.close()
    print("Training finished.")


if __name__ == "__main__":
    train_model()

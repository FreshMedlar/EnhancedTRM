import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import random

# Import models from your other files
from reward_model import RewardModel
from world_model import WorldModel
# We are not using ActionModel for decision making in this MPC (Planning) version,
# but we can keep it if you want to clone behavior later. For now, the Planner is the "Actor".

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENV_NAME = "LunarLander-v3"
BATCH_SIZE = 64
LR = 1e-3
MAX_BUFFER = 10000
PLAN_HORIZON = 10     # How far into the future to dream
PLAN_CANDIDATES = 50  # How many random sequences to test

# --- UTILS ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def to_tensor(x, dtype=torch.float32):
    return torch.tensor(x, dtype=dtype, device=DEVICE)

def one_hot_encode(action_indices, num_classes):
    """
    Converts integer action indices [Batch] into One-Hot vectors [Batch, Num_Classes].
    """
    if not isinstance(action_indices, torch.Tensor):
        action_indices = torch.tensor(action_indices, device=DEVICE).long()
    
    # Ensure it's long type for scatter
    action_indices = action_indices.long()
    
    # If input is scalar (0-dim), unsqueeze it
    if action_indices.dim() == 0:
        action_indices = action_indices.unsqueeze(0)
        
    batch_size = action_indices.shape[0]
    one_hot = torch.zeros(batch_size, num_classes, device=DEVICE)
    one_hot.scatter_(1, action_indices.unsqueeze(1), 1)
    return one_hot

# --- PLANNING LOGIC (MPC) ---
def mpc_planner(curr_obs, world_model, reward_model, act_dims):
    """
    Sample random actions, simulate them in the World Model, 
    pick the first action of the sequence that yields the highest predicted reward.
    """
    world_model.eval()
    reward_model.eval()

    # 1. Expand current observation to match candidates batch size
    # Shape: [candidates, obs_dim]
    curr_obs_t = to_tensor(curr_obs).unsqueeze(0).repeat(PLAN_CANDIDATES, 1)

    # 2. Generate random action sequences
    # Shape: [candidates, horizon]
    # We generate integers here, but will convert to one-hot for the world model
    action_seqs = torch.randint(0, act_dims, (PLAN_CANDIDATES, PLAN_HORIZON), device=DEVICE)

    cumulative_rewards = torch.zeros(PLAN_CANDIDATES, device=DEVICE)
    
    states = curr_obs_t
    
    # 3. Dream (Simulate Future)
    with torch.no_grad():
        for t in range(PLAN_HORIZON):
            # Get actions for this timestep across all candidates
            actions_t_idx = action_seqs[:, t] 
            
            # Convert to One-Hot for the World Model MLP
            actions_t_onehot = one_hot_encode(actions_t_idx, act_dims)
            
            # Predict Next State
            next_states = world_model(states, actions_t_onehot)
            
            # Predict Reward for this state
            # Note: In LunarLander, reward is given for the transition, but 
            # here we approximate it based on the resulting state.
            rewards = reward_model(next_states).squeeze()
            
            # Discount factor usually applies, but simple sum works for short horizon
            cumulative_rewards += rewards
            
            states = next_states

    # 4. Pick Best
    best_candidate_idx = torch.argmax(cumulative_rewards)
    best_first_action = action_seqs[best_candidate_idx, 0].item()

    world_model.train()
    reward_model.train()
    
    return int(best_first_action)


# --- MAIN TRAINING LOOP ---
def main():
    env = gym.make(ENV_NAME)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    print(f"Obs Dim: {obs_dim}, Act Dim: {act_dim}, Device: {DEVICE}")

    # Initialize Models
    # Note: World Model inputs = Obs + Action(One-Hot)
    world_model = WorldModel(obs_dim, act_dim, 128, 128, obs_dim).to(DEVICE)
    reward_model = RewardModel(obs_dim, 128, 128, 1).to(DEVICE)
    
    # Optimizers
    w_optimizer = optim.Adam(world_model.parameters(), lr=LR)
    r_optimizer = optim.Adam(reward_model.parameters(), lr=LR)
    
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer(MAX_BUFFER)

    num_episodes = 500
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        step_count = 0
        
        while not done:
            # --- 1. SELECT ACTION ---
            # Initially, the world model is garbage (random weights).
            # We should act randomly for a bit to fill the buffer, 
            # OR act based on the planner (which will be random initially).
            # Let's add epsilon-greedy exploration.
            
            epsilon = max(0.05, 1.0 - episode / 50) # Decay exploration
            
            if len(replay_buffer) < 200 or random.random() < epsilon:
                action = env.action_space.sample()
            else:
                # Use the Planner (The "Brain")
                action = mpc_planner(obs, world_model, reward_model, act_dim)

            # --- 2. EXECUTE ---
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # --- 3. STORE ---
            replay_buffer.push(obs, action, reward, next_obs, done)
            
            # --- 4. TRAIN MODELS (World & Reward) ---
            # We train every step, but using a batch of past data
            if len(replay_buffer) > BATCH_SIZE:
                b_obs, b_act, b_rew, b_next_obs, b_done = replay_buffer.sample(BATCH_SIZE)
                
                b_obs_t = to_tensor(b_obs)
                b_next_obs_t = to_tensor(b_next_obs)
                b_rew_t = to_tensor(b_rew).unsqueeze(1) # [Batch, 1]
                
                # Convert integer actions to One-Hot for the network
                b_act_onehot = one_hot_encode(b_act, act_dim)

                # A. Train World Model (Predict Next State)
                w_optimizer.zero_grad()
                pred_next_obs = world_model(b_obs_t, b_act_onehot)
                w_loss = criterion(pred_next_obs, b_next_obs_t)
                w_loss.backward()
                w_optimizer.step()
                
                # B. Train Reward Model (Predict Reward from State)
                # Note: Usually reward depends on (s, a, s'), but here input is just s'
                r_optimizer.zero_grad()
                pred_rewards = reward_model(b_next_obs_t) # Using ground truth next state for stability
                r_loss = criterion(pred_rewards, b_rew_t)
                r_loss.backward()
                r_optimizer.step()
            
            obs = next_obs
            total_reward += reward
            step_count += 1

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1:3d} | Reward: {total_reward:6.2f} | Steps: {step_count} | Epsilon: {epsilon:.2f}")

    env.close()

if __name__ == "__main__":
    main()

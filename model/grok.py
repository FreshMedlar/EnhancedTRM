import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import copy


#
# TinyRecursiveModel
#

class TinyRecursiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_loops=16):
        """
        Args:
            input_dim: Dimension of the problem input (x)
            hidden_dim: Dimension of the latent scratchpad (z)
            output_dim: Dimension of the answer (y)
            num_loops: How many times to recurse (think) before outputting
        """
        super().__init__()
        self.num_loops = num_loops
        self.hidden_dim = hidden_dim
        
        # --- Embeddings ---
        # Projects input x and current answer y into the latent space size
        self.x_proj = nn.Linear(input_dim, hidden_dim)
        self.y_proj = nn.Linear(output_dim, hidden_dim)
        
        # --- The "Tiny" Network ---
        # A simple MLP that will be reused recursively.
        # It takes concatenated (x, y, z) and outputs updates for z.
        self.think_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # --- The Output Head ---
        # Converts the final latent state z back into an answer y
        self.act_net = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, initial_y=None):
        batch_size = x.size(0)
        
        # 1. Initialize States
        # x_emb: Fixed representation of the problem
        x_emb = self.x_proj(x) 
        
        # z: The "Scratchpad" (Latent thought vector), initialized to zeros
        z = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # y: The "Current Answer", initialized to zeros or a start token
        if initial_y is None:
            # Assuming continuous output for simplicity; for tokens use embeddings
            y = torch.zeros(batch_size, self.act_net.out_features, device=x.device)
        else:
            y = initial_y

        # 2. Recursive Reasoning Loop (The "Think" Phase)
        # We reuse the SAME small network multiple times.
        for step in range(self.num_loops):
            # Project current y into hidden space
            y_emb = self.y_proj(y)
            
            # Combine distinct streams: 
            # Problem (x) + Current Guess (y) + Current Thoughts (z)
            combined_input = torch.cat([x_emb, y_emb, z], dim=-1)
            
            # Update thoughts (z) based on the combined state
            # This is a residual update (z_new = z_old + update)
            z_update = self.think_net(combined_input)
            z = z + z_update  
            
            # 3. Refinement (The "Act" Phase)
            # Update the answer y based on the new thoughts
            # In the official paper, this might happen less frequently or at every step
            y = self.act_net(z)

        return y

# --- Example Usage ---
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 5
    INPUT_DIM = 64   # Size of the puzzle/problem embedding
    HIDDEN_DIM = 128 # Size of the "brain" (scratchpad)
    OUTPUT_DIM = 10  # Size of the solution
    
    # Instantiate the model
    model = TinyRecursiveModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_loops=12)
    
    # Dummy data (e.g., a batch of 5 logic puzzles)
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    
    # Forward pass (Thinking process...)
    final_answer = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {final_answer.shape}")
    print("Model successfully reasoned recursively!")

# -------------------------------------------------------------
# 1. LEARNING MODEL
# -------------------------------------------------------------

class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden1, hidden2, out_dim):
        super().__init__()
        # Input is Observation + Action (One-hot or Gumbel-Softmax)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, obs, action):
        # Concatenate observation and action
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)

class ActionModel(nn.Module):
    def __init__(self, obs_dim, hidden1, hidden2, act_dim):
        super().__init__()
        # Input is Observation + Goal
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden1), # Assuming goal is same size as obs
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, act_dim)
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)

class DistanceCritic(nn.Module):
    def __init__(self, obs_dim, hidden1, hidden2, out_dim=1):
        super().__init__()
        # Input is Observation + Goal, Output is estimated distance
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)

# -------------------------------------------------------------
# 1.2 TEACHING MODEL
# -------------------------------------------------------------

# Hyperparameters
t_learning_rate = 3e-4
t_gamma = 0.99            # Discount factor
t_lmbda = 0.95            # GAE smoothing
t_eps_clip = 0.2          # PPO Clipping
t_K_epochs = 4            # Updates per iteration
t_T_horizon = 2000        # Timesteps per update

class PPO(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(PPO, self).__init__()
        self.data = []
        
        # Actor: Decides action probabilities
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: Estimates value of state
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=t_learning_rate)

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
            
        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), \
                                              torch.tensor(a_lst), \
                                              torch.tensor(r_lst), \
                                              torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), \
                                              torch.tensor(prob_a_lst)
        self.data = [] # Clear buffer
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(t_K_epochs):
            # Calculate Advantage (TD Error style)
            # Note: A full GAE implementation is better but slightly more complex.
            # This simplified version works well for simple envs.
            td_target = r + t_gamma * self.critic(s_prime) * done_mask
            delta = td_target - self.critic(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = t_gamma * t_lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            # PPO Update
            pi = self.actor(s)
            pi_a = pi.gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-t_eps_clip, 1+t_eps_clip) * advantage
            
            # Loss = -PolicyLoss + ValueLoss
            loss = -torch.min(surr1, surr2) + 0.5 * nn.functional.smooth_l1_loss(self.critic(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

def train_teacher_model(env_string):
    env = gym.make(env_string)
    model = PPO(env.observation_space.shape[0], env.action_space.n)
    
    model_path = "ppo_best_teacher.pth"

    # ---------------------------------------------------------
    # CHECK AND LOAD LOGIC
    # ---------------------------------------------------------
    if os.path.exists(model_path):
        print(f"Found '{model_path}'. Loading model...")
        model.load_state_dict(torch.load(model_path))
        
        # --- Evaluation Loop ---
        print("Evaluating loaded model performance...")
        model.eval()  # Set to eval mode (stops dropout/batchnorm updates)
        
        eval_score = 0.0
        s, _ = env.reset()
        done = False
        
        with torch.no_grad():  # No gradients needed for evaluation
            while not done:
                s_tensor = torch.from_numpy(s).float()
                prob = model.actor(s_tensor)
                
                # For strict evaluation, you often take the argmax (most probable action)
                # But to mimic your PPO behavior exactly, we sample:
                m = Categorical(prob)
                a = m.sample().item() 
                
                s_prime, r, terminated, truncated, _ = env.step(a)
                done = terminated or truncated
                eval_score += r
                s = s_prime
        
        print(f"Evaluation complete. Score: {eval_score:.2f}")
        
        # Switch back to training mode if you plan to continue training this model
        model.train() 
    else:
        print(f"No file named '{model_path}' found. Training from scratch...")

    score = 0.0
    print_interval = 20

    for n_epi in range(3000):
        s, _ = env.reset()
        done = False
        while not done:
            prob = model.actor(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            
            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            model.put_data((s, a, r/100.0, s_prime, prob[a].item(), done))
            s = s_prime
            score += r
            
            if len(model.data) == t_T_horizon:
                model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"Episode: {n_epi}, Avg Score: {score/print_interval:.1f}")
            score = 0.0

    env.close()
    torch.save(model.state_dict(), "ppo_teacher_model.pth")
    print("PPO Model Saved.")
    return model

# -------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------

def is_goal(s, g):
    s = s if isinstance(s, np.ndarray) else s.detach().cpu().numpy().squeeze()
    g = g if isinstance(g, np.ndarray) else g.detach().cpu().numpy().squeeze()
    
    # Tolerances
    tol_pos = 0.05
    tol_vel = 0.1
    tol_ang = 0.05
    
    # Safe guard for shape mismatches
    if s.shape != g.shape: return False

    return (abs(s[0] - g[0]) < tol_pos and 
            abs(s[1] - g[1]) < tol_pos and 
            abs(s[2] - g[2]) < tol_vel and 
            abs(s[3] - g[3]) < tol_vel and 
            abs(s[4] - g[4]) < tol_ang and
            s[6] == g[6] and s[7] == g[7])

def plan_action(obs, world_model, distance_critic, goal):
    """
    The action should be decided based on the observation and the goal
    """
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    min_d = float('inf')
    best_action = 0
    
    # Need to detach to prevent gradient tracking during inference planning
    with torch.no_grad():
        for a in range(act_dims):
            a_t = F.one_hot(torch.tensor([a]), num_classes=act_dims).float().to(device)
            next_s = world_model(obs_t, a_t)
            d = distance_critic(next_s, goal).item()
            
            if d < min_d:
                min_d = d
                best_action = a
    return best_action

def find_bridge_point(s, goal, distance_critic, buffer):
    if len(buffer) < 50:
        return None
        
    with torch.no_grad():
        d_sg = distance_critic(s, goal).item()
        
        # Sample random candidates
        indices = np.random.randint(0, len(buffer), size=min(len(buffer), 20))
        
        for idx in indices:
            b_np = buffer[idx][3] # next_obs
            b = torch.tensor(b_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            d_sb = distance_critic(s, b).item()
            d_bg = distance_critic(b, goal).item()
            
            # Triangle inequality violation heuristic:
            # If going S->G is 'longer' than S->B + B->G, B helps "bridge" the gap
            # (Note: In metric space, triangle inequality always holds. In value functions, 
            # violations indicate B is a useful waypoint).
            if d_sg > d_sb + d_bg + 0.1: # Added small margin
                return b
    return None

def train_policy_in_imagination(world_model, distance_critic, action_model, a_optimizer, obs_batch, goal):
    """
    Trains the actor entirely in the latent space of the world model.
    """
    world_model.requires_grad_(False)
    distance_critic.requires_grad_(False)
    
    horizon = 5 # Short horizon for stability
    traj_distances = []
    
    state = obs_batch
    
    # Expand goal to match batch size if necessary
    if goal.shape[0] != state.shape[0]:
        goal_batch = goal.repeat(state.shape[0], 1)
    else:
        goal_batch = goal
    
    for t in range(horizon):
        logits = action_model(state, goal_batch)
        
        # Reparameterization trick: allows gradients to flow through sampling
        action_soft = F.gumbel_softmax(logits, tau=1.0, hard=False)
        
        # World model predicts next state based on soft action
        next_state = world_model(state, action_soft)
        
        # Critic estimates distance to goal
        dist = distance_critic(next_state, goal_batch)
        traj_distances.append(dist)
        
        state = next_state
        
    # Loss: minimize the average predicted distance over the horizon
    policy_loss = torch.stack(traj_distances).mean()
    
    a_optimizer.zero_grad()
    policy_loss.backward()
    a_optimizer.step()
    
    world_model.requires_grad_(True)
    distance_critic.requires_grad_(True)
    
    return policy_loss.item()

# -------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------


def train_model():
    # -------------------------------------------------------------
    # 1. SETUP & HYPERPARAMETERS
    # -------------------------------------------------------------
    env_string = "LunarLander-v3"
    device = torch.device("cpu")
    env = gym.make(env_string)
    obs_dims = env.observation_space.shape[0]
    act_dims = env.action_space.n
    
    # Define fixed goal state
    # x, y, vx, vy, angle, angular_vel, leg, leg
    goal_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
    g_global = torch.tensor(goal_state, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Create models
    world_model = TinyRecursiveModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, num_loops=12)
    world_model = WorldModel(obs_dims, act_dims, 128, 128, obs_dims).to(device)
    action_model = ActionModel(obs_dims, 128, 128, act_dims).to(device)
    distance_critic = DistanceCritic(obs_dims, 128, 128, 1).to(device)
    target_distance = copy.deepcopy(distance_critic)
    
    # Optimizers
    w_optimizer = optim.AdamW(world_model.parameters(), lr=1e-3)
    a_optimizer = optim.AdamW(action_model.parameters(), lr=1e-4)
    d_optimizer = optim.AdamW(distance_critic.parameters(), lr=1e-3)
    
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

    print("Starting training...")
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0.0
        current_goal = g_global
        
        # Logging placeholders
        log_action_loss = 0.0
        log_world_loss = 0.0
        log_dist_loss = 0.0
        
        # 1. Bridge Point Search
        if np.random.rand() < epsilon and len(buffer) > 500:
            s0_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            bridge = find_bridge_point(s0_t, g_global, distance_critic, buffer)
            if bridge is not None:
                current_goal = bridge
    
        steps = 0
        while not done:
            steps += 1
            
            # 2. Action Selection (MPC or Random)
            if len(buffer) < 200:
                action = env.action_space.sample()
            else:
                if np.random.rand() < 0.1:
                    action = env.action_space.sample()
                else:
                    action = plan_action(obs, world_model, distance_critic, current_goal)
    
            # 3. Environment Step
            next_obs, real_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            buffer.append((obs, action, real_reward, next_obs, done))
            if len(buffer) > MAX_BUFFER: buffer.pop(0)
    
            # Check goal reached
            if current_goal is not g_global and is_goal(next_obs, current_goal):
                 current_goal = g_global
    
            # 4. Online Training (Batch Update)
            if len(buffer) > 200:
                # Sample Batch
                batch_idx = np.random.choice(len(buffer), size=32)
                b_obs = torch.stack([torch.tensor(buffer[i][0], dtype=torch.float32, device=device) for i in batch_idx])
                b_act_idx = torch.stack([torch.tensor(buffer[i][1], dtype=torch.long, device=device) for i in batch_idx])
                b_next = torch.stack([torch.tensor(buffer[i][3], dtype=torch.float32, device=device) for i in batch_idx])
                b_done = torch.stack([torch.tensor(buffer[i][4], dtype=torch.float32, device=device) for i in batch_idx]).unsqueeze(1)
                
                # Prepare goal batch
                batch_goal = current_goal.repeat(32, 1)
    
                # --- Train World Model ---
                b_act_onehot = F.one_hot(b_act_idx, num_classes=act_dims).float()
                pred_next = world_model()
                pred_next = world_model(b_obs, b_act_onehot)
                world_loss = mse_criterion(pred_next, b_next)
                
                w_optimizer.zero_grad()
                world_loss.backward()
                w_optimizer.step()
                log_world_loss = world_loss.item()
    
                # --- Train Distance Critic ---
                # If goal reached, distance is 0, else 1 + gamma * remaining
                # Simplification: Standard Bellman update on distance
                with torch.no_grad():
                    next_dist = target_distance(b_next, batch_goal)
                    target_d = 1.0 + gamma * next_dist * (1 - b_done)
                
                pred_dist = distance_critic(b_obs, batch_goal)
                dist_loss = mse_criterion(pred_dist, target_d)
                
                d_optimizer.zero_grad()
                dist_loss.backward()
                d_optimizer.step()
                log_dist_loss = dist_loss.item()
    
                # Soft update target critic
                for target_param, param in zip(target_distance.parameters(), distance_critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
                # --- Train Policy (Imagination) ---
                # We use the separate function we fixed earlier
                log_action_loss = train_policy_in_imagination(world_model, distance_critic, action_model, a_optimizer, b_obs, batch_goal)
    
            obs = next_obs
            ep_reward += real_reward
    
        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1:3d} | Reward: {ep_reward:6.1f} | A_Loss: {log_action_loss:.3f} | W_Loss: {log_world_loss:.4f} | D_Loss: {log_dist_loss:.4f}")
    
    env.close()
    print("Training finished.")


if __name__ == "__main__":
    train_model()


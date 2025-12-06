
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import random
from collections import deque

# --- Hyperparameters ---
GAMMA = 0.99
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
LR_MODEL = 1e-3
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 64
BRIDGE_SEARCH_ITER = 20  # Max attempts to find a bridge point

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Networks ---

class ModelNetwork(nn.Module):
    """
    f(s, a) -> s_next
    Learns the deterministic transition dynamics.
    """
    def __init__(self, state_dim, action_dim):
        super(ModelNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)

class DistanceCritic(nn.Module):
    """
    D(s, g) -> distance (scalar)
    Estimates min steps from state s to goal g.
    """
    def __init__(self, state_dim):
        super(DistanceCritic, self).__init__()
        # Input is concatenated state and goal
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.net(x)

class Actor(nn.Module):
    """
    mu(s, g) -> action
    Outputs action to minimize distance to goal.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=1)
        return self.max_action * self.net(x)

# --- Replay Buffer ---

class ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, goal, distance):
        self.buffer.append((state, action, next_state, goal, distance))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, next_state, goal, distance = zip(*batch)
        return (
            torch.FloatTensor(np.array(state)).to(device),
            torch.FloatTensor(np.array(action)).to(device),
            torch.FloatTensor(np.array(next_state)).to(device),
            torch.FloatTensor(np.array(goal)).to(device),
            torch.FloatTensor(np.array(distance)).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# --- GDG Agent ---

class GDGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        
        # Initialize Networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = DistanceCritic(state_dim).to(device)
        self.critic_target = DistanceCritic(state_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.model = ModelNetwork(state_dim, action_dim).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=LR_MODEL)

        self.buffer = ReplayBuffer(BUFFER_SIZE, state_dim, action_dim)

    def select_action(self, state, goal, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)
        
        action = self.actor(state, goal).cpu().data.numpy().flatten()
        if noise != 0:
            action = (action + np.random.normal(0, noise, size=action.shape))
            
        return action

    def update(self):
        if len(self.buffer) < BATCH_SIZE:
            return

        state, action, next_state, goal, reward_dist = self.buffer.sample(BATCH_SIZE)

        # 1. Update Model Network f(s, a) -> s'
        # Loss L_f = MSE(predicted_next_state, actual_next_state)
        pred_next_state = self.model(state, action)
        model_loss = nn.MSELoss()(pred_next_state, next_state)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        # 2. Update Distance Critic D(s, g)
        # Target = d + D'(s', g)
        # Note: The paper uses 'd' as transition count (usually 1 per step)
        with torch.no_grad():
            target_dist = reward_dist + self.critic_target(next_state, goal)
        
        current_dist = self.critic(state, goal)
        critic_loss = nn.MSELoss()(current_dist, target_dist)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 3. Update Actor
        # Minimize D(f(s, mu(s, g)), g)
        # We use the learned model 'f' to differentiate through the state transition
        
        # Predict action using current policy
        pred_action = self.actor(state, goal)
        
        # Predict next state using learned model
        # We must use the model here because we need gradients flow from D -> s' -> a
        pred_next_state_actor = self.model(state, pred_action)
        
        # Compute distance from predicted next state to goal
        actor_loss = self.critic(pred_next_state_actor, goal).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 4. Soft Update Target Networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def search_bridge_point(self, start_state, goal_state, sample_fn):
        """
        Algorithm 2: Search for a bridge point.
        sample_fn: A function that returns random valid states from the environment/buffer.
        """
        start_tensor = torch.FloatTensor(start_state.reshape(1, -1)).to(device)
        goal_tensor = torch.FloatTensor(goal_state.reshape(1, -1)).to(device)

        # Current estimated distance
        with torch.no_grad():
            d_sg = self.critic(start_tensor, goal_tensor).item()

        for _ in range(BRIDGE_SEARCH_ITER):
            # Sample a potential bridge point
            bridge_state = sample_fn() 
            bridge_tensor = torch.FloatTensor(bridge_state.reshape(1, -1)).to(device)

            with torch.no_grad():
                d_sb = self.critic(start_tensor, bridge_tensor).item()
                d_bg = self.critic(bridge_tensor, goal_tensor).item()

            # Check if bridge provides a "shortcut" or better path
            # The paper condition: if D(s,g) > D(s,b) + D(b,g)
            # This implies the direct estimate is pessimistic or the bridge helps decomposition.
            if d_sg > d_sb + d_bg:
                return bridge_state # Found a valid bridge point
        
        return None # No better bridge found

# --- Example Usage ---

if __name__ == "__main__":
    # Mock Environment for dimension setup
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = GDGAgent(state_dim, action_dim, max_action)
    
    # Mock training loop step
    s, _ = env.reset()
    g = s # In real multi-goal RL, goal 'g' is different from 's'
    
    # To simulate the paper's loop:
    # 1. Decide if we need a bridge point
    bridge = agent.search_bridge_point(s, g, lambda: env.observation_space.sample())
    current_goal = bridge if bridge is not None else g
    
    # 2. Take action
    a = agent.select_action(s, current_goal)
    next_s, r, term, trunc, _ = env.step(a)
    
    # 3. Store (Note: distance 'd' is typically 1 for a single step)
    agent.buffer.push(s, a, next_s, current_goal, distance=1.0)
    
    # 4. Update
    agent.update()
    
    print("Agent update complete. Bridge point found:", bridge is not None)

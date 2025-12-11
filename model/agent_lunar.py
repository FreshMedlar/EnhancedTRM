import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gymnasium as gym

# --- Hyperparameters ---
LEARNING_RATE = 5e-4
MINIBATCH_SIZE = 100
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = int(1e5)
INTERPOLATION_PARAMETER = 1e-3
NUMBER_EPISODES = 2000
MAX_TIMESTEPS_PER_EPISODE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
CHECKPOINT_FILE = 'checkpoint.pth'

class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        return self.fc3(x)

class ReplayMemory(object):
    def __init__(self, capacity):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        return states, next_states, actions, rewards, dones

class Agent():
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = Network(state_size, action_size).to(self.device)
        self.target_qnetwork = Network(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(REPLAY_BUFFER_SIZE)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            if len(self.memory.memory) > MINIBATCH_SIZE:
                experiences = self.memory.sample(100)
                self.learn(experiences, DISCOUNT_FACTOR)

    def act(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, discount_factor):
        states, next_states, actions, rewards, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + discount_factor * next_q_targets * (1 - dones)
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, INTERPOLATION_PARAMETER)

    def soft_update(self, local_model, target_model, interpolation_parameter):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(interpolation_parameter * local_param.data + (1.0 - interpolation_parameter) * target_param.data)
            
    def load_weights(self, path):
        if os.path.isfile(path):
            print(f"Loading weights from {path}")
            state_dict = torch.load(path, map_location=self.device)
            self.local_qnetwork.load_state_dict(state_dict)
            self.target_qnetwork.load_state_dict(state_dict)
        else:
            print("No checkpoint found. Starting from scratch.")

    def save_weights(self, path):
        print(f"\nSaving weights to {path}")
        torch.save(self.local_qnetwork.state_dict(), path)

if __name__ == "__main__":
    # Initialize Environment
    # Note: 'LunarLander-v3' requires gymnasium>=1.0.0. Use 'v2' if using older versions.
    env_name = 'LunarLander-v3' 
    try:
        env = gym.make(env_name)
    except gym.error.NameNotFound:
        print(f"Warning: {env_name} not found. Falling back to LunarLander-v2")
        env = gym.make('LunarLander-v2')
        
    state_shape = env.observation_space.shape
    state_size = env.observation_space.shape[0]
    number_actions = env.action_space.n
    
    print('State shape: ', state_shape)
    print('State size: ', state_size)
    print('Number of actions: ', number_actions)

    # Initialize Agent
    agent = Agent(state_size, number_actions)
    
    # Load weights if they exist
    agent.load_weights(CHECKPOINT_FILE)

    # Training Loop
    epsilon = EPSILON_START
    scores_on_100_episodes = deque(maxlen=100)

    for episode in range(1, NUMBER_EPISODES + 1):
        state, _ = env.reset()
        score = 0
        
        for t in range(MAX_TIMESTEPS_PER_EPISODE):
            action = agent.act(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
                
        scores_on_100_episodes.append(score)
        epsilon = max(EPSILON_END, EPSILON_DECAY * epsilon)
        
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end="")
        
        if episode % 100 == 0:
            print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        
        # Check if environment is solved (Average score >= 200)
        if np.mean(scores_on_100_episodes) >= 200.0:
            print(f'\nEnvironment solved in {episode - 100} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
            agent.save_weights(CHECKPOINT_FILE)
            break
            
    env.close()

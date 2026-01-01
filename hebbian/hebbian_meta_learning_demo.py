import torch
import torch.nn as nn
import numpy as np
try:
    import gym
except ImportError:
    import gymnasium as gym
import time
import multiprocessing as mp
from typing import Tuple, List

# ======================================================================================
# Meta-Learning through Hebbian Plasticity in Random Networks
# --------------------------------------------------------------------------------------
# This implementation demonstrates the paradigm where we evolve the PARAMETERS of 
# Hebbian learning rules rather than the network weights themselves.
# 
# Key concepts:
# 1. Lifetime Learning: An agent starts each episode with random weights.
# 2. Local Rules: Synapses update based on pre- and post-synaptic activity.
# 3. Meta-Evolution: An Evolution Strategy (ES) evolves the Hebbian coefficients.
# 4. Result: A network that "learns" to solve a task within its own lifetime.
# ======================================================================================

class HebbianMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron where weights are updated via Hebbian rules.
    Instead of bias, we use the activations themselves to drive plasticity.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) for i in range(len(dims)-1)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns all layer activations for Hebbian updates."""
        activations = [x]
        curr = x
        for i, layer in enumerate(self.layers):
            curr = layer(curr)
            if i < len(self.layers) - 1:
                curr = torch.tanh(curr)
            activations.append(curr)
        return tuple(activations)

    @torch.no_grad()
    def apply_hebbian_update(self, activations: List[torch.Tensor], heb_coeffs: List[torch.Tensor]):
        """
        Updates weights using the ABCD Hebbian Rule:
        delta_w = lr * (A * pre * post + B * pre + C * post + D)
        
        Where 'pre' and 'post' are activations of the connected neurons.
        Each synapse has its own set of (A, B, C, D, lr) coefficients.
        """
        for i, layer in enumerate(self.layers):
            # Pre-synaptic (input to layer) and post-synaptic (output of layer)
            pre = activations[i].view(-1)
            post = activations[i+1].view(-1)
            
            # Coefficients for this layer: [out_features, in_features, 5]
            # 5 coefficients: A, B, C, D, LearningRate
            c = heb_coeffs[i] 
            
            # Outer product of activations to get pre * post for every synapse
            pre_post = torch.outer(post, pre)
            
            # Hebbian rule: Δw = η * (A*pre*post + B*pre + C*post + D)
            # c[..., 0]=A, c[..., 1]=B, c[..., 2]=C, c[..., 3]=D, c[..., 4]=η
            # Note: We use broadcasting for pre (repeated across rows) and post (repeated across columns)
            delta_w = c[..., 4] * (
                c[..., 0] * pre_post +           # A * pre * post
                c[..., 1] * pre.unsqueeze(0) +  # B * pre
                c[..., 2] * post.unsqueeze(1) + # C * post
                c[..., 3]                       # D
            )
            
            layer.weight.add_(delta_w)
            
            # Normalize weights to prevent explosion
            max_val = layer.weight.abs().max()
            if max_val > 0:
                layer.weight.div_(max_val)

def fitness_function(env_name, heb_coeffs_list, seed=None):
    """
    Evaluates a single agent in the environment.
    The agent starts with RANDOM weights and learns via Hebbian rules.
    """
    try:
        env = gym.make(env_name)
    except:
        # Fallback for newer gymnasium
        import gymnasium as gym_new
        env = gym_new.make(env_name)
        
    if seed is not None:
        try: env.seed(seed)
        except: pass
        
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        discrete = True
    else:
        action_dim = env.action_space.shape[0]
        discrete = False
        
    # Initialize network
    model = HebbianMLP(obs_dim, [64, 32], action_dim)
    
    # Randomize initial weights (The 'Random Networks' part of the paradigm)
    for layer in model.layers:
        nn.init.uniform_(layer.weight, -0.1, 0.1)
        
    total_reward = 0
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0] # Gymnasium fix
    
    done = False
    truncated = False
    
    while not (done or truncated):
        # Forward pass
        state_tensor = torch.from_numpy(obs).float()
        activations = model(state_tensor)
        output = activations[-1]
        
        # Select action
        if discrete:
            action = torch.argmax(output).item()
        else:
            action = torch.tanh(output).numpy()
            
        # Environment step
        step_res = env.step(action)
        if len(step_res) == 4:
            obs, reward, done, info = step_res
        else:
            obs, reward, done, truncated, info = step_res
            
        total_reward += reward
        
        # LIFETIME LEARNING: Update weights using the evolved Hebbian rules
        model.apply_hebbian_update(activations, heb_coeffs_list)
        
    env.close()
    return total_reward

class EvolutionStrategy:
    """
    Simple Evolution Strategy (ES) to evolve Hebbian coefficients.
    Based on OpenAI's ES: x_new = x_old + lr * sum(reward_i * noise_i)
    """
    def __init__(self, layer_shapes, pop_size=50, sigma=0.1, lr=0.2):
        self.pop_size = pop_size
        self.sigma = sigma
        self.lr = lr
        
        # Meta-Parameters: For each synapse, we evolve 5 Hebbian coefficients (A, B, C, D, η)
        self.heb_coeffs = [
            torch.randn(out_dim, in_dim, 5) * 0.1
            for out_dim, in_dim in layer_shapes
        ]
        
    def get_population(self):
        """Generates a population of jittered coefficients using antithetic sampling."""
        population = []
        for _ in range(self.pop_size // 2):
            noise = [torch.randn_like(c) for c in self.heb_coeffs]
            # Plus noise
            pop_plus = [c + self.sigma * n for c, n in zip(self.heb_coeffs, noise)]
            # Minus noise
            pop_minus = [c - self.sigma * n for c, n in zip(self.heb_coeffs, noise)]
            population.append((pop_plus, noise))
            population.append((pop_minus, [-n for n in noise]))
        return population

    def update(self, population, rewards):
        """Standardizes rewards and updates the meta-parameters."""
        rewards = np.array(rewards)
        if np.std(rewards) > 1e-6:
            rewards = (rewards - np.mean(rewards)) / np.std(rewards)
        
        for i, c in enumerate(self.heb_coeffs):
            # Accumulate gradient: sum(reward * noise)
            grad = torch.zeros_like(c)
            for j in range(self.pop_size):
                grad += rewards[j] * population[j][1][i]
            
            # Update meta-parameters
            self.heb_coeffs[i] += self.lr / (self.pop_size * self.sigma) * grad

def train():
    ENV_NAME = "CartPole-v1"
    POP_SIZE = 40
    GENERATIONS = 50
    
    print(f"Starting Hebbian Meta-Learning on {ENV_NAME}")
    print("Each agent starts with RANDOM weights and must 'learn' to balance the pole during its lifetime.")
    
    # Determine layer shapes
    # Logic to get dims without full env init if possible, but simple for demo:
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    env.close()
    
    layer_shapes = [(64, obs_dim), (32, 64), (action_dim, 32)]
    es = EvolutionStrategy(layer_shapes, pop_size=POP_SIZE, sigma=0.1, lr=0.1)
    
    best_reward = -np.inf
    
    # Multiple processes for evaluation
    pool = mp.Pool(processes=mp.cpu_count())
    
    for gen in range(GENERATIONS):
        start_time = time.time()
        
        # 1. Generate population
        pop = es.get_population()
        
        # 2. Evaluate fitness (Parallel)
        # Using a list of (coeffs) for the pool
        eval_args = [(ENV_NAME, [p[0][i] for i in range(len(layer_shapes))]) for p in pop]
        rewards = pool.starmap(fitness_function, eval_args)
        
        # 3. Update coefficients
        es.update(pop, rewards)
        
        avg_reward = np.mean(rewards)
        max_reward = np.max(rewards)
        if max_reward > best_reward: best_reward = max_reward
        
        duration = time.time() - start_time
        print(f"Gen {gen:02d} | Avg Reward: {avg_reward:6.1f} | Max: {max_reward:6.1f} | Best: {best_reward:6.1f} | Time: {duration:.2f}s")
        
        if avg_reward > 495: # CartPole-v1 success threshold
            print("Successfully learned Hebbian rules that solve the task!")
            break
            
    pool.close()
    pool.join()

if __name__ == "__main__":
    train()

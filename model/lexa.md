

Here is a clean, self-contained PyTorch implementation of the **LEXA-style curiosity module**.

Since you already have the World Model, you just need the **Ensemble** to calculate the curiosity signal (model disagreement) and the logic to switch between "Exploring" (maximizing disagreement) and "Achieving" (minimizing goal distance).

You can drop this class directly into your agent setup.

### **The Python Implementation**

This code implements the **Latent Disagreement Ensemble**. It assumes your World Model provides latent states (e.g., from a VAE or RSSM).

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # Concatenate state and action
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

```

---

### **Integration Guide (Pseudo-code)**

Here is how to hook this into your existing training loop. The key is that you have **two separate policy heads** (or a conditioned policy) sharing the same world model backbone.

#### **1. Setup**

```python
# Assuming your world model produces latent states of size 1024
curiosity_module = CuriosityEnsemble(state_dim=1024, action_dim=action_size).cuda()
ensemble_opt = torch.optim.Adam(curiosity_module.parameters(), lr=1e-4)

```

#### **2. The "Explorer" Phase (Training the Ensemble)**

When the agent is exploring, it generates intrinsic rewards for itself.

```python
# Get latent state from your World Model
z_t = world_model.encode(image_t)
a_t = explorer_policy(z_t)

# 1. Calculate Intrinsic Reward (Zero-shot exploration)
intrinsic_reward = curiosity_module.get_intrinsic_reward(z_t, a_t)

# 2. Train the Explorer Policy
# Maximize this intrinsic_reward using PPO/SAC/Dreamer logic
explorer_agent.update(z_t, a_t, intrinsic_reward)

# 3. Train the Ensemble (Reality Check)
# The ensemble must learn the ACTUAL dynamics to know what is surprising
z_next_real = world_model.encode(image_next)
curiosity_module.train_ensemble(z_t, a_t, z_next_real, ensemble_opt)

```

#### **3. The "Achiever" Phase (Goal Reaching)**

When you want the agent to achieve a goal, you swap the reward function.

```python
# User specifies a goal image (encoded to latent)
g_latent = world_model.encode(goal_image)

# 1. Calculate Goal Reward
# The Achiever tries to get z_t close to g_latent
goal_reward = curiosity_module.get_goal_reward(z_t, g_latent)

# 2. Train/Execute Achiever Policy
achiever_agent.update(z_t, a_t, goal_reward)

```

### **Why this code works well**

1. **Metric Stability:** `torch.var` is much more stable than training a separate density model (like VAEs or Flow models) just for curiosity.
2. **Modularity:** It doesn't touch your World Model weights. It sits on top, observing the latents and actions.
3. **Universal:** It works regardless of whether your World Model is a VAE, an RSSM (Dreamer), or a simple JEPA.

**Would you like a snippet for the "Hindsight Experience Replay" (HER) logic to train the Achiever faster?**
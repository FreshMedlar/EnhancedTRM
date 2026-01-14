import torch
import torch.nn as nn
import numpy as np
import time

class HebbianLinear(nn.Module):
    """
    A Hebbian layer where weights are plastic (change every step) 
    and coefficients are meta-learned (fixed during episode).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Plastic weights: reset every episode, initialized randomly
        # We use a buffer so it doesn't get updated by the optimizer (ES) directly
        self.register_buffer('weight', torch.empty(out_features, in_features))
        
        # Hebbian coefficients per synapse: [A, B, C, D, bias_offset]
        # These are what the ES will optimize.
        # Index 0: A -> term pre * post
        # Index 1: B -> term pre
        # Index 2: C -> term post
        # Index 3: D -> constant term
        # Index 4: η -> local learning rate
        self.heb_coeffs = nn.Parameter(torch.randn(out_features, in_features, 5) * 0.1)

    def forward(self, x):
        # x shape: (batch_size, in_features)
        y = torch.matmul(x, self.weight.t())
        return torch.tanh(y)

    def update_weights(self, pre, post):
        """
        Applies: dW = η * (A*pre*post + B*pre + C*post + D)
        """
        # pre: (in_features,)
        # post: (out_features,)
        
        A = self.heb_coeffs[..., 0]
        B = self.heb_coeffs[..., 1]
        C = self.heb_coeffs[..., 2]
        D = self.heb_coeffs[..., 3]
        eta = self.heb_coeffs[..., 4]
        
        # Calculate components of the update rule
        # A * o_pre * o_post
        interaction = A * torch.ger(post, pre)
        # B * o_pre
        pre_term = B * pre.unsqueeze(0)
        # C * o_post
        post_term = C * post.unsqueeze(1)
        # D (Constant term)
        bias_term = D
        
        # dW calculation
        dW = eta * (interaction + pre_term + post_term + bias_term)
        
        # Update current lifetime weights
        self.weight.data += dW
        # Clamp to prevent weights from exploding during the sequence
        self.weight.data = torch.clamp(self.weight.data, -3.0, 3.0)

    def reset_plastic_weights(self):
        # Start each episode with random initial weights (the "substrate")
        nn.init.normal_(self.weight, std=0.1)

class HebbianNet(nn.Module):
    """A network with multiple Hebbian layers."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = HebbianLinear(input_dim, hidden_dim)
        self.layer2 = HebbianLinear(hidden_dim, output_dim)

    def forward(self, x):
        # Returns activations for each layer to allow local updates
        h = self.layer1(x)
        out = self.layer2(h)
        return h, out

    def update(self, x_in, h_act, out_act):
        # Ensure dimensions are 1-D even if input is scalar
        pre = x_in.view(-1)
        h = h_act.view(-1)
        post = out_act.view(-1)
        
        self.layer1.update_weights(pre, h)
        self.layer2.update_weights(h, post)

    def reset_episode(self):
        self.layer1.reset_plastic_weights()
        self.layer2.reset_plastic_weights()

def get_sequence_batch(seq_len=20):
    """
    Generates a sequence where x_{t+1} = x_t + slope.
    The agent must adapt to 'slope' within the episode to predict correctly.
    """
    slope = (torch.rand(1).item() - 0.5) * 0.5
    start = torch.rand(1).item()
    
    inputs = []
    targets = []
    curr = start
    for _ in range(seq_len):
        inputs.append(curr)
        curr += slope
        targets.append(curr)
        
    return torch.tensor(inputs).view(-1, 1), torch.tensor(targets).view(-1, 1)

def evaluate_lifetime(model, num_episodes=5, seq_len=20):
    """Evaluates how well the rules learned by ES allow the model to adapt."""
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            model.reset_episode()
            inputs, targets = get_sequence_batch(seq_len)
            ep_loss = 0
            for i in range(len(inputs)):
                x = inputs[i:i+1] # Input at step t
                target = targets[i:i+1] # Desired at step t+1
                
                h, pred = model(x)
                
                # Record loss BEFORE update (this measures predictive ability)
                loss = torch.mean((pred - target)**2)
                ep_loss += loss.item()
                
                # Adapt weights based on the activity triggered by current sample
                model.update(x, h, pred)
                
            total_loss += ep_loss / len(inputs)
    return total_loss / num_episodes

def train_meta_es():
    """Optimizes Hebbian rules using Evolution Strategies."""
    # Define network: 1 input -> 16 hidden -> 1 output
    model = HebbianNet(input_dim=1, hidden_dim=16, output_dim=1)
    
    # ES Hyperparameters
    POPULATION_SIZE = 30
    SIGMA = 0.1
    LEARNING_RATE = 0.1
    GENERATIONS = 50
    
    # Flatten coefficients for optimization
    flat_params = nn.utils.parameters_to_vector(model.parameters())
    
    print(f"--- Meta-Training Started ---")
    print(f"Searching for {len(flat_params)} coefficients (per-synapse rule parameters)")
    
    for gen in range(GENERATIONS):
        # Create population: random perturbations around the current mean
        noise = torch.randn(POPULATION_SIZE, len(flat_params))
        fitness = []
        
        for i in range(POPULATION_SIZE):
            # Apply jittered parameters
            candidate_params = flat_params + SIGMA * noise[i]
            nn.utils.vector_to_parameters(candidate_params, model.parameters())
            
            # Evaluate the 'learning rule' over several adaptation episodes
            loss = evaluate_lifetime(model, num_episodes=2)
            fitness.append(-loss) # Fitness is negative loss
            
        fitness = torch.tensor(fitness)
        
        # Optimize using estimated gradient (ES update)
        # Normalize fitness for stability
        std = fitness.std() + 1e-8
        norm_fitness = (fitness - fitness.mean()) / std
        
        # Step towards better performance
        grad = torch.mean(norm_fitness.view(-1, 1) * noise, dim=0)
        flat_params = flat_params + LEARNING_RATE * grad
        
        if (gen + 1) % 5 == 0 or gen == 0:
            avg_loss = -fitness.mean().item()
            print(f"Gen {gen+1:3d} | Avg Adapt Loss: {avg_loss:.6f} | Best Loss: {-fitness.max().item():.6f}")

    # Load final optimized weights
    nn.utils.vector_to_parameters(flat_params, model.parameters())
    return model

if __name__ == "__main__":
    start = time.time()
    
    # Execute meta-training
    meta_agent = train_meta_es()
    
    # Verify Performance
    print("\n--- Final Verification ---")
    test_loss = evaluate_lifetime(meta_agent, num_episodes=10, seq_len=30)
    print(f"Avg Prediction Error after Adaptation: {test_loss:.6f}")
    print(f"Completed in {time.time() - start:.2f} seconds")

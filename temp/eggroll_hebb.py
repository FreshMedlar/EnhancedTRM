
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import Tuple, List

# Import Eggroll components
from eggroll_demo import (
    Rank1NoiseGenerator, 
    NoiseConfig, 
    FitnessShaper, 
    AntitheticShaper, 
    EGGROLLUpdater, 
    UpdateConfig
)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==============================================================================
# 1. BATCHED HEBBIAN MODEL
# ==============================================================================

class BatchedHebbianLinear(nn.Module):
    """
    A Hebbian layer where weights are plastic and coefficients are meta-learned.
    Supports a POPULATION dimension to evaluate multiple parameter sets in parallel.
    """
    def __init__(self, in_features, out_features, population_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.population_size = population_size
        
        # Plastic weights: [Pop, Out, In]
        # Not parameters, but buffers we reset
        self.register_buffer('weight', torch.empty(population_size, out_features, in_features))
        
        # Coefficients will be passed in during forward/update, 
        # as they vary per population member.
        # But we need to store them locally during the episode execution.
        # We'll initialize them with zeros and set them properly before checks.
        self.register_buffer('coeffs', torch.zeros(population_size, out_features, in_features, 5))

    def set_coeffs(self, coeffs):
        """
        coeffs: [Pop, Out, In, 5]
        """
        self.coeffs = coeffs

    def forward(self, x):
        # x shape: [Pop, Batch(1), In] or [Pop, In]
        # We want: [Pop, Batch, Out]
        
        # Matrix multiplication: (Pop, B, In) @ (Pop, In, Out) -> (Pop, B, Out)
        # weight is (Pop, Out, In). Transpose last two dims -> (Pop, In, Out)
        
        if x.ndim == 2: # [Pop, In]
            x = x.unsqueeze(1) # [Pop, 1, In]
            
        w_t = self.weight.transpose(1, 2) # [Pop, In, Out]
        y = torch.bmm(x, w_t) # [Pop, 1, Out]
        
        return torch.tanh(y)

    def update_weights(self, pre, post):
        """
        pre: [Pop, 1, In] (activations from previous layer)
        post: [Pop, 1, Out] (activations from this layer)
        """
        # Squeeze the batch(1) dim for calculation
        pre = pre.squeeze(1)   # [Pop, In]
        post = post.squeeze(1) # [Pop, Out]
        
        A = self.coeffs[..., 0] # [Pop, Out, In]
        B = self.coeffs[..., 1]
        C = self.coeffs[..., 2]
        D = self.coeffs[..., 3]
        eta = self.coeffs[..., 4]
        
        # Interaction: A * (post.T @ pre) ?? No, outer product per batch item
        # post: [Pop, Out], pre: [Pop, In]
        # einsum 'bi,bj->bij' -> [Pop, Out, In]
        interaction = A * torch.einsum('bo,bi->boi', post, pre)
        
        # Pre term: B * pre (expanded)
        pre_term = B * pre.unsqueeze(1) # [Pop, 1, In] -> Broadcasts to [Pop, Out, In]
        
        # Post term: C * post (expanded)
        post_term = C * post.unsqueeze(2) # [Pop, Out, 1] -> Broadcasts
        
        # Bias term
        bias_term = D
        
        # dW
        dW = eta * (interaction + pre_term + post_term + bias_term)
        
        # Update
        self.weight = self.weight + dW
        self.weight = torch.clamp(self.weight, -3.0, 3.0)

    def reset_plastic_weights(self):
        # Initialize randomly: [Pop, Out, In]
        nn.init.normal_(self.weight, std=0.1)

class BatchedHebbianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, population_size):
        super().__init__()
        self.layer1 = BatchedHebbianLinear(input_dim, hidden_dim, population_size)
        self.layer2 = BatchedHebbianLinear(hidden_dim, output_dim, population_size)
        
        # Calculate parameter counts to help splitting flattened vector later
        self.l1_params = hidden_dim * input_dim * 5
        self.l2_params = output_dim * hidden_dim * 5
        self.total_params = self.l1_params + self.l2_params
        
        self.dims = {
            'l1': (population_size, hidden_dim, input_dim, 5),
            'l2': (population_size, output_dim, hidden_dim, 5)
        }

    def load_flat_params(self, flat_params_batch):
        """
        flat_params_batch: [Pop, TotalParams]
        Distributes parameters to layers.
        """
        p1 = flat_params_batch[:, :self.l1_params]
        p2 = flat_params_batch[:, self.l1_params:]
        
        self.layer1.set_coeffs(p1.view(self.dims['l1']))
        self.layer2.set_coeffs(p2.view(self.dims['l2']))

    def forward(self, x):
        h = self.layer1(x)
        out = self.layer2(h)
        return h, out
    
    def update(self, x_in, h_act, out_act):
        self.layer1.update_weights(x_in, h_act)
        self.layer2.update_weights(h_act, out_act)
        
    def reset_episode(self):
        self.layer1.reset_plastic_weights()
        self.layer2.reset_plastic_weights()

# ==============================================================================
# 2. UTILS
# ==============================================================================

def get_sequence_batch(seq_len=20, device="cpu"):
    """
    Same sequence generator as original.
    Returns: [Seq, 1], [Seq, 1]
    """
    slope = (torch.rand(1, device=device).item() - 0.5) * 0.5
    start = torch.rand(1, device=device).item()
    
    inputs = []
    targets = []
    curr = start
    for _ in range(seq_len):
        inputs.append(curr)
        curr += slope
        targets.append(curr)
        
    return torch.tensor(inputs, device=device).view(-1, 1), torch.tensor(targets, device=device).view(-1, 1)

def evaluate_population(model: BatchedHebbianNet, num_episodes=2, seq_len=20):
    """
    Runs episodes for the entire population in parallel.
    Returns: [Pop] tensor of mean MSE losses.
    """
    pop_size = model.layer1.population_size
    total_loss = torch.zeros(pop_size, device=DEVICE)
    
    # We run the SAME sequence for all population members to reduce variance
    # Or different? Standard ES usually uses same environment seed for all mutants per step, 
    # but here the environment is stochastic (random slope).
    # To reduce variance, we should use the SAME task instance for all pop members in this generation.
    
    for _ in range(num_episodes):
        model.reset_episode()
        
        # Get one sequence
        inputs, targets = get_sequence_batch(seq_len, device=DEVICE)
        
        # Broadcast inputs to [Pop, Seq, 1] for processing loop?
        # Actually we process step by step.
        # Input at step t: [1, 1] -> broadcast to [Pop, 1]
        
        ep_loss = torch.zeros(pop_size, device=DEVICE)
        
        for t in range(len(inputs)):
            x_step = inputs[t].view(1, 1).expand(pop_size, 1) # [Pop, 1]
            y_step = targets[t].view(1, 1).expand(pop_size, 1) # [Pop, 1]
            
            # Forward [Pop, 1] -> [Pop, 1, Out]
            # Since model expects [Pop, In], we pass x_step
            h, pred = model(x_step) # pred: [Pop, 1, 1]
            
            # Loss computation
            # (Pop, 1, 1) - (Pop, 1) -> (Pop, 1, 1)
            loss = (pred.squeeze(1) - y_step)**2 # [Pop, 1]
            ep_loss += loss.squeeze(1)
            
            # Update plasticity
            # x_step is [Pop, 1], h is [Pop, 1, Hid], pred is [Pop, 1, Out]
            # we need to pass inputs in format [Pop, 1, Dim]
            # x_step expanded to [Pop, 1, 1]
            model.update(x_step.unsqueeze(1), h, pred)
            
        total_loss += ep_loss / seq_len
        
    return total_loss / num_episodes

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================

def train_eggroll_hebb():
    # Setup
    POPULATION_SIZE = 30 # Must match original for fair comparison
    if POPULATION_SIZE % 2 != 0: POPULATION_SIZE += 1 # Ensure even for antithetic
    
    SIGMA = 0.1
    LEARNING_RATE = 0.1
    GENERATIONS = 50
    
    MODEL_IN = 1
    MODEL_HID = 16
    MODEL_OUT = 1
    
    # Calculate parameter size
    # L1: 16*1*5 = 80
    # L2: 1*16*5 = 80
    # Total = 160
    TOTAL_PARAMS = (MODEL_HID * MODEL_IN * 5) + (MODEL_OUT * MODEL_HID * 5)
    
    # 1. Initialize Central Parameters (The "Center" of the search)
    # This corresponds to W in eggroll_demo
    # Shape: [TOTAL_PARAMS, 1] so we can treat it as a vector for Rank1Noise
    # Actually Rank1NoiseGen takes (Out, In). We can set In=1, Out=TOTAL_PARAMS.
    center_params = torch.randn(TOTAL_PARAMS, 1, device=DEVICE) * 0.1
    
    # 2. Initialize Components
    # Note: Rank1NoiseGenerator.sample() returns A[M, Out], B[M, In]
    # Here A[M, Params], B[M, 1].
    # Perturbation = A @ B.T.   Here B is just scalars.
    # So Perturbation_i = A_i * B_i.
    
    noise_cfg = NoiseConfig(population_size=POPULATION_SIZE, sigma=SIGMA)
    noise_gen = Rank1NoiseGenerator(noise_cfg, output_size=TOTAL_PARAMS, input_size=1, device=DEVICE)
    
    # Note: We don't use VectorizedScorer because our model is non-linear/stateful.
    # We use our own evaluation loop (BatchedHebbianNet).
    
    fitness_shaper = FitnessShaper()
    antithetic_shaper = AntitheticShaper() # Reduces 2M fitnesses to M updates
    
    update_cfg = UpdateConfig(learning_rate=LEARNING_RATE, weight_decay=0.0) # W_decay 0 for fairness with hebb_meta
    updater = EGGROLLUpdater(update_cfg)
    
    # 3. Initialize Batched Model Container
    # We will load perturbed weights into this model every step
    batched_model = BatchedHebbianNet(MODEL_IN, MODEL_HID, MODEL_OUT, POPULATION_SIZE)
    batched_model.to(DEVICE)
    
    print(f"--- Eggroll Hebbian Training Started ---")
    print(f"Population: {POPULATION_SIZE}, Generations: {GENERATIONS}")
    print(f"Optimizing {TOTAL_PARAMS} params")
    
    start_time = time.time()
    
    for gen in range(GENERATIONS):
        # A. Sample Noise
        # A: [M, Params], B: [M, 1]
        # M = PopSize // 2
        A, B = noise_gen.sample()
        
        # B. Reconstruct Population Params
        # We need ALL perturbations (positive and negative)
        # Eggroll logic:
        # W_pos = W + sigma * A @ B.T
        # W_neg = W - sigma * A @ B.T
        
        # A @ B.T results in [M, Params]. Wait.
        # A: [M, Params], B: [M, 1]. 
        # A @ B.T is NOT what we want for broadcasting.
        # we want (A * B) basically since B is scalar per perturbation direction.
        
        # Let's look at Rank1NoiseGenerator logic: A[M, D_out], B[M, D_in]
        # We need perturbations P such that P_i = A_i * B_i (outer product usually).
        # Here D_in=1. So outer product is just scaling.
        # P = A * B -> [M, Params]
        
        perturbations = A * B # [M, Params]
        
        # Expand center params to [M, Params]
        center_flat = center_params.squeeze(1) # [Params]
        
        # Pos and Neg candidates
        # [M, Params]
        cands_pos = center_flat.unsqueeze(0) + SIGMA * perturbations
        cands_neg = center_flat.unsqueeze(0) - SIGMA * perturbations
        
        # Concatenate for evaluation: [2M, Params] -> [Pop, Params]
        all_cands = torch.cat([cands_pos, cands_neg], dim=0)
        
        # C. Evaluate Population
        batched_model.load_flat_params(all_cands)
        
        # Run!
        # Returns [Pop] tensor of losses
        losses = evaluate_population(batched_model, num_episodes=2)
        
        # D. Fitness (Negative Loss)
        fitness = -losses
        
        # E. Update
        # 1. Shape Fitness (Rank transform)
        # Note: FitnessShaper expects [Pop]
        shaped_fitness = fitness_shaper.rank_transform(fitness)
        
        # 2. Antithetic Combine
        # Returns [M]
        delta_fitness = antithetic_shaper(shaped_fitness)
        
        # 3. Apply Update
        # updater.apply_update(W, A, B, fitness)
        # It computes delta = (A * fitness) @ B
        # Let's assume W is [Params, 1]
        center_params = updater.apply_update(center_params, A, B, delta_fitness)
        
        if (gen + 1) % 5 == 0 or gen == 0:
            avg_loss = losses.mean().item()
            best_loss = losses.min().item()
            print(f"Gen {gen+1:3d} | Avg Adapt Loss: {avg_loss:.6f} | Best Loss: {best_loss:.6f}")
            
    train_time = time.time() - start_time
    print(f"Eggroll Training Complete in {train_time:.2f}s")
    
    # Return best model (center)
    final_model = BatchedHebbianNet(MODEL_IN, MODEL_HID, MODEL_OUT, population_size=1)
    final_model.to(DEVICE)
    final_model.load_flat_params(center_params.view(1, -1))
    return final_model

def evaluate_final(model):
    print("\n--- Final Verification ---")
    start = time.time()
    # Need a wrapper to evaluate single model instance with 'average' style
    # Our evaluate_population expects BatchedHebbianNet.
    # Our final_model IS a BatchedHebbianNet with pop=1.
    
    loss = evaluate_population(model, num_episodes=10, seq_len=30)
    print(f"Avg Prediction Error after Adaptation: {loss.item():.6f}")
    print(f"Verification time: {time.time() - start:.2f}s")

if __name__ == "__main__":
    model = train_eggroll_hebb()
    evaluate_final(model)

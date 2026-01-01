
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import os
import math
from typing import List, Tuple

# Use the model definition we just created
from model import HebbianReservoir

# --- Configuration ---
BATCH_SIZE = 256          # Sequences per step
SEQ_LEN = 128             # Length of sequence to train on (and lifetime of agent)
HIDDEN_SIZE = 512        # Reduced from 2048 to fit memory with full matrices
POPULATION_SIZE = 64     # Reduced population for memory safety
SIGMA = 0.1
LEARNING_RATE = 0.01     # For meta-parameters
STEPS = 5000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hebbian Rules are 5 coefficients: A, B, C, D, Eta
HEB_DIM = 5

# --- Utilities ---

def load_data(path):
    with open(path, 'r') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for i,ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, vocab_size, stoi, itos

def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)

def rank_transformation(fitness):
    N = len(fitness)
    indices = torch.argsort(fitness)
    ranks = torch.zeros_like(fitness)
    ranks[indices] = torch.arange(N, device=fitness.device, dtype=torch.float32)
    return (ranks / (N - 1)) - 0.5

# --- Eggroll Components ---

class Rank1NoiseGenerator:
    def __init__(self, size1, size2, device="cpu"):
        self.size1 = size1
        self.size2 = size2
        self.device = device
        
    def sample(self, pop_size) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns A, B such that Noise = A @ B.T
        # A: [M, size1], B: [M, size2]
        M = pop_size // 2
        A = torch.randn(M, self.size1, device=self.device)
        B = torch.randn(M, self.size2, device=self.device)
        return A, B

class ScalarNoiseGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        
    def sample(self, pop_size) -> torch.Tensor:
        M = pop_size // 2
        return torch.randn(M, device=self.device)

class AdamOptimizer:
    def __init__(self, params_shape, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.m = torch.zeros(params_shape, device=DEVICE)
        self.v = torch.zeros(params_shape, device=DEVICE)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.step = 0
        
    def update(self, param, grad):
        self.step += 1
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (grad ** 2)
        m_hat = self.m / (1 - self.betas[0] ** self.step)
        v_hat = self.v / (1 - self.betas[1] ** self.step)
        return param + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

# --- Vectorized Training Logic ---

def compute_fitness_vectorized(
    model,
    base_coeffs,        # [H, H, 5]
    coeff_noise_list,   # List of 5 tuples (A, B) for noise
    base_alpha,
    noise_alpha,
    sigma,
    data_x,             # [Batch, Seq]
    data_y              # [Batch, Seq]
):
    # Unpack basic dimensions
    pop_size = len(noise_alpha) * 2
    M = pop_size // 2
    H = model.hidden_size
    batch_size_val = data_x.shape[0]
    
    # 1. Reconstruct Population Parameters
    
    # Alpha: [Pop]
    E_alpha_all = torch.cat([noise_alpha, -noise_alpha], dim=0)
    alpha_pop = base_alpha + sigma * E_alpha_all # [Pop]
    
    # Coeffs: [Pop, H, H, 5]
    # We construct them layer by layer to save memory if needed, 
    # but here we build the full tensor for speed.
    # Base: [H, H, 5] -> Expand to [Pop, H, H, 5]
    # Noise: For each of 5 channels, compute A@B.T
    
    # Build population coefficients efficiently
    coeffs_pop = base_coeffs.unsqueeze(0).repeat(pop_size, 1, 1, 1)
    for i in range(5):
        A, B_mat = coeff_noise_list[i]
        noise = torch.bmm(A.unsqueeze(2), B_mat.unsqueeze(1))
        coeffs_pop[:M, ..., i] += sigma * noise
        coeffs_pop[M:, ..., i] -= sigma * noise
        
    # 2. Initialize Episodes
    
    # W_rec_pop: [Pop, H, H]
    # Each agent starts with a fresh random W_rec.
    # IMPORTANT: All agents share same initial random weights for fairness?
    # Or should each agent rely on its logic regardless of init?
    # Usually "Lifetime Learning" implies robustness to initial conditions.
    # We'll use ONE random init for the whole population this generation.
    
    # Generate one random W
    W_init = torch.randn(H, H, device=DEVICE)
    # Spectral radius
    eigs = torch.linalg.eigvals(W_init)
    max_eig = torch.max(torch.abs(eigs))
    if max_eig > 0: W_init *= (model.spectral_radius / max_eig)
    
    # Broadcast to population
    W_rec_pop = W_init.unsqueeze(0).repeat(pop_size, 1, 1) # [Pop, H, H]
    
    # State h: [Pop, Batch, H]
    h = torch.zeros(pop_size, batch_size_val, H, device=DEVICE)
    
    # Prepare Data
    # x: [Seq, Batch]
    x_seq = data_x.permute(1, 0)
    y_seq = data_y.permute(1, 0)
    
    # One Hot helper
    # We need to one-hot encode x for injection
    # In 'model.py' we assumed x is passed.
    # Here x is indices.
    
    all_loss = torch.zeros(pop_size, device=DEVICE)
    
    # 3. Time Loop
    for t in range(SEQ_LEN):
        xt = x_seq[t] # [Batch]
        yt = y_seq[t] # [Batch]
        
        # One Hot Encode: [Batch, Vocab]. We assume Vocab <= InputSize (65)
        # model.input_size should be the vocab size (65)
        xt_oh = F.one_hot(xt, model.input_size).float() # [Batch, In]
        
        # Input Injection
        # Update first `input_size` cols of h
        h_in = h.clone()
        h_in[:, :, :model.input_size] = xt_oh.unsqueeze(0) # Broadcast to Pop
        
        # Recurrent Projection
        # h_in: [Pop, B, H]
        # W_rec_pop: [Pop, H, H]
        # We need per-batch-item matmul.
        # h_in @ W.T 
        # [Pop, B, H] x [Pop, H, H] (transpose last two) -> [Pop, B, H]
        # torch.bmm works on 3D. We have Pop as batch dim.
        # W_rec_pop.transpose(1, 2) -> [Pop, H, H]
        
        # Wait, inside the population, we have 'B' batches of data.
        # Can we merge Pop and Batch dimensions?
        # W_rec is unique per Pop. 'B' shares W_rec.
        # h_in is unique per Pop and Batch.
        # Let's perform calculation per Pop:
        
        # Reshape h_in for bmm?
        # No, torch.bmm(h_in, W.transpose) works if h_in is [P, B, H] and W is [P, H, H].
        # It treats P as batch, and does (B,H) x (H,H) -> (B,H).
        
        pre_act = torch.bmm(h_in, W_rec_pop.transpose(1, 2))
        
        # Leaky Update
        # alpha_pop: [Pop]. Expand to [Pop, 1, 1]
        alpha = torch.clamp(alpha_pop, 0.01, 0.99).view(-1, 1, 1)
        
        update = torch.tanh(pre_act) # Mask is already in W_rec_pop
        # Note: W_rec logic in model.py applied mask to W. W_rec_pop should maintain sparsity 
        # if updates are masked!
        
        h_new = (1 - alpha) * h_in + alpha * update
        
        # Calculate Loss
        # Output: Last 65 neurons
        logits = h_new[:, :, -model.output_size:] # [Pop, Batch, Out]
        
        # targets: [Batch]. Expand to [Pop, Batch]
        targets = yt.unsqueeze(0).repeat(pop_size, 1)
        
        # Cross Entropy
        # Flatten: [Pop*Batch, Out]
        ce = F.cross_entropy(
            logits.reshape(-1, model.output_size),
            targets.reshape(-1),
            reduction='none'
        )
        # Reshape back to [Pop, Batch] and average over Batch
        loss_t = ce.view(int(pop_size), int(batch_size_val)).mean(dim=1)
        all_loss += loss_t
        
        # HEBBIAN UPDATE
        # Update W_rec_pop
        # pre: h_in [P, B, H]
        # post: h_new [P, B, H]
        # coeffs: [P, H, H, 5]
        
        # Calculate Delta
        # A, B, C, D, Eta
        A = coeffs_pop[..., 0] # [P, H, H]
        B = coeffs_pop[..., 1]
        C = coeffs_pop[..., 2]
        D = coeffs_pop[..., 3]
        Eta = coeffs_pop[..., 4]
        
        # ABCD coefficients
        A, B, C, D, Eta = [coeffs_pop[..., i] for i in range(5)] # [P, H, H]
        
        # Compute term-wise delta (averaged over batch)
        # term1: A * (post * pre^T)
        # h_new: [P, B, H], h_in: [P, B, H]
        # We want sum over B of h_new[p,b,:].unsqueeze(2) @ h_in[p,b,:].unsqueeze(1)
        # This is equivalent to einsum('pbi, pbj -> pij', h_new, h_in) / batch_size_val
        pre_post_avg = torch.einsum('pbi, pbj -> pij', h_new, h_in) / batch_size_val # [P, H, H]
        
        # term2/3: B * pre / C * post
        # pre_mean: [P, H], post_mean: [P, H]
        pre_mean = h_in.mean(dim=1) # [P, H]
        post_mean = h_new.mean(dim=1) # [P, H]
        
        # Expand for broadcasting:
        # B: [P, H, H], pre_mean: [P, 1, H]
        # C: [P, H, H], post_mean: [P, H, 1]
        
        delta = Eta * (A * pre_post_avg + B * pre_mean.unsqueeze(1) + C * post_mean.unsqueeze(2) + D)
        
        # Apply
        W_rec_pop.add_(delta)
        
        # Normalize
        # Max per individual
        # [P, H, H] -> Max over dim 1,2 -> [P]
        max_vals = W_rec_pop.abs().flatten(1).max(1)[0].reshape(pop_size, 1, 1)
        # Only normalize if max_val > 5.0 (or some threshold)
        # This avoids dividing by very small numbers when weights are small.
        # For vectorized, we need to apply this conditionally per population member.
        # Create a mask for elements that exceed the threshold
        normalize_mask = (max_vals > 5.0)
        # Apply division only where mask is true
        W_rec_pop = torch.where(normalize_mask, W_rec_pop / (max_vals + 1e-8), W_rec_pop)
        
        # Update h for next step
        h = h_new
        
    return -all_loss / SEQ_LEN # Negative loss handles 'fitness' maximization


@torch.no_grad()
def evaluate_meta(model, base_coeffs, base_alpha, data, batch_size, seq_len, device):
    """Evaluates the base (evolved) coefficients on unseen data."""
    model.eval()
    x, y = get_batch(data, batch_size, seq_len, device)
    
    H, B = model.hidden_size, x.shape[0]
    VOCAB = model.input_size
    
    # Setup model for evaluation
    model.leaky_alpha.data.fill_(base_alpha.item())
    model.reset_weights() # Fresh random weights for the evaluation session
    
    h = torch.zeros(B, H, device=device)
    total_loss = 0
    
    for t in range(seq_len):
        xt_oh = F.one_hot(x[:, t], VOCAB).float()
        
        h_pre = h.clone()
        h = model.forward(xt_oh, h)
        
        logits = model.get_output(h)
        total_loss += F.cross_entropy(logits, y[:, t]).item()
        
        # Meta-generalization: update weights while solving the task
        model.apply_hebbian_update(h_pre, h, base_coeffs)
        
    return total_loss / seq_len

# --- Main ---

def main():
    wandb.init(project="hebbian-meta-learning", name="reservoir-eggroll")
    
    # Data
    data_path = os.path.join(os.path.dirname(__file__), "../evo_res/tinyshakespeare.txt")
    if not os.path.exists(data_path):
        # Fallback to current dir or check paths
        possible = ["tinyshakespeare.txt", "evo_res/tinyshakespeare.txt", "../evo_res/tinyshakespeare.txt"]
        for p in possible:
            if os.path.exists(p):
                data_path = p
                break
                
    print(f"Loading data from {data_path}")
    raw_data, vocab_size, stoi, itos = load_data(data_path)
    
    # 90-10 Split
    n_train = int(len(raw_data) * 0.9)
    train_data = raw_data[:n_train]
    test_data = raw_data[n_train:]
    
    print(f"Data split: Train {len(train_data)}, Test {len(test_data)}")
    
    # Model Init
    # Input/Output sizes match vocab. 
    # Logic: "input will be simply one_hot encoded in the first 65 neurons"
    # "output will be simply the last 65 neurons"
    # Assuming vocab_size is around 65.
    
    model = HebbianReservoir(
        input_size=vocab_size, 
        hidden_size=HIDDEN_SIZE, 
        output_size=vocab_size
    ).to(DEVICE)
    
    # Meta Parameters (The Genome)
    # Hebbian Coeffs: [H, H, 5]
    base_coeffs = torch.randn(HIDDEN_SIZE, HIDDEN_SIZE, HEB_DIM, device=DEVICE) * 0.01
    base_alpha = model.leaky_alpha.data.clone()
    
    # Optimizers
    noise_gens_coeffs = [Rank1NoiseGenerator(HIDDEN_SIZE, HIDDEN_SIZE, DEVICE) for _ in range(HEB_DIM)]
    noise_gen_alpha = ScalarNoiseGenerator(device=DEVICE)
    
    opt_coeffs = AdamOptimizer(base_coeffs.shape, lr=LEARNING_RATE)
    opt_alpha = AdamOptimizer(base_alpha.shape, lr=LEARNING_RATE)
    
    print(f"Starting Training: Pop {POPULATION_SIZE}, Hidden {HIDDEN_SIZE}, Device {DEVICE}")
    
    for step in range(STEPS):
        # Get Data
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # 1. Sample Noise
        # Alpha
        noise_alpha_samp = noise_gen_alpha.sample(POPULATION_SIZE)
        
        # Coeffs (List of (A, B))
        noise_coeffs_samps = [g.sample(POPULATION_SIZE) for g in noise_gens_coeffs]
        
        # 2. Evaluate Fitness (Vectorized)
        fitness = compute_fitness_vectorized(
            model,
            base_coeffs,
            noise_coeffs_samps,
            base_alpha,
            noise_alpha_samp,
            SIGMA,
            x, y
        )
        
        # 3. Optimize
        # Rank Transform
        shaped_fit = rank_transformation(fitness)
        
        # Reconstruct Gradients
        M = POPULATION_SIZE // 2
        diff_fit = (shaped_fit[:M] - shaped_fit[M:]) / 2.0
        
        # Alpha Grad
        grad_alpha = (diff_fit * noise_alpha_samp).sum() / (M * SIGMA)
        base_alpha = opt_alpha.update(base_alpha, grad_alpha)
        base_alpha = torch.clamp(base_alpha, 0.01, 0.99)
        
        # Coeffs Grads (Loop over 5 types)
        grad_coeffs = torch.zeros_like(base_coeffs)
        
        for c_idx in range(HEB_DIM):
            A, B_mat = noise_coeffs_samps[c_idx] # [M, H]
            # Grad = sum(diff * A * B.T)
            # diff is [M]. We broadcast.
            # grad = torch.einsum('m,mh,mk->hk', diff_fit, A, B_mat)
            grad_c = torch.einsum('m,mh,mk->hk', diff_fit, A, B_mat)
            grad_coeffs[..., c_idx] = grad_c / (M * SIGMA)
            
        base_coeffs = opt_coeffs.update(base_coeffs, grad_coeffs)
        
        # Logging
        loss = -fitness.max().item()
        log_dict = {"train/loss": loss, "train/alpha": base_alpha.item()}
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss:.4f}, Alpha {base_alpha.item():.3f}")
            
        if step % 100 == 0:
            test_loss = evaluate_meta(model, base_coeffs, base_alpha, test_data, BATCH_SIZE, SEQ_LEN, DEVICE)
            print(f"Step {step}: Test Loss {test_loss:.4f}")
            log_dict["test/loss"] = test_loss
            
        wandb.log(log_dict)
            
    wandb.finish()

if __name__ == "__main__":
    main()

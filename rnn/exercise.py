import torch
import torch.nn as nn
import argparse
import time
import math
from eggroll.noise import NoiseConfig, Rank1NoiseGenerator
from eggroll.update import UpdateConfig, EGGROLLUpdater

class Reservoir(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=0.9, sparsity=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Fixed input weights
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1, requires_grad=False)
        
        # Fixed recurrent weights
        W_res = torch.randn(hidden_size, hidden_size)
        # Apply sparsity
        mask = torch.rand(hidden_size, hidden_size) > sparsity
        W_res[mask] = 0
        # Spectral radius normalization
        eigenvalues = torch.linalg.eigvals(W_res)
        max_eigenvalue = torch.max(torch.abs(eigenvalues))
        W_res = W_res * (spectral_radius / max_eigenvalue)
        
        self.W_res = nn.Parameter(W_res, requires_grad=False)
        self.tanh = nn.Tanh()

    def forward(self, x, h_0=None):
        """
        x: [batch, seq_len, input_size]
        h_0: [batch, hidden_size]
        """
        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_0

        states = []
        for t in range(seq_len):
            u_t = x[:, t, :]
            h = self.tanh(torch.matmul(u_t, self.W_in.T) + torch.matmul(h, self.W_res.T))
            states.append(h)
        
        return torch.stack(states, dim=1), h

class ESNReadout(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        # Initialize with zeros or small random values
        self.W_out = nn.Parameter(torch.zeros(output_size, hidden_size))

    def forward(self, states):
        # states: [batch, seq_len, hidden_size]
        return torch.matmul(states, self.W_out.T)

def generate_sine_data(batch_size, seq_len, device):
    t = torch.linspace(0, 10, seq_len + 1, device=device).unsqueeze(0).expand(batch_size, -1)
    # Add random phase shift
    phase = torch.rand(batch_size, 1, device=device) * 2 * math.pi
    data = torch.sin(t + phase)
    
    x = data[:, :-1].unsqueeze(-1) # Input: sin(t)
    y = data[:, 1:].unsqueeze(-1)  # Target: sin(t+1)
    return x, y

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    input_size = 1
    hidden_size = 128
    output_size = 1
    seq_len = 50
    batch_size = 32
    iterations = 200
    
    # Components
    reservoir = Reservoir(input_size, hidden_size).to(device)
    # Ideally we'd use these weights, but Eggroll manages its own weights conceptually
    # We will maintain W_out explicitly to use with Eggroll tools
    W_out = torch.zeros(output_size, hidden_size, device=device)

    # Eggroll Config
    noise_config = NoiseConfig(
        population_size=args.pop_size,
        sigma=args.sigma,
        seed=42
    )
    update_config = UpdateConfig(
        learning_rate=args.lr,
        momentum=0.9
    )
    
    noise_gen = Rank1NoiseGenerator(noise_config, output_size, hidden_size, device=str(device))
    updater = EGGROLLUpdater(update_config, output_size, hidden_size, device=str(device))
    
    print("Starting training...")
    start_time = time.time()
    
    for i in range(iterations):
        # 1. Get Data and Reservoir States
        x, target = generate_sine_data(batch_size, seq_len, device)
        with torch.no_grad(): # Reservoir is fixed
            states, _ = reservoir(x)
            # Flatten batch and time for readout training: [B*T, H]
            # Actually, to compute fitness per "batch" or "trajectory" is better?
            # Let's simple treat B*T as samples.
            
            # Eggroll expects H_q (inputs to linear layer) and H_d (context)
            # But here we just have simple Linear regression setup: Y = X @ W.T
            # Let's map to Eggroll's vectorized scoring concept.
            # In simple regression, score = -MSE.
            # score_i = - mean((Y_pred_i - Y_target)^2)
            
            # Let's implement the efficient rank-1 scoring manually for regression 
            # as scoring.py is optimized for Dot-Product Attention / Retrieval.
            
            # H: [B*T, hidden_size]
            H = states.reshape(-1, hidden_size)
            # targets: [B*T, output_size]
            targets = target.reshape(-1, output_size)
            
            # 2. Sample Perturbations
            # A: [M, output_size], B: [M, input_size=hidden_size]
            # delta_W = A @ B.T (broadcasting over M)
            A, B = noise_gen.sample(step=i)
            M = A.shape[0]
            
            # 3. Compute Perturbed Outputs Efficiently
            # Y_base = H @ W.T
            Y_base = H @ W_out.T
            
            # We want Y_pert = H @ (W + sigma * a @ b.T).T
            #              = H @ W.T + sigma * H @ b @ a.T
            # Let's compute H @ b first (Proj Inputs)
            
            # H_B = H @ B.T -> [N_samples, M]
            H_B = H @ B.T
            
            # Perturbation term: sigma * H_B @ A 
            # Wait, dim mismatch. A is [M, out].
            # We want for each m: (H @ b_m) * a_m^T
            # H_B col m is (H @ b_m). 
            # We want result [N_samples, out].
            # term_m = (H @ b_m).unsqueeze(1) * a_m.unsqueeze(0)
            
            # Let's proceed carefully with full population evaluation.
            # We have 2*M candidates (antithetic).
            
            # Vectorized approach:
            # P = H @ B.T # [N_samples, M]
            # term = P.unsqueeze(-1) * A.unsqueeze(0) # [N_samples, M, out]
            
            # Optimization: since we compute MSE, maybe we can keep it cleaner.
            sigma = args.sigma
            
            # Pos candidates
            Y_pos = Y_base.unsqueeze(1) + sigma * (H @ B.T).unsqueeze(-1) * A.unsqueeze(0) # [N, M, out]
            # Neg candidates
            Y_neg = Y_base.unsqueeze(1) - sigma * (H @ B.T).unsqueeze(-1) * A.unsqueeze(0)
            
            # 4. Compute Fitness (Negative MSE)
            # targets: [N, out] -> expand to [N, 1, out]
            T_expanded = targets.unsqueeze(1)
            
            mse_pos = ((Y_pos - T_expanded)**2).mean(dim=(0, 2)) # Average over samples and outputs -> [M]
            mse_neg = ((Y_neg - T_expanded)**2).mean(dim=(0, 2)) # [M]
            
            fitness_pos = -mse_pos
            fitness_neg = -mse_neg
            
            # Combine for update
            # Eggroll update rule: sum( fitness * direction )
            # directions are A_m @ B_m.T
            # We need to pass shaped fitness to updater.
            
            # Antithetic shaping: (f_pos - f_neg)/2
            # This is simplified. Eggroll's shaped_fitness expects [2*M] usually if using ranking.
            # But let's use the raw difference for regression which is often robust enough, 
            # or use the provided FitnessShaper?
            # Let's try raw antithetic diff first for simplicity as in simple ES.
            # Actually, let's look at updater.compute_update signature:
            # takes A, B, shaped_fitness (size M).
            # If we pass M shaped fitness values, it assumes they correspond to the directions.
            # If using antithetic, the "gradient estimate" for direction m is roughly ~ (J(w+) - J(w-))/(2 sigma)
            
            # Let's construct the "fitness" for the direction.
            # approximate gradient along direction m is (f_pos - f_neg) / (2*sigma)
            # approximate update is learning_rate * grad.
            # The updater computes: sum( fitness[m] * A[m] @ B[m].T )
            
            # So if we set input fitness[m] = (f_pos[m] - f_neg[m]) / (2*sigma), 
            # then updater calculates step = mean( fitness[m] * dir[m] )
            # We want to maximize fitness (minimize MSE).
            
            raw_diff = (fitness_pos - fitness_neg) / (2 * sigma)
            
            # 5. Update Weights
            # Note: The updater divides by M inside compute_update.
            W_out = updater.apply_update(W_out, A, B, raw_diff, sigma)
            
            # Logging
            current_mse = ((Y_base - targets)**2).mean().item()
            if i % 20 == 0:
                print(f"Iter {i}: MSE = {current_mse:.6f}")

    print(f"Final MSE: {current_mse:.6f}")
    print(f"Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--pop_size", type=int, default=128)
    args = parser.parse_args()
    
    train(args)

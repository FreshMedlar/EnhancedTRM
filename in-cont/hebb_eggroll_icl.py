import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import time
import argparse
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

# Force WandB to offline mode
os.environ["WANDB_MODE"] = "online"

# --- Hyperparameters ---
@dataclass
class Config:
    # Task Params (from tran_in_contx.py)
    BATCH_SIZE: int = 64
    SEQ_LEN: int = 50
    X_DIM: int = 20
    
    # Evolution Params (Eggroll)
    POP_SIZE: int = 64  # Total population (must be even for antithetic)
    SIGMA: int = 0.02   # Noise standard deviation
    ES_LR: float = 0.1  # ES learning rate
    STEPS: int = 1000   # Evolution generations
    
    # Model Params
    NUM_HEBB_COEFFS: int = 9  # A, B, C, D, E, F (prev x), G (prev y), H (prev x*y), eta
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# --- 1. Data Generation (Linear Regression) ---
def get_batch(batch_size, seq_len, x_dim, device):
    # Same logic as tran_in_contx.py
    w = torch.randn(batch_size, x_dim, 1, device=device)
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    y = x @ w  # (B, L, 1)
    return x, y, w

# --- 2. Hebbian Model (Vectorized across Population) ---
class HebbianICLModel:
    """
    Handles a population of single-layer Hebbian networks.
    Each network starts with random weights and updates via evolved Hebbian rules.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Per-synapse coefficients: (1, X_DIM, 6)
        self.coeffs = torch.zeros(1, cfg.X_DIM, cfg.NUM_HEBB_COEFFS, device=cfg.DEVICE)
        
    def evaluate_population(self, x_seq, y_seq, pop_coeffs):
        """
        pop_coeffs: [P, X_DIM, 6]
        x_seq: [B, L, X_DIM]
        y_seq: [B, L, 1]
        """
        P = pop_coeffs.shape[0]
        B, L, Dx = x_seq.shape
        device = self.cfg.DEVICE
        
        # Reshape coeffs for broadcasting: [P, 1, Dx, 6]
        c = pop_coeffs.unsqueeze(1) 
        
        # Initialize Weights for each population member and batch: [P, B, 1, Dx]
        W = torch.randn(P, B, 1, Dx, device=device) * 0.1
        xt_prev = torch.zeros(P, B, Dx, device=device)
        yt_prev = torch.zeros(P, B, 1, device=device)
        
        total_loss = torch.zeros(P, device=device)
        per_pos_loss = torch.zeros(L, device=device)
        
        for t in range(L):
            xt = x_seq[:, t, :].unsqueeze(0).expand(P, -1, -1)   # [P, B, Dx]
            yt = y_seq[:, t, :].unsqueeze(0).expand(P, -1, -1)   # [P, B, 1]
            
            # 1. Forward Pass: y_pred = W @ x
            y_pred = torch.einsum('pbio,pbo->pbi', W, xt)      # [P, B, 1]
            
            # 2. Compute Loss
            step_loss = (y_pred - yt).pow(2)                   # [P, B, 1]
            total_loss += step_loss.mean(dim=(1, 2))
            per_pos_loss[t] = step_loss.mean()
            
            # 3. Hebbian Update (ABCD+E Rule)
            # Delta W = eta * (A*y*x + B*x + C*y + D - E*W)
            x_exp = xt.unsqueeze(2)
            y_exp = yt.unsqueeze(3).expand(-1, -1, 1, Dx)
            
            A = c[..., 0].unsqueeze(1) # [P, 1, 1, Dx]
            B = c[..., 1].unsqueeze(1)
            C = c[..., 2].unsqueeze(1)
            D = c[..., 3].unsqueeze(1)
            E = c[..., 4].unsqueeze(1)
            F_coeff = c[..., 5].unsqueeze(1)
            G_coeff = c[..., 6].unsqueeze(1)
            H_coeff = c[..., 7].unsqueeze(1)
            eta = c[..., 8].unsqueeze(1)
            
            x_prev_exp = xt_prev.unsqueeze(2) # [P, B, 1, Dx]
            y_prev_exp = yt_prev.unsqueeze(3).expand(-1, -1, 1, Dx) # [P, B, 1, Dx]
            prod_prev_exp = y_prev_exp * x_prev_exp
            
            delta_w = eta * (
                A * y_exp * x_exp +    # A * y * x
                B * x_exp +            # B * x
                C * y_exp +            # C * y
                D -                   # D
                E * W +               # -E * W
                F_coeff * x_prev_exp + # F * x_{t-1}
                G_coeff * y_prev_exp + # G * y_{t-1}
                H_coeff * prod_prev_exp # H * x_{t-1} * y_{t-1}
            )
            
            W = W + delta_w
            W = torch.clamp(W, -10.0, 10.0)
            xt_prev = xt.clone()
            yt_prev = yt.clone()
            
        fitness = -total_loss / L
        return fitness, per_pos_loss

# --- 3. Eggroll Evolution Strategy Components ---

class Rank1NoiseGenerator:
    """Generates rank-1 noise for the coefficients tensor."""
    def __init__(self, M, D_out, D_in, device, sigma=0.02):
        self.M = M
        self.D_out = D_out
        self.D_in = D_in
        self.device = device
        self.sigma = sigma
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        A = torch.randn(self.M, self.D_out, device=self.device)
        B = torch.randn(self.M, self.D_in, device=self.device)
        return A, B

class FitnessShaper:
    def rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """Transform fitness to ranks in [-0.5, 0.5]."""
        N = fitness.shape[0]
        _, indices = fitness.sort()
        ranks = torch.zeros_like(fitness)
        ranks[indices] = torch.arange(N, dtype=fitness.dtype, device=fitness.device)
        return (ranks / (N - 1)) - 0.5

class EGGROLLUpdater:
    def __init__(self, lr=0.1):
        self.lr = lr
        
    def compute_update(self, A, B, shaped_fitness):
        M = A.shape[0]
        weighted_A = A * shaped_fitness.unsqueeze(-1)
        delta_W = weighted_A.T @ B
        return delta_W / M

# --- 4. Main Training Loop ---

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=cfg.STEPS)
    parser.add_argument("--pop_size", type=int, default=cfg.POP_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.ES_LR)
    parser.add_argument("--sigma", type=float, default=cfg.SIGMA)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    cfg.STEPS = args.steps
    cfg.POP_SIZE = args.pop_size
    cfg.ES_LR = args.lr
    cfg.SIGMA = args.sigma
    
    print(f"=== Hebbian Eggroll ICL Demo (Original Stabler Version) ===")
    print(f"Device: {cfg.DEVICE} | Coeffs: {cfg.NUM_HEBB_COEFFS}")
    
    if args.wandb:
        wandb.init(project="In-Context-Learning", config=vars(cfg))
    else:
        wandb.init(mode="disabled")
    
    model = HebbianICLModel(cfg)
    
    M = cfg.POP_SIZE // 2
    noise_gen = Rank1NoiseGenerator(M, cfg.X_DIM, cfg.NUM_HEBB_COEFFS, cfg.DEVICE, cfg.SIGMA)
    fitness_shaper = FitnessShaper()
    updater = EGGROLLUpdater(lr=cfg.ES_LR)
    
    for step in range(cfg.STEPS):
        start_time = time.time()
        x_seq, y_seq, _ = get_batch(cfg.BATCH_SIZE, cfg.SEQ_LEN, cfg.X_DIM, cfg.DEVICE)
        
        # Sample Rank-1 Noise
        A, B = noise_gen.sample()
        noise = torch.einsum('mi,mj->mij', A, B) * cfg.SIGMA
        
        W_curr = model.coeffs.squeeze(0)
        pop_coeffs = torch.cat([W_curr.unsqueeze(0) + noise, W_curr.unsqueeze(0) - noise], dim=0)
        
        fitness, per_pos_loss_pop = model.evaluate_population(x_seq, y_seq, pop_coeffs)
        
        shaped_f_all = fitness_shaper.rank_transform(fitness)
        shaped_fitness = (shaped_f_all[:M] - shaped_f_all[M:]) / 2.0
        
        delta_W = updater.compute_update(A, B, shaped_fitness)
        norm = torch.norm(delta_W)
        if norm > 1.0: delta_W = delta_W / norm
            
        model.coeffs = (W_curr + cfg.ES_LR * delta_W).unsqueeze(0)
        
        current_loss = -fitness.mean().item()
        wandb.log({"loss": current_loss, "fitness_max": fitness.max().item()}, step=step)
        
        if step % 20 == 0:
            print(f"Step {step:04d} | Loss: {current_loss:.4f} | Max Fit: {fitness.max().item():.4f} | Time: {time.time() - start_time:.2f}s")
        
    print("\nTraining Complete.")
    
    x_test, y_test, _ = get_batch(1000, cfg.SEQ_LEN, cfg.X_DIM, cfg.DEVICE)
    _, test_per_pos_loss = model.evaluate_population(x_test, y_test, model.coeffs)
    
    mse_per_position = test_per_pos_loss.cpu().numpy()
    print(f"\n{'Pos':<5} | {'MSE Loss':<10} | {'Visual'}")
    print("-" * 35)
    max_l = mse_per_position[0]
    for i, loss in enumerate(mse_per_position):
        bar_len = int((loss / max_l) * 20) if max_l > 0 else 0
        bar = "█" * min(bar_len, 40)
        print(f"{i:<5} | {loss:.4f}     | {bar}")
        
    wandb.finish()

if __name__ == "__main__":
    train()

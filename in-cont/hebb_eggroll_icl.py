import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import time
import argparse
import os
import sys
from typing import Tuple, List, Optional
from dataclasses import dataclass

# Add project root to path to allow importing from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hebb import HebbianICLModel, Rank1NoiseGenerator, FitnessShaper, EGGROLLUpdater

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
# Models and ES components are now imported from models.hebb

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

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
from models.hebb import BatchedHebbianNet
from models.eggroll import (
    Rank1NoiseGenerator, 
    NoiseConfig, 
    FitnessShaper, 
    AntitheticShaper, 
    EGGROLLUpdater, 
    UpdateConfig
)

# --- Hyperparameters ---
@dataclass
class Config:
    # Task Params (Matching RNN baseline)
    BATCH_SIZE: int = 128
    SEQ_LEN: int = 50          
    INPUT_DIM: int = 20            
    HIDDEN_DIM: int = 0
    OUTPUT_DIM: int = 1
    
    # Evolution Params (Eggroll)
    POP_SIZE: int = 64  
    SIGMA: float = 0.02
    ES_LR: float = 0.1
    STEPS: int = 1000   
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# --- 1. Data Generation (Linear Regression Case) ---
def get_batch(batch_size, seq_len, x_dim, device):
    # Standard ICL task: predicting y = x @ w
    # Normalize inputs for stability
    w = torch.randn(batch_size, x_dim, 1, device=device) / np.sqrt(x_dim)
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    y = x @ w 
    return x, y

# --- 2. Evaluation Loop ---
def evaluate_population(model: BatchedHebbianNet, seq_len=50):
    """
    Runs episodes for the entire population in parallel.
    Uses the labels for Hebbian adaptation (Supervised Hebbian).
    """
    pop_size = model.layer1.population_size
    x_seq, y_seq = get_batch(cfg.BATCH_SIZE, seq_len, cfg.INPUT_DIM, model.device)
    
    total_loss = torch.zeros(pop_size, device=model.device)
    first_step_loss = torch.zeros(pop_size, device=model.device)
    last_step_loss = torch.zeros(pop_size, device=model.device)
    
    model.reset_episode()
    
    for t in range(seq_len):
        x_step = x_seq[:, t, :].unsqueeze(0).expand(pop_size, -1, -1) 
        y_target = y_seq[:, t, :].unsqueeze(0).expand(pop_size, -1, -1) 
        
        h, pred = model(x_step) 
        
        step_loss = ((pred - y_target)**2).mean(dim=(1, 2))
        total_loss += step_loss
        
        if t == 0: first_step_loss = step_loss
        if t == seq_len - 1: last_step_loss = step_loss
        
        # Update plasticity using error for Delta-rule learning
        error = y_target - pred
        model.update(x_step, h, error)
        
    return total_loss / seq_len, first_step_loss, last_step_loss

# --- 3. Main Training Loop ---

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=cfg.STEPS)
    parser.add_argument("--pop_size", type=int, default=cfg.POP_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.ES_LR)
    parser.add_argument("--sigma", type=float, default=cfg.SIGMA)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    cfg.STEPS = args.steps
    cfg.POP_SIZE = args.pop_size if args.pop_size % 2 == 0 else args.pop_size + 1
    cfg.ES_LR = args.lr
    cfg.SIGMA = args.sigma
    
    print(f"=== Hebbian Eggroll ICL (Supervised Update) ===")
    print(f"Device: {cfg.DEVICE} | Pop: {cfg.POP_SIZE} | Steps: {cfg.STEPS} | Dim: {cfg.INPUT_DIM}")
    
    if args.wandb:
        wandb.init(project="In-Context-Learning-Hebbian", config=vars(cfg))
    else:
        wandb.init(mode="disabled")
    
    # Initialize Param Shapes
    dummy_model = BatchedHebbianNet(cfg.INPUT_DIM, cfg.HIDDEN_DIM, cfg.OUTPUT_DIM, cfg.POP_SIZE, device=cfg.DEVICE)
    TOTAL_PARAMS = dummy_model.total_params
    print(f"Total Meta-Params (Coefficients): {TOTAL_PARAMS}")
    
    center_params = torch.randn(TOTAL_PARAMS, 1, device=cfg.DEVICE) * 0.01
    noise_cfg = NoiseConfig(population_size=cfg.POP_SIZE, sigma=cfg.SIGMA)
    noise_gen = Rank1NoiseGenerator(noise_cfg, output_size=TOTAL_PARAMS, input_size=1, device=cfg.DEVICE)
    
    fitness_shaper = FitnessShaper()
    antithetic_shaper = AntitheticShaper()
    update_cfg = UpdateConfig(learning_rate=cfg.ES_LR, weight_decay=1e-4) # Small decay for stability
    updater = EGGROLLUpdater(update_cfg)
    
    batched_model = BatchedHebbianNet(cfg.INPUT_DIM, cfg.HIDDEN_DIM, cfg.OUTPUT_DIM, cfg.POP_SIZE, device=cfg.DEVICE)
    
    start_time = time.time()
    for gen in range(cfg.STEPS):
        A, B = noise_gen.sample()
        perturbations = A * B 
        center_flat = center_params.squeeze(1)
        
        all_cands = torch.cat([
            center_flat.unsqueeze(0) + cfg.SIGMA * perturbations,
            center_flat.unsqueeze(0) - cfg.SIGMA * perturbations
        ], dim=0)
        
        batched_model.load_flat_params(all_cands)
        losses, first_losses, last_losses = evaluate_population(batched_model, seq_len=cfg.SEQ_LEN)
        
        fitness = -losses
        shaped_fitness = fitness_shaper.rank_transform(fitness)
        delta_fitness = antithetic_shaper(shaped_fitness) 
        
        center_params = updater.apply_update(center_params, A, B, delta_fitness)
        
        avg_loss = losses.mean().item()
        best_loss = losses.min().item()
        wandb.log({
            "loss": avg_loss, 
            "best_loss": best_loss,
            "first_step_loss": first_losses.mean().item(),
            "last_step_loss": last_losses.mean().item()
        }, step=gen)
        
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"Gen {gen+1:4d} | Avg L: {avg_loss:.4f} | Best L: {best_loss:.4f} | First L: {first_losses.mean().item():.4f} | Last L: {last_losses.mean().item():.4f}")
            
    wandb.finish()
    
    # Final Visual Verification
    print("\n--- Final Test (Loss per Position) ---")
    final_model = BatchedHebbianNet(cfg.INPUT_DIM, cfg.HIDDEN_DIM, cfg.OUTPUT_DIM, population_size=1, device=cfg.DEVICE)
    final_model.load_flat_params(center_params.view(1, -1))
    
    final_model.reset_episode()
    test_x, test_y = get_batch(100, cfg.SEQ_LEN, cfg.INPUT_DIM, cfg.DEVICE)
    
    pos_losses = []
    for t in range(cfg.SEQ_LEN):
        x_step = test_x[:, t, :].unsqueeze(0) # [1, B, D]
        y_target = test_y[:, t, :].unsqueeze(0)
        
        h, pred = final_model(x_step)
        err = ((pred - y_target)**2).mean().item()
        pos_losses.append(err)
        
        # Supervised update using ERROR
        # This enables Delta-rule learning
        error = y_target - pred
        final_model.update(x_step, h, error)
        
    print(f"\n{'Pos':<5} | {'MSE Loss':<10} | {'Visual'}")
    print("-" * 35)
    max_l = pos_losses[0]
    for i, loss in enumerate(pos_losses):
        bar_len = int((loss / max_l) * 20) if max_l > 0 else 0
        bar = "█" * bar_len
        print(f"{i:<5} | {loss:.4f}     | {bar}")

if __name__ == "__main__":
    train()

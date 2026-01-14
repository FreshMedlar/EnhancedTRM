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

# --- Hyperparameters ---
@dataclass
class Config:
    # Task Params
    BATCH_SIZE: int = 32
    SEQ_LEN: int = 50
    X_DIM: int = 128     # Total input dimension across all layers (will be updated)
    LAYER_DIMS: List[int] = None # Will be [D_in, D_h1, D_out]
    
    # Evolution Params (Eggroll)
    POP_SIZE: int = 64  # Total population
    SIGMA: float = 0.02   # Noise standard deviation
    ES_LR: float = 0.1  # ES learning rate
    STEPS: int = 500    # Evolution generations
    
    # Model Params
    NUM_HEBB_COEFFS: int = 6  # A, B, C, D, E, eta
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

# --- 1. Data Preparation (Tiny Shakespeare) ---
def prepare_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    # 90-10 split
    n = int(0.9 * len(data))
    train_data = data[:n]
    test_data = data[n:]
    
    return train_data, test_data, vocab_size, stoi, itos

def get_batch(data, batch_size, seq_len, vocab_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x_indices = torch.stack([data[i:i+seq_len] for i in ix])
    y_indices = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    
    # Convert to one-hot for ICL task
    # x: [B, L, vocab_size], y: [B, L, vocab_size]
    x = F.one_hot(x_indices, num_classes=vocab_size).float().to(device)
    y = F.one_hot(y_indices, num_classes=vocab_size).float().to(device)
    
    return x, y

# --- 2. Training Loop ---
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
    
    print(f"=== Shakespeare Two-Block Hebbian ICL ===")
    
    train_data, test_data, vocab_size, stoi, itos = prepare_data("shake/tinyshakespeare.txt")
    
    # Configure two blocks: vocab_size -> vocab_size -> vocab_size
    cfg.LAYER_DIMS = [vocab_size, vocab_size, vocab_size]
    cfg.X_DIM = sum(cfg.LAYER_DIMS[:-1]) # dim_in + dim_h1
    
    print(f"Vocab Size: {vocab_size} | Layers: {cfg.LAYER_DIMS} | Total X_DIM: {cfg.X_DIM}")
    print(f"Device: {cfg.DEVICE}")
    
    if args.wandb:
        wandb.init(project="In-Context-Learning", name="Shakespeare-TwoBlock-Hebb-ICL", config=vars(cfg))
    else:
        wandb.init(mode="disabled")
    
    model = HebbianICLModel(cfg)
    
    M = cfg.POP_SIZE // 2
    # noise_gen: (M, total_X_DIM, NUM_COEFFS)
    noise_gen = Rank1NoiseGenerator(M, cfg.X_DIM, cfg.NUM_HEBB_COEFFS, cfg.DEVICE, cfg.SIGMA)
    fitness_shaper = FitnessShaper()
    updater = EGGROLLUpdater(lr=cfg.ES_LR)
    
    for step in range(cfg.STEPS):
        start_time = time.time()
        x_seq, y_seq = get_batch(train_data, cfg.BATCH_SIZE, cfg.SEQ_LEN, vocab_size, cfg.DEVICE)
        
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
    
    # Final Evaluation for ICL proof
    print("\n--- Testing Shakespeare In-Context Learning Capabilities ---")
    x_test, y_test = get_batch(test_data, 100, cfg.SEQ_LEN, vocab_size, cfg.DEVICE)
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

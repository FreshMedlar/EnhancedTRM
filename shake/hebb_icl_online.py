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
    BATCH_SIZE: int = 1  # No batching
    X_DIM: int = 128     # Total input dimension across all layers (will be updated)
    LAYER_DIMS: List[int] = None # Will be [D_in, D_h1, D_out]
    
    # Evolution Params (Eggroll)
    POP_SIZE: int = 128  # Total population
    SIGMA: float = 0.02   # Noise standard deviation
    ES_LR: float = 0.01  # ES learning rate
    
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
    
    return data, vocab_size, stoi, itos

def get_sequential_token(data, idx, vocab_size, device):
    """Gets a single token and target."""
    x_indices = data[idx].unsqueeze(0).unsqueeze(0) # [1, 1]
    y_indices = data[idx+1].unsqueeze(0).unsqueeze(0) # [1, 1]
    
    # Convert to one-hot for ICL task
    x = F.one_hot(x_indices, num_classes=vocab_size).float().to(device)
    y = F.one_hot(y_indices, num_classes=vocab_size).float().to(device)
    
    return x, y

# --- 2. Training Loop ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=cfg.POP_SIZE)
    parser.add_argument("--lr", type=float, default=cfg.ES_LR)
    parser.add_argument("--sigma", type=float, default=cfg.SIGMA)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    
    cfg.POP_SIZE = args.pop_size
    cfg.ES_LR = args.lr
    cfg.SIGMA = args.sigma
    
    print(f"=== Shakespeare Online (Continuous Stream) Hebbian ICL ===")
    
    data, vocab_size, stoi, itos = prepare_data("shake/tinyshakespeare.txt")
    
    # Configure two blocks: vocab_size -> vocab_size -> vocab_size
    cfg.LAYER_DIMS = [vocab_size, vocab_size, vocab_size]
    # Total number of synapses across all layers
    cfg.X_DIM = sum(cfg.LAYER_DIMS[i] * cfg.LAYER_DIMS[i+1] for i in range(len(cfg.LAYER_DIMS)-1))
    
    print(f"Vocab Size: {vocab_size} | Layers: {cfg.LAYER_DIMS} | Total Synapses (X_DIM): {cfg.X_DIM}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Dataset Size: {len(data)} characters")
    
    if args.wandb:
        wandb.init(project="In-Context-Learning", name="Shakespeare-Online-Hebb-ICL", config=vars(cfg))
    else:
        wandb.init(mode="disabled")
    
    model = HebbianICLModel(cfg)
    
    POP_SIZE = cfg.POP_SIZE // 2
    noise_gen = Rank1NoiseGenerator(POP_SIZE, cfg.X_DIM, cfg.NUM_HEBB_COEFFS, cfg.DEVICE, cfg.SIGMA)
    fitness_shaper = FitnessShaper()
    updater = EGGROLLUpdater(lr=cfg.ES_LR)
    
    states = None
    
    # Process the entire dataset token by token in one continuous stream
    num_tokens = len(data) - 1
    
    for i in range(num_tokens):
        start_time = time.time()
        
        x, y = get_sequential_token(data, i, vocab_size, cfg.DEVICE)
        
        # Sample Rank-1 Noise
        A, B = noise_gen.sample()
        noise = torch.einsum('mi,mj->mij', A, B) * cfg.SIGMA
        
        W_curr = model.coeffs.squeeze(0)
        pop_coeffs = torch.cat([W_curr.unsqueeze(0) + noise, W_curr.unsqueeze(0) - noise], dim=0)
        
        # Evaluate population on the single token transition
        # We pass the persistent states and update them
        fitness, per_pos_loss_pop, states = model.evaluate_population(x, y, pop_coeffs, states=states)
        
        shaped_f_all = fitness_shaper.rank_transform(fitness)
        shaped_fitness = (shaped_f_all[:POP_SIZE] - shaped_f_all[POP_SIZE:]) / 2.0
        
        delta_W = updater.compute_update(shaped_fitness, A=A, B=B)
        norm = torch.norm(delta_W)
        if norm > 1.0: delta_W = delta_W / norm
            
        model.coeffs = (W_curr + cfg.ES_LR * delta_W).unsqueeze(0)
        
        current_loss = -fitness.mean().item()
        
        if i % 100 == 0:
            wandb.log({"loss": current_loss, "fitness_max": fitness.max().item(), "char_idx": i}, step=i)
            print(f"Token {i:07d}/{num_tokens} | Loss: {current_loss:.4f} | Max Fit: {fitness.max().item():.4f} | Time: {time.time() - start_time:.4f}s")
        
    print("\nTraining Complete.")
    wandb.finish()

if __name__ == "__main__":
    train()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
import math
from typing import Tuple, List, Dict
from esn import SparseESN
import os

# --- Configuration ---
BATCH_SIZE = 32
SEQ_LEN = 16
HIDDEN_SIZE = 2048
DENSITY = 0.1
POPULATION_SIZE = 128
SIGMA = 0.08
LEARNING_RATE = 0.001
STEPS = 10000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Loading ---
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

# --- Evolution Components ---

class Rank1NoiseGenerator:
    def __init__(self, output_size, input_size, device="cpu"):
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        
    def sample(self, pop_size) -> Tuple[torch.Tensor, torch.Tensor]:
        M = pop_size // 2
        A = torch.randn(M, self.output_size, device=self.device)
        B = torch.randn(M, self.input_size, device=self.device)
        return A, B

class ScalarNoiseGenerator:
    def __init__(self, device="cpu"):
        self.device = device
        
    def sample(self, pop_size) -> torch.Tensor:
        M = pop_size // 2
        return torch.randn(M, device=self.device)

class AdamOptimizer:
    def __init__(self, params_shape: List[int], lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = torch.zeros(params_shape, device=DEVICE)
        self.v = torch.zeros(params_shape, device=DEVICE)
        self.step = 0
        
    def update(self, param, grad):
        self.step += 1
        
        self.m = self.betas[0] * self.m + (1 - self.betas[0]) * grad
        self.v = self.betas[1] * self.v + (1 - self.betas[1]) * (grad ** 2)
        
        m_hat = self.m / (1 - self.betas[0] ** self.step)
        v_hat = self.v / (1 - self.betas[1] ** self.step)
        
        param_new = param + self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
        return param_new

def compute_fitness_vectorized_optimized(model, 
                                         W_rec_base,
                                         alpha_base,
                                         noise_rec,
                                         noise_alpha, 
                                         sigma, data_x, data_y):
    
    A_rec, B_rec = noise_rec # Shapes: [M, Hidden], [M, Hidden]
    epsilon_alpha = noise_alpha # Shape: [M]

    # Combine +/- perturbations into a single tensor for vectorized processing
    # A -> [2M, H], B -> [2M, H] (Positive then Negative blocks)
    A_rec_all = torch.cat([A_rec, -A_rec], dim=0) 
    B_rec_all = torch.cat([B_rec, B_rec], dim=0)   # B is shared for +/- in A(B^T) or we flip A

    # Prepare Alpha
    E_alpha_all = torch.cat([epsilon_alpha, -epsilon_alpha], dim=0).view(-1, 1, 1)
    alpha_all = alpha_base + sigma * E_alpha_all
    
    # Constants
    pop_size = A_rec_all.shape[0]
    batch_size = data_x.shape[0] # BATCH_SIZE
    
    # Initial State: [Pop, Batch, Hidden]
    h = torch.zeros(pop_size, batch_size, model.hidden_size, device=DEVICE)
    loss_all = torch.zeros(pop_size, device=DEVICE)

    # Pre-calculate Base Masks (Optimization: Don't re-prune every member)
    # We use the mask from the current 'mean' model for efficiency
    mask_rec = (model.W_rec.data != 0).float()
    
    # Pre-process Inputs
    # x_oh: [Batch, Seq, Vocab] -> Permute to [Seq, Batch, Vocab]
    x_oh = F.one_hot(data_x, model.input_size).float()
    x_seq = x_oh.permute(1, 0, 2)
    y_seq = data_y.permute(1, 0) # [Seq, Batch]

    for t in range(SEQ_LEN):
        xt = x_seq[t] # [Batch, Vocab]
        
        
        # --- 1. Input Injection ---
        # "Simple use some neurons of the reservoir as input neurons"
        # "Insert the input as one hot encoded in the reservoir"
        # xt is [Batch, Vocab]. We assume Vocab == Input Size.
        # h is [Pop, Batch, Hidden].
        # We overwrite the first `model.input_size` neurons of `h` with `xt`.
        h[:, :, :model.input_size] = xt.unsqueeze(0)

        # There is NO input projection step anymore. Input is IN the state.

        # --- 2. Recurrent Calculation (Low-Rank Optimized) ---
        # Base: h @ W_rec.T
        # h: [Pop, Batch, H], W_rec: [H, H]
        # Reshape h for linear: [Pop*Batch, H]
        h_flat = h.view(-1, model.hidden_size)
        rec_base = F.linear(h_flat, W_rec_base * mask_rec) # [Pop*Batch, H]
        rec_base = rec_base.view(pop_size, batch_size, model.hidden_size)
        
        # Perturbation: A(B^T h)
        # We need dot product of h[p] and B[p] for every p.
        # h: [Pop, Batch, H], B_rec_all: [Pop, H]
        # (h * B_rec_all.unsqueeze(1)).sum(-1) -> [Pop, Batch] (Dot product)
        delta_rec_projected = (h * B_rec_all.unsqueeze(1)).sum(dim=2, keepdim=True) # [Pop, Batch, 1]
        
        # Scale A: [Pop, Batch, 1] * [Pop, 1, H] -> [Pop, Batch, H]
        rec_noise = delta_rec_projected * A_rec_all.unsqueeze(1)
        
        rec_act = rec_base + sigma * rec_noise
        
        
        # --- 3. Update State ---
        h_update = F.leaky_relu(rec_act, negative_slope=0.01)
        h = (1 - alpha_all) * h + alpha_all * h_update
        
        # --- 4. Loss ---
        logits = h[:, :, :model.output_size] # [Pop, Batch, Out]
        yt = y_seq[t] # [Batch]
        
        # Efficient Loss Calc
        # Reshape for CrossEntropy: [Pop*Batch, Out]
        flat_logits = logits.reshape(-1, model.output_size)
        flat_targets = yt.repeat(pop_size)
        
        loss_t = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        loss_t = loss_t.view(pop_size, batch_size).mean(dim=1)
        loss_all += loss_t

    return -(loss_all / SEQ_LEN)



def rank_transformation(fitness):
    N = len(fitness)
    indices = torch.argsort(fitness)
    ranks = torch.zeros_like(fitness)
    ranks[indices] = torch.arange(N, device=fitness.device, dtype=torch.float32)
    return (ranks / (N - 1)) - 0.5

def main():
    wandb.init(project="evo-reservoir", name="vectorized-evo")
    
    # Checkpoint dir
    os.makedirs("checkpoints", exist_ok=True)
    
    data_path = "tinyshakespeare.txt"
    if not os.path.exists(data_path):
        data_path = "tinyshakespeare.txt" 
    
    raw_data, vocab_size, stoi, itos = load_data(data_path)
    n_train = int(len(raw_data) * 0.9)
    train_data = raw_data[:n_train]
    test_data = raw_data[n_train:]
    
    print(f"Vocab: {vocab_size}, Train: {len(train_data)}, Test: {len(test_data)}")
    
    model = SparseESN(vocab_size, HIDDEN_SIZE, vocab_size, density=DENSITY).to(DEVICE)
    
    # Init optimizers
    noise_rec = Rank1NoiseGenerator(HIDDEN_SIZE, HIDDEN_SIZE, device=DEVICE)
    noise_alpha = ScalarNoiseGenerator(device=DEVICE)
    
    opt_rec = AdamOptimizer(model.W_rec.shape, lr=LEARNING_RATE)
    opt_alpha = AdamOptimizer(model.leaky_alpha.shape, lr=LEARNING_RATE)
    
    print("Starting Vectorized Evolution...")
    
    for step in range(STEPS):
        x, y = get_batch(train_data, BATCH_SIZE, SEQ_LEN, DEVICE)
        
        # 1. Sample Noise
        A_rec, B_rec = noise_rec.sample(POPULATION_SIZE)
        E_alpha = noise_alpha.sample(POPULATION_SIZE)
        
        # 2. Evaluate (Vectorized)
        fitness = compute_fitness_vectorized_optimized(
            model, 
            model.W_rec.data, model.leaky_alpha.data,
            (A_rec, B_rec), E_alpha,
            SIGMA, x, y
        )
        
        # 3. Shape Fitness
        shaped_fit = rank_transformation(fitness)
        M = POPULATION_SIZE // 2
        diff_fit = (shaped_fit[:M] - shaped_fit[M:]) / 2.0
        
        # 4. Updates
        M = POPULATION_SIZE // 2
        
        # --- W_rec ---
        # Vectorized gradient: sum(diff_fit[i] * A[i] @ B[i].T)
        grad_rec = torch.einsum('m,mh,mk->hk', diff_fit, A_rec, B_rec)
        grad_rec /= (M * SIGMA)
        
        W_rec_new = opt_rec.update(model.W_rec.data, grad_rec)
        
        # Prune
        k_rec = int(HIDDEN_SIZE * HIDDEN_SIZE * DENSITY)
        flat = W_rec_new.view(-1)
        thr = torch.topk(torch.abs(flat), k_rec).values.min()
        mask = torch.abs(W_rec_new) >= thr
        W_rec_new = W_rec_new * mask.float()
        
        model.W_rec.data = W_rec_new
        if step % 50 == 0:
             rho = model.enforce_spectral_radius(0.95)
        else:
             rho = 0.0 # Skip for speed
        
        # --- Alpha ---
        grad_alpha = (diff_fit * E_alpha).sum() / (M * SIGMA)
        
        alpha_new = opt_alpha.update(model.leaky_alpha.data, grad_alpha)
        model.leaky_alpha.data = torch.clamp(alpha_new, 0.01, 0.99)
        
        # Logging
        loss = -fitness.max().item()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss {loss:.4f} | Alpha {model.leaky_alpha.item():.3f}")
            wandb.log({
                "train/loss": loss, 
                "train/alpha": model.leaky_alpha.item()
            })
            
        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                xt, yt = get_batch(test_data, BATCH_SIZE, SEQ_LEN, DEVICE)
                h = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, device=DEVICE)
                xt_oh = F.one_hot(xt, vocab_size).float()
                loss_test = 0
                
                # Non-vectorized forward for testing using shared model weights
                
                for t in range(SEQ_LEN):
                    # Using model() calls esn.py forward
                    h = model(xt_oh[:, t, :], h)
                    logits = model.get_output(h)
                    loss_test += F.cross_entropy(logits, yt[:, t]).item()
                loss_test /= SEQ_LEN
                print(f"Test Loss: {loss_test:.4f}")
                wandb.log({"test/loss": loss_test})

        if step > 0 and step % 1000 == 0:
            ckpt_path = f"checkpoints/model_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_rec_state_dict': opt_rec.m, # Saving opt state as well
                'optimizer_alpha_state_dict': opt_alpha.m,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    wandb.finish()

if __name__ == "__main__":
    main()

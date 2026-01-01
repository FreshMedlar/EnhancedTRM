import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import math
import numpy as np
from eggroll.noise import NoiseConfig, Rank1NoiseGenerator
from eggroll.update import UpdateConfig, EGGROLLUpdater

class ShakespeareReservoir(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Trainable weights (initialized normally for reservoir)
        self.W_in = nn.Parameter(torch.randn(hidden_size, input_size) * 0.1)
        self.W_res = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1 / math.sqrt(hidden_size))
        self.W_out = nn.Parameter(torch.zeros(output_size, hidden_size))
        
        self.tanh = nn.Tanh()

    def forward(self, x, h_0=None, W_in_p=None, W_res_p=None, W_out_p=None):
        """
        x: [batch, seq_len, input_size] (one-hot)
        h_0: [batch, hidden_size]
        *_p: optional perturbed weights for Eggroll population evaluation
        """
        # Use provided weights or defaults
        win = W_in_p if W_in_p is not None else self.W_in
        wres = W_res_p if W_res_p is not None else self.W_res
        wout = W_out_p if W_out_p is not None else self.W_out

        batch_size, seq_len, _ = x.shape
        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_0

        states = []
        for t in range(seq_len):
            u_t = x[:, t, :]
            # h = tanh(W_in @ u_t + W_res @ h)
            h = self.tanh(F.linear(u_t, win) + F.linear(h, wres))
            states.append(h)
        
        states = torch.stack(states, dim=1) # [B, T, H]
        logits = F.linear(states, wout)     # [B, T, O]
        return logits, h

def load_shakespeare(path, train_ratio=0.9):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chars = sorted(list(set(text)))
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    
    data = torch.tensor([char_to_ix[ch] for ch in text], dtype=torch.long)
    n = int(train_ratio * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    return train_data, val_data, vocab_size, char_to_ix, ix_to_char

def get_batch(data, batch_size, seq_len, vocab_size, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x_indices = torch.stack([data[i:i+seq_len] for i in ix])
    y_indices = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    
    # One-hot encode x
    x = F.one_hot(x_indices, num_classes=vocab_size).float().to(device)
    y = y_indices.to(device)
    return x, y

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # Load data
    train_data, val_data, vocab_size, char_to_ix, ix_to_char = load_shakespeare(
        "tinyshakespeare.txt", train_ratio=0.9
    )
    print(f"Vocab size: {vocab_size}")

    # Hyperparameters
    hidden_size = args.hidden_size
    seq_len = args.seq_len
    batch_size = args.batch_size
    iterations = args.iterations
    
    # Model
    model = ShakespeareReservoir(vocab_size, hidden_size, vocab_size).to(device)
    
    # Weight matrices to train
    W_in = model.W_in.data.clone()
    W_res = model.W_res.data.clone()
    W_out = model.W_out.data.clone()

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
    
    # Separate generators and updaters for each matrix
    noise_gen_in = Rank1NoiseGenerator(noise_config, hidden_size, vocab_size, device=str(device))
    noise_gen_res = Rank1NoiseGenerator(noise_config, hidden_size, hidden_size, device=str(device))
    noise_gen_out = Rank1NoiseGenerator(noise_config, vocab_size, hidden_size, device=str(device))
    
    updater_in = EGGROLLUpdater(update_config, hidden_size, vocab_size, device=str(device))
    updater_res = EGGROLLUpdater(update_config, hidden_size, hidden_size, device=str(device))
    updater_out = EGGROLLUpdater(update_config, vocab_size, hidden_size, device=str(device))
    
    print("Starting training...")
    start_time = time.time()
    sigma = args.sigma
    
    for i in range(iterations):
        # 1. Get Data
        x, target = get_batch(train_data, batch_size, seq_len, vocab_size, device)
        
        with torch.no_grad():
            # 2. Sample Perturbations
            A_in, B_in = noise_gen_in.sample(step=i)
            A_res, B_res = noise_gen_res.sample(step=i)
            A_out, B_out = noise_gen_out.sample(step=i)
            M = A_in.shape[0]
            
            # 3. Evaluate Population
            # We need to compute fitness for POS and NEG directions.
            # For each direction j in [0, M-1]:
            # W_pos = W + sigma * A[j] @ B[j].T
            # W_neg = W - sigma * A[j] @ B[j].T
            
            # Since we have 3 matrices, each "individual" j has 3 perturbations.
            # To be efficient, we can evaluate them sequentially or find a way to vectorize.
            # Vectorizing RNNs over populations is tricky. 
            # Let's do a simple loop for now if M is small, or just evaluate the base first.
            
            # Actually, let's just do the base evaluation for logging
            logits_base, _ = model(x, W_in_p=W_in, W_res_p=W_res, W_out_p=W_out)
            loss_base = F.cross_entropy(logits_base.reshape(-1, vocab_size), target.reshape(-1))
            
            # Efficient population evaluation for Eggroll in RNNs is hard without custom kernels.
            # However, we can use the Rank-1 property:
            # h_t = tanh( (W_in + sigma a_in b_in^T) u_t + (W_res + sigma a_res b_res^T) h_{t-1} )
            #     = tanh( W_in u_t + sigma a_in (b_in^T u_t) + W_res h_{t-1} + sigma a_res (b_res^T h_{t-1}) )
            
            # Let's implement a vectorized population forward pass.
            # Population size M directions -> 2*M individuals (antithetic).
            # But that might explode memory. Let's stick to a manageable M.
            
            def eval_population(x, target, W_in, W_res, W_out, A_in, B_in, A_res, B_res, A_out, B_out, sigma, sign=1.0):
                # x: [B, T, V]
                # target: [B, T]
                B_size, T_size, V_size = x.shape
                H_size = W_in.shape[0]
                M = A_in.shape[0]
                
                # h: [M, B, H]
                h = torch.zeros(M, B_size, H_size, device=device)
                all_logits = []
                
                for t in range(T_size):
                    u_t = x[:, t, :] # [B, V]
                    
                    # W_in part: [B, H]
                    base_in = F.linear(u_t, W_in) # [B, H]
                    # perturbation in: sigma * sign * (u_t @ B_in^T) @ A_in^T
                    # (u_t @ B_in^T): [B, M]
                    # result: [M, B, H]
                    pert_in = sigma * sign * torch.einsum('bm,mh->mbh', (u_t @ B_in.T), A_in)
                    
                    # W_res part: [M, B, H]
                    base_res = F.linear(h, W_res) # [M, B, H]
                    # perturbation res: sigma * sign * (h @ B_res^T) @ A_res^T
                    # h: [M, B, H], B_res: [M, H] -> we want for each m: (h[m] @ B_res[m]) * A_res[m]
                    # (h[m] @ B_res[m]): [B]
                    # result: [M, B, H]
                    h_dot_b = torch.einsum('mbh,mh->mb', h, B_res) # [M, B]
                    pert_res = sigma * sign * torch.einsum('mb,mh->mbh', h_dot_b, A_res)
                    
                    h = torch.tanh(base_in.unsqueeze(0) + pert_in + base_res + pert_res)
                
                # After sequences, compute logits for the whole population
                # h: [M, B, H] - wait, we need all states for all time steps for cross entropy?
                # Actually, usually we just predict the last character or all? Shakespeare is usually all.
                # If we need all states, memory will be an issue. 
                # Let's just use the final state for simplicity to see if it trains, 
                # OR we only compute loss on the last character of the sequence.
                # For TinyShakespeare, predicting the whole sequence is better.
                # Let's just compute the loss for the LAST character to save memory for now, 
                # or keep a few if M is small.
                
                # W_out part:
                # logits = h @ W_out.T + sigma * sign * (h @ B_out.T) @ A_out^T
                h_dot_bout = torch.einsum('mbh,mh->mb', h, B_out)
                logits = F.linear(h, W_out) + sigma * sign * torch.einsum('mb,mo->mbo', h_dot_bout, A_out)
                
                # loss per individual: [M]
                target_last = target[:, -1] # [B]
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_last.repeat(M), reduction='none')
                loss = loss.reshape(M, B_size).mean(dim=1)
                return -loss # fitness
            
            # Evaluate pos/neg
            # Since we only use the last char, let's see. 
            # If we want to use the whole sequence, we'd need to store 'h' for all 't'.
            
            fitness_pos = eval_population(x, target, W_in, W_res, W_out, A_in, B_in, A_res, B_res, A_out, B_out, sigma, 1.0)
            fitness_neg = eval_population(x, target, W_in, W_res, W_out, A_in, B_in, A_res, B_res, A_out, B_out, sigma, -1.0)
            
            # Combine for update
            raw_diff = (fitness_pos - fitness_neg) / (2 * sigma)
            
            # 5. Update Weights
            W_in = updater_in.apply_update(W_in, A_in, B_in, raw_diff, sigma)
            W_res = updater_res.apply_update(W_res, A_res, B_res, raw_diff, sigma)
            W_out = updater_out.apply_update(W_out, A_out, B_out, raw_diff, sigma)
            
            # Sync model weights for next base eval/inference
            model.W_in.data.copy_(W_in)
            model.W_res.data.copy_(W_res)
            model.W_out.data.copy_(W_out)
            
            # Logging and Validation
            if i % args.log_interval == 0:
                # Validation Loss (Full sequence evaluation)
                x_val, target_val = get_batch(val_data, batch_size, seq_len, vocab_size, device)
                logits_val, _ = model(x_val)
                loss_val = F.cross_entropy(logits_val.reshape(-1, vocab_size), target_val.reshape(-1))
                
                preds = torch.argmax(logits_val, dim=-1)
                acc = (preds == target_val).float().mean()
                
                print(f"Iter {i:4d} | Loss: {loss_base:.4f} | Val Loss: {loss_val:.4f} | Val Acc: {acc:.4f}")

    print(f"Final Val Loss: {loss_val:.4f}")
    print(f"Total Time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=0.02)
    parser.add_argument("--pop_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    
    train(args)

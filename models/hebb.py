import torch
from dataclasses import dataclass
from typing import Tuple

class HebbianICLModel:
    """
    Handles a population of single-layer Hebbian networks.
    Each network starts with random weights and updates via evolved Hebbian rules.
    """
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        # Apply any overrides
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        
        # Per-synapse coefficients: (1, X_DIM, NUM_HEBB_COEFFS)
        self.coeffs = torch.zeros(1, self.cfg.X_DIM, self.cfg.NUM_HEBB_COEFFS, device=self.cfg.DEVICE)
        
    def evaluate_population(self, x_seq, y_seq, pop_coeffs):
        """
        pop_coeffs: [P, X_DIM, NUM_HEBB_COEFFS]
        x_seq: [B, L, X_DIM]
        y_seq: [B, L, 1]
        """
        P = pop_coeffs.shape[0]
        B, L, Dx = x_seq.shape
        device = self.cfg.DEVICE
        
        # Reshape coeffs for broadcasting: [P, 1, Dx, NUM_HEBB_COEFFS]
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

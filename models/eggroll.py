import torch
from dataclasses import dataclass
from typing import Tuple

@dataclass
class NoiseConfig:
    population_size: int = 128
    sigma: float = 0.02
    seed: int = 42

class Rank1NoiseGenerator:
    """Generates rank-1 perturbations A and B."""
    def __init__(
        self, 
        config: NoiseConfig, 
        output_size: int,
        input_size: int,
        device: str = "cpu"
    ):
        self.config = config
        self.output_size = output_size
        self.input_size = input_size
        self.device = device
        self.num_directions = config.population_size // 2
        
    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns A: [M, D_out], B: [M, D_in]"""
        M = self.num_directions
        # Note: In a real implementation, you might want to use a fixed generator/seed per step
        A = torch.randn(M, self.output_size, device=self.device)
        B = torch.randn(M, self.input_size, device=self.device)
        return A, B

class FitnessShaper:
    def rank_transform(self, fitness: torch.Tensor) -> torch.Tensor:
        """Transform fitness to ranks in [-0.5, 0.5]."""
        N = fitness.shape[0]
        _, indices = fitness.sort()
        ranks = torch.zeros_like(fitness)
        ranks[indices] = torch.arange(N, dtype=fitness.dtype, device=fitness.device)
        return (ranks / (N - 1)) - 0.5

class AntitheticShaper:
    """Computes (f_pos - f_neg) / 2."""
    def __call__(self, fitness: torch.Tensor) -> torch.Tensor:
        # fitness: [2M] -> returns [M]
        M = fitness.shape[0] // 2
        return (fitness[:M] - fitness[M:]) / 2

@dataclass
class UpdateConfig:
    learning_rate: float = 0.05
    clip_norm: float = 1.0
    weight_decay: float = 1e-4

class EGGROLLUpdater:
    def __init__(self, config: UpdateConfig):
        self.config = config
        
    def compute_update(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        shaped_fitness: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute delta_W = (1/M) * sum(f_j * (A_j @ B_j^T))
        Efficiently: (A * f)^T @ B
        """
        M = A.shape[0]
        # shaped_fitness: [M]
        weighted_A = A * shaped_fitness.unsqueeze(-1) # [M, D_out]
        delta_W = weighted_A.T @ B # [D_out, D_in]
        return delta_W / M
    
    def apply_update(
        self,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        shaped_fitness: torch.Tensor
    ) -> torch.Tensor:
        
        delta_W = self.compute_update(A, B, shaped_fitness)
        
        # Gradient Clipping
        norm = torch.norm(delta_W, p='fro')
        if norm > self.config.clip_norm:
            delta_W = delta_W * (self.config.clip_norm / norm)
            
        # Update W
        W_new = W + self.config.learning_rate * delta_W
        
        # Weight Decay
        if self.config.weight_decay > 0:
            W_new = W_new * (1 - self.config.learning_rate * self.config.weight_decay)
            
        return W_new

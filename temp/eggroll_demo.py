
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
import math
import time

"""
EGGROLL ALGORITHM: Single-File Implementation
=============================================

What is EGGROLL?
----------------
EGGROLL (Efficient Gradient-Free Generalized Rank-One Low-latency Learning) is an
Evolution Strategy (ES) designed for optimizing embedding models efficiently.
Standard ES requires computing scores for 'PopSize' perturbed weights, which is expensive.
EGGROLL uses rank-1 perturbations: W_new = W + A @ B.T
This allows computing scores for ALL perturbations in a single vectorized pass
without ever explicitly forming the large perturbed weight matrices.

Key Components:
1. VectorizedScorer:
   Uses the algebraic expansion of (q(W+AB^T) . d(W+AB^T)) to compute scores for
   the entire population in one go. Complexity reduces from O(PopSize * D^2) to
   matrix multiplications.

2. Rank-1 Noise:
   Perturbations are generated as outer products of smaller vectors A and B.

3. Antithetic Sampling:
   For every perturbation P, we also evaluate -P. This reduces gradient variance.

4. Fitness Shaping (NDCG):
   The optimization objective is often ranking performance (NDCG). We compute NDCG
   for every perturbation, rank the results to remove outliers, and use these
   ranks to weight the update.

5. Update Rule:
   W <- W + lr * (1/M) * sum(fitness_i * A_i @ B_i^T)
   This approximates the natural gradient of the expected fitness.

"""

# ==============================================================================
# 1. SCORING (Vectorized Score Computation)
# ==============================================================================

class VectorizedScorer:
    """
    Computes scores for rank-1 perturbations efficiently.
    For W_i = W + sigma * A_i @ B_i^T, computes scores for all i in one pass.
    """
    def __init__(self, sigma: float = 0.02):
        self.sigma = sigma
        
    def compute_base_scores(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        W: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute unperturbed embeddings and scores.
        Args:
            H_q: [B, D_in] query embeddings
            H_d: [B, P, D_in] document embeddings
            W: [D_out, D_in] projection matrix
        Returns:
            q_base: [B, D_out]
            d_base: [B, P, D_out]
            scores_base: [B, P]
        """
        q_base = H_q @ W.T
        d_base = torch.einsum('bpd,ed->bpe', H_d, W)
        scores_base = torch.einsum('bd,bpd->bp', q_base, d_base)
        return q_base, d_base, scores_base
    
    def compute_perturbed_scores_rank1(
        self,
        H_q: torch.Tensor,
        H_d: torch.Tensor,
        q_base: torch.Tensor,
        d_base: torch.Tensor,
        scores_base: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores for all perturbations without materializing E.
        Math: score_i = base + sigma*(s_q*ad + s_d*aq) + sigma^2*s_q*s_d*a2
        Returns: [B, P, N] where N = 2*M (antithetic pairs)
        """
        sigma = self.sigma
        M = A.shape[0]
        
        # Scalar products with direction vectors
        s_q = H_q @ B.T  # [B, M]
        s_d = torch.einsum('bpd,md->bpm', H_d, B)  # [B, P, M]
        
        # Projections onto perturbation output directions
        aq = q_base @ A.T  # [B, M]
        ad = torch.einsum('bpd,md->bpm', d_base, A)  # [B, P, M]
        
        # Norm squared of A rows
        a2 = (A * A).sum(dim=-1)  # [M]
        
        # Expansion terms
        # linear_term: [B, P, M]
        linear_term = s_q.unsqueeze(1) * ad + s_d * aq.unsqueeze(1)
        # quad_term: [B, P, M]
        quad_term = s_q.unsqueeze(1) * s_d * a2.view(1, 1, M)
        
        # Compute for +sigma and -sigma (Antithetic)
        scores_pos = scores_base.unsqueeze(-1) + sigma * linear_term + sigma**2 * quad_term
        scores_neg = scores_base.unsqueeze(-1) - sigma * linear_term + sigma**2 * quad_term
        
        return torch.cat([scores_pos, scores_neg], dim=-1)

# ==============================================================================
# 2. NOISE GENERATION
# ==============================================================================

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

# ==============================================================================
# 3. METRIC (NDCG)
# ==============================================================================

class NDCGComputer:
    """Computes NDCG@k for batched scores."""
    def __init__(self, k: int = 10, device: str = "cpu"):
        self.k = k
        self.device = device
        positions = torch.arange(k, device=device, dtype=torch.float32)
        self.discounts = 1.0 / torch.log2(positions + 2)
        
    def compute_dcg(self, relevance: torch.Tensor) -> torch.Tensor:
        gains = (2.0 ** relevance) - 1.0
        # If relevance has fewer than k items, slice discounts
        curr_k = min(self.k, relevance.shape[-1])
        return (gains[..., :curr_k] * self.discounts[:curr_k]).sum(dim=-1)
    
    def compute_idcg(self, relevance: torch.Tensor) -> torch.Tensor:
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        return self.compute_dcg(sorted_rel)
    
    def compute_ndcg(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor
    ) -> torch.Tensor:
        """
        scores: [B, P, N] (batch, population, perturbations)
          OR    [B, P]    (batch, population) - for base scores
        relevance: [B, P]
        
        Returns:
             Mean NDCG across batch.
             If scores is [B, P, N], returns [N]
             If scores is [B, P], returns scalar
        """
        is_3d = (scores.ndim == 3)
        if not is_3d:
            scores = scores.unsqueeze(-1) # [B, P, 1]
            
        B, P, N = scores.shape
        k = min(self.k, P)
        
        # Get top-k indices for each perturbation
        _, topk_indices = scores.topk(k, dim=1) # [B, k, N]
        
        # Gather relevance for these top-k items
        # relevance: [B, P] -> [B, P, N]
        rel_expanded = relevance.unsqueeze(-1).expand(-1, -1, N)
        topk_rel = torch.gather(rel_expanded, dim=1, index=topk_indices) # [B, k, N]
        
        # Compute DCG
        gains = (2.0 ** topk_rel.float()) - 1.0
        dcg = (gains * self.discounts[:k].view(1, k, 1)).sum(dim=1) # [B, N]
        
        # Compute IDCG (ideal DCG based on ground truth)
        idcg = self.compute_idcg(relevance) # [B]
        idcg = idcg.unsqueeze(-1).clamp(min=1e-10) # [B, 1]
        
        ndcg = dcg / idcg
        ndcg_mean = ndcg.mean(dim=0) # [N]
        
        if not is_3d:
            return ndcg_mean.squeeze()
        return ndcg_mean

# ==============================================================================
# 4. FITNESS SHAPING
# ==============================================================================

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

# ==============================================================================
# 5. UPDATE (Optimization)
# ==============================================================================

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

# ==============================================================================
# 6. MAIN EXAMPLE
# ==============================================================================

def main():
    print("=== Eggroll Algorithm Single-File Demo ===")
    
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    D_IN = 32
    D_OUT = 16
    BATCH_SIZE = 64   # Queries per batch
    POOL_SIZE = 20    # Documents per query
    
    STEPS = 500
    
    noise_cfg = NoiseConfig(population_size=64, sigma=0.05)
    update_cfg = UpdateConfig(learning_rate=0.1, weight_decay=1e-3)
    
    # Initialize Components
    scorer = VectorizedScorer(sigma=noise_cfg.sigma)
    noise_gen = Rank1NoiseGenerator(noise_cfg, D_OUT, D_IN, device=DEVICE)
    ndcg_computer = NDCGComputer(k=5, device=DEVICE)
    fitness_shaper = FitnessShaper()
    antithetic_shaper = AntitheticShaper()
    updater = EGGROLLUpdater(update_cfg)
    
    # --- Synthetic Data Generation ---
    # We create a random "ground truth" projection W_true
    # The goal is to learn W that performs as well as W_true on the ranking task
    torch.manual_seed(42)
    W_true = torch.randn(D_OUT, D_IN, device=DEVICE)
    W_true = F.normalize(W_true, dim=1)
    
    # Initial learned weights (random)
    W_learned = torch.randn(D_OUT, D_IN, device=DEVICE) * 0.1
    
    print("\nStarting Training Loop...")
    start_time = time.time()
    
    for step in range(STEPS):
        # 1. Sample Data Batch
        # Generate random query and document embeddings
        H_q = torch.randn(BATCH_SIZE, D_IN, device=DEVICE)
        H_d = torch.randn(BATCH_SIZE, POOL_SIZE, D_IN, device=DEVICE)
        
        # Compute Ground Truth Scores & Relevance
        with torch.no_grad():
            q_gt = H_q @ W_true.T
            d_gt = torch.einsum('bpd,ed->bpe', H_d, W_true)
            scores_gt = torch.einsum('bd,bpd->bp', q_gt, d_gt)
            
            # Create relevance: Top 3 docs from GT are relevant (1), others 0
            # This creates a hard target to approximate
            _, indices = scores_gt.topk(3, dim=1)
            relevance = torch.zeros_like(scores_gt)
            relevance.scatter_(1, indices, 1.0)
        
        # 2. Sample Perturbations
        A, B = noise_gen.sample()
        
        # 3. Compute Base Scores (Current Model)
        q_base, d_base, scores_base = scorer.compute_base_scores(H_q, H_d, W_learned)
        
        # 4. Compute Perturbed Scores
        scores_perturbed = scorer.compute_perturbed_scores_rank1(
            H_q, H_d, q_base, d_base, scores_base, A, B
        )
        
        # 5. Compute Fitness (NDCG)
        # Base Performance
        base_ndcg = ndcg_computer.compute_ndcg(scores_base, relevance)
        # Perturbed Performance
        perturbed_ndcg = ndcg_computer.compute_ndcg(scores_perturbed, relevance)
        
        # 6. Shape Fitness
        #   a. Rank Transform (for robustness)
        fitness_ranks = fitness_shaper.rank_transform(perturbed_ndcg)
        #   b. Antithetic Combination
        shaped_fitness = antithetic_shaper(fitness_ranks)
        
        # 7. Update Weights
        W_learned = updater.apply_update(W_learned, A, B, shaped_fitness)
        
        # Logging
        if step % 20 == 0:
            print(f"Step {step:03d} | Base NDCG: {base_ndcg.item():.4f} | "
                  f"Best Perturbation: {perturbed_ndcg.max().item():.4f}")
            
    print(f"\nTraining Complete in {time.time() - start_time:.2f}s")
    print(f"Final NDCG: {base_ndcg.item():.4f}")

if __name__ == "__main__":
    main()

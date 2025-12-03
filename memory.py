from typing_extensions import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ModernHopfieldNetwork(nn.Module):
    def __init__(self, beta: float = 10.0):
        """
        A Modern Hopfield Network (Continuous/Dense Associative Memory).
        
        Args:
            beta (float): Inverse temperature parameter. 
                          High beta -> acts like hard max (retrieves single closest pattern).
                          Low beta -> acts like average (retrieves metastable mixture).
        """
        super().__init__()
        self.beta = beta
        # We register patterns as a buffer so they move to device with the model
        # but are not treated as learnable parameters by an optimizer by default.
        self.register_buffer("stored_patterns", None)

    def one_shot_learn(self, patterns: torch.Tensor):
        """
        Instantly learns patterns by storing them in the memory matrix.
        
        Args:
            patterns (Tensor): Shape (num_patterns, feature_dim)
        """
        # Normalize patterns to ensure stability in dot products
        # Modern Hopfield networks usually assume patterns lie on a sphere
        norm_patterns = F.normalize(patterns, p=2, dim=1)
        self.stored_patterns = norm_patterns
        print(f"Stored {patterns.shape[0]} patterns of dimension {patterns.shape[1]}.")

    def energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the energy of the current state.
        The network dynamics minimize this energy function.
        
        E = - (1/beta) * lse(beta * state @ stored_patterns.T)
        """
        if self.stored_patterns is None:
            raise RuntimeError("No patterns stored. Call one_shot_learn() first.")
            
        # lse = LogSumExp
        activations = self.beta * (state @ self.stored_patterns.T)
        return -(1.0 / self.beta) * torch.logsumexp(activations, dim=1)

    @override
    def forward(self, state: torch.Tensor, num_updates: int = 1) -> torch.Tensor:
        """
        Retrieves memory by updating the state to minimize energy.
        This update rule is mathematically equivalent to the Attention mechanism.
        
        Rule: xi_new = softmax(beta * state @ patterns.T) @ patterns
        
        Args:
            state (Tensor): The query/corrupted pattern. Shape (batch_size, feature_dim)
            num_updates (int): How many recurrent iterations to perform.
        
        Returns:
            Tensor: The reconstructed/retrieved pattern.
        """
        if self.stored_patterns is None:
            raise RuntimeError("No patterns stored. Call one_shot_learn() first.")

        x = F.normalize(state, p=2, dim=1)
        
        for _ in range(num_updates):
            # 1. Similarity: Dot product between state (Query) and memories (Keys)
            similarity = x @ self.stored_patterns.T # (batch, num_patterns)
            
            # 2. Weighting: Softmax over the similarities
            # High beta makes the attention mechanism sharper
            weights = F.softmax(self.beta * similarity, dim=1) 
            
            # 3. Update: Weighted sum of memories (Values)
            x = weights @ self.stored_patterns # (batch, feature_dim)
            
            # Re-normalize to keep state on the hypersphere
            x = F.normalize(x, p=2, dim=1)
            
        return x

# --- Example Usage ---

def run_demo():
    # Configuration
    dim = 64
    n_patterns = 3
    beta = 1.0 # High beta for strict retrieval
    
    model = ModernHopfieldNetwork(beta=beta)
    
    # 1. Create orthogonal-ish patterns (random vectors)
    torch.manual_seed(42)
    clean_patterns = torch.randn(n_patterns, dim)
    clean_patterns = F.normalize(clean_patterns, p=2, dim=1)
    
    # 2. One-Shot Learning
    # The model memorizes these instantly without backprop
    model.one_shot_learn(clean_patterns)
    
    # 3. Create a corrupted query
    # Take the first pattern and add significant noise
    target_pattern = clean_patterns[0:1] 
    noise = torch.randn_like(target_pattern) * 0.5
    corrupted_input = target_pattern + noise
    corrupted_input = F.normalize(corrupted_input, p=2, dim=1)
    
    # 4. Retrieve
    recovered_pattern = model(corrupted_input, num_updates=100)
    
    # 5. Verify Similarity (Cosine Similarity)
    # Cosine similarity between recovered and original target
    score = F.cosine_similarity(recovered_pattern, target_pattern)
    
    print(f"\nTarget Pattern Index: 0")
    print(f"Cosine Similarity (Corrupted vs Target): {F.cosine_similarity(corrupted_input, target_pattern).item():.4f}")
    print(f"Cosine Similarity (Recovered vs Target): {score.item():.4f}")
    
    # Visual check
    if score > 0.99:
        print("SUCCESS: Pattern perfectly retrieved.")
    else:
        print("FAILURE: Retrieval failed.")

if __name__ == "__main__":
    run_demo()




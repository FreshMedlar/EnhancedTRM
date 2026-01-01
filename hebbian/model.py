import torch
import torch.nn as nn
import torch.nn.functional as F

class HebbianReservoir(nn.Module):
    """
    A Reservoir Computing model where the recurrent weights are plastic and evolve
    during the lifetime of the agent using Hebbian learning rules.
    
    The 'genome' of this agent is NOT the weights W_rec, but the coefficients
    of the Hebbian plasticity rule (A, B, C, D, LearningRate).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        leaky_alpha: float = 0.1,
        spectral_radius: float = 0.95,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        
        # Leaky Rate (Evolvable Scalar)
        self.leaky_alpha = nn.Parameter(torch.tensor(float(leaky_alpha)))
        
        # W_rec - This is the "Plastic" weight matrix.
        # It is reset at the beginning of every lifetime (episode).
        # We register it as a buffer so it's part of the state dict, but it's not a parameter.
        self.register_buffer('W_rec', torch.zeros(hidden_size, hidden_size))
        
        # Hebbian Coefficients are NOT stored here directly if we are using Eggroll 
        # to generate them dynamically from a population. 
        # However, for inference/single instance usage, we can store a 'mean' set.
        # We'll expect them to be passed in or set externally during training.

    def reset_weights(self):
        """
        Resets W_rec to random values scaled by spectral radius.
        This must be called at the start of every 'lifetime'.
        """
        with torch.no_grad():
            # Random initialization
            W = torch.randn(self.hidden_size, self.hidden_size, device=self.W_rec.device)
            
            # Spectral Radius enforcement
            eigs = torch.linalg.eigvals(W)
            max_eig = torch.max(torch.abs(eigs))
            if max_eig > 0:
                W *= (self.spectral_radius / max_eig)
            
            self.W_rec.copy_(W)

    def forward(self, x, h):
        """
        Standard Reservior Forward Pass (with Input Injection).
        Does NOT perform Hebbian update (that is done explicitly).
        
        x: [Batch, InputSize] (Input vector/indices)
        h: [Batch, HiddenSize] (Current State)
        
        Returns: 
            h_new: [Batch, HiddenSize] (New State)
        """
        
        # 1. Input Injection
        # "The input will be simply one_hot encoded in the first 65 neurons"
        # We assume 'x' is already the correct shape [Batch, InputSize] or we construct it.
        # If x is indices [Batch], we one-hot encode.
        # Here we assume x is the feature vector to be injected.
        
        h_in = h.clone()
        # We overwrite the first `input_size` neurons with x
        # Note: This implies the reservoir state effectively "sees" x immediately
        h_in[:, :self.input_size] = x
        
        # 2. Recurrent Projection
        # W_rec is [Hidden, Hidden]
        # pre_activation = h_in @ W_rec.T
        pre_activation = F.linear(h_in, self.W_rec)
        
        # 3. Leaky Integration
        alpha = torch.clamp(self.leaky_alpha, 0.01, 0.99)
        update = torch.tanh(pre_activation)
        
        h_new = (1 - alpha) * h_in + alpha * update
        
        # Note: We return h_new. 
        # For Hebbian update, we need 'h_in' (pre) and 'h_new' (post) is one interpretation,
        # or 'h' (pre) and 'h_new' (post). 
        # Given the "Injection" strategy, 'h_in' represents the state at time t *including input*,
        # so it is the effective "pre-synaptic" activity driving the recurrent connections.
        
        return h_new

    def get_output(self, h):
        """
        "The output will be simply the last 65 neurons of the reservoir."
        """
        return h[:, -self.output_size:]

    @torch.no_grad()
    def apply_hebbian_update(self, h_pre: torch.Tensor, h_post: torch.Tensor, coeffs: torch.Tensor):
        """
        Updates W_rec using the ABCD Hebbian Rule across a batch.
        coeffs: [Hidden, Hidden, 5] (A, B, C, D, LR)
        """
        # ABCD coefficients
        A, B, C, D, Eta = [coeffs[..., i] for i in range(5)]
        
        # Compute term-wise delta (averaged over batch)
        # term1: A * (post * pre^T)
        pre_post = torch.bmm(h_post.unsqueeze(2), h_pre.unsqueeze(1)).mean(0)
        
        # term2/3: B * pre / C * post
        pre_mean = h_pre.mean(0).unsqueeze(0)  # [1, Hidden] -> Broadcasts to [Hidden, Hidden]
        post_mean = h_post.mean(0).unsqueeze(1) # [Hidden, 1] -> Broadcasts to [Hidden, Hidden]
        
        delta = Eta * (A * pre_post + B * pre_mean + C * post_mean + D)
        
        # Update and normalize
        self.W_rec.add_(delta)
        max_val = self.W_rec.abs().max()
        if max_val > 5.0:
            self.W_rec.div_(max_val)

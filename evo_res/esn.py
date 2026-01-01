import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseESN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        density: float = 0.1,
        leaky_alpha: float = 0.,
        spectral_radius: float = 0.95,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Make alpha a parameter so it can be evolved
        self.leaky_alpha = nn.Parameter(torch.tensor(float(leaky_alpha)))

        # 1. No separate Input Layer matrix
        # We will inject input directly into the first `input_size` neurons
        
        # 2. Recurrent Layer (Sparse)
        W_rec = torch.randn(hidden_size, hidden_size)
        # Create the initial topology
        mask_rec = (torch.rand(hidden_size, hidden_size) < density).float()
        
        # Apply Spectral Radius scaling ONLY to the active connections
        W_rec = W_rec * mask_rec
        eigenvalues = torch.linalg.eigvals(W_rec)
        max_eigen = torch.max(torch.abs(eigenvalues))
        if max_eigen > 0:
            W_rec = W_rec * (spectral_radius / max_eigen)
            
        self.W_rec = nn.Parameter(W_rec)
        # CRITICAL: Register mask as a buffer so it saves with the model but doesn't get updated by gradients
        self.register_buffer('mask_rec', mask_rec)

    def forward(self, x, h):
        """
        Used for INFERENCE/TESTING only. 
        """
        # 1. Inject Input directly into state
        # x shape: [Batch, Input]
        # h shape: [Batch, Hidden]
        # We assume x matches the first input_size neurons
        h[:, :self.input_size] = x
        
        # 2. Apply Masks strictly
        W_rec_masked = self.W_rec * self.mask_rec
        
        # 3. Linear Step (Only Recurrent)
        # The input influence is now embedded in 'h' via the injection
        pre_activation = F.linear(h, W_rec_masked)
        
        # 4. Leaky Integration
        alpha = torch.clamp(self.leaky_alpha, 0.01, 0.99)
        
        update = F.leaky_relu(pre_activation, negative_slope=0.01)
        
        h_new = (1 - alpha) * h + alpha * update
        
        return h_new

    def get_output(self, h):
        return h[:, :self.output_size]

    def enforce_spectral_radius(self, target_radius: float = 0.95):
        """
        Rescales weights and updates the internal mask based on non-zero weights.
        """
        with torch.no_grad():
            # Update mask based on current weights (in case pruning happened in the loop)
            self.mask_rec = (self.W_rec.data != 0).float()
            
            # Re-apply mask to ensure clean weights
            self.W_rec.data *= self.mask_rec
            
            try:
                eigenvalues = torch.linalg.eigvals(self.W_rec)
                max_eigen = torch.max(torch.abs(eigenvalues))
                if max_eigen > 0:
                     scale = target_radius / max_eigen
                     self.W_rec.data.mul_(scale)
                return max_eigen.item()
            except:
                return 0.0
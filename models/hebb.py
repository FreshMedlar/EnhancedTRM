import torch
import torch.nn as nn
from typing import List

# ==============================================================================
# 1. BATCHED HEBBIAN MODEL
# ==============================================================================

class BatchedHebbianLinear(nn.Module):
    """
    A Hebbian layer where weights are plastic and coefficients are meta-learned.
    Supports a POPULATION dimension to evaluate multiple parameter sets in parallel.
    """
    def __init__(self, in_features, out_features, population_size, device="cpu", act_fn='tanh', per_synapse=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.population_size = population_size
        self.device = device
        self.act_fn = act_fn
        self.per_synapse = per_synapse
        
        # Plastic weights: [Pop, Out, In]
        self.register_buffer('weight', torch.empty(population_size, out_features, in_features, device=device))
        
        # Coefficients: [Pop, 6] or [Pop, Out, In, 6]
        if per_synapse:
            self.register_buffer('coeffs', torch.zeros(population_size, out_features, in_features, 6, device=device))
        else:
            self.register_buffer('coeffs', torch.zeros(population_size, 6, device=device))

    def set_coeffs(self, coeffs):
        self.coeffs = coeffs

    def forward(self, x):
        if x.ndim == 2: # [Pop, In]
            x = x.unsqueeze(1) # [Pop, 1, In]
            
        w_t = self.weight.transpose(1, 2) # [Pop, In, Out]
        y = torch.bmm(x, w_t) # [Pop, Batch, Out]
        
        if self.act_fn == 'tanh':
            return torch.tanh(y)
        else:
            return y

    def update_weights(self, pre, post):
        """
        pre: [Pop, Batch, In] 
        post: [Pop, Batch, Out] 
        """
        # Unpack coefficients
        if self.per_synapse:
            A, B, C, D, E, eta = [self.coeffs[..., i] for i in range(6)]
        else:
            # Broadcast to [Pop, 1, 1] for layers
            A, B, C, D, E, eta = [self.coeffs[:, i:i+1].unsqueeze(2) for i in range(6)]
        
        # Interaction: outer product averaged over batch
        # post: [P, B, O], pre: [P, B, I] -> [P, O, I]
        interaction = A * torch.einsum('pbo,pbi->poi', post, pre) / pre.shape[1]
        
        # Pre term: broadcast x_avg [P, 1, I]
        pre_avg = pre.mean(dim=1).unsqueeze(1) 
        pre_term = B * pre_avg
        
        # Post term: broadcast y_avg [P, O, 1]
        post_avg = post.mean(dim=1).unsqueeze(2)
        post_term = C * post_avg
        
        # dW using the ABCD + E rule (weight decay)
        dW = eta * (interaction + pre_term + post_term + D - E * self.weight)
        
        # Update
        self.weight = self.weight + dW
        self.weight = torch.clamp(self.weight, -3.0, 3.0)

    def reset_plastic_weights(self):
        # Initialize with small noise so hidden layers have initial activity
        # This prevents the "zero-gradient" trap where W2 never updates because h=0
        torch.nn.init.normal_(self.weight, std=0.1)

class BatchedHebbianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, population_size, device="cpu", per_synapse=False):
        super().__init__()
        self.device = device
        self.per_synapse = per_synapse
        self.hidden_dim = hidden_dim
        
        if hidden_dim > 0:
            self.layer1 = BatchedHebbianLinear(input_dim, hidden_dim, population_size, device=device, act_fn='tanh', per_synapse=per_synapse)
            self.layer2 = BatchedHebbianLinear(hidden_dim, output_dim, population_size, device=device, act_fn='identity', per_synapse=per_synapse)
            
            if per_synapse:
                self.l1_params = hidden_dim * input_dim * 6
                self.l2_params = output_dim * hidden_dim * 6
            else:
                self.l1_params = 6
                self.l2_params = 6
        else:
            # Single layer model
            self.layer1 = BatchedHebbianLinear(input_dim, output_dim, population_size, device=device, act_fn='identity', per_synapse=per_synapse)
            self.l1_params = (output_dim * input_dim * 6) if per_synapse else 6
            self.l2_params = 0
            
        self.total_params = self.l1_params + self.l2_params
        self.to(device)

    def load_flat_params(self, flat_params_batch):
        p1 = flat_params_batch[:, :self.l1_params]
        self.layer1.set_coeffs(p1.view(-1, 6) if not self.per_synapse else p1.view(flat_params_batch.shape[0], -1, 6))
        
        if self.hidden_dim > 0:
            p2 = flat_params_batch[:, self.l1_params:]
            self.layer2.set_coeffs(p2.view(-1, 6) if not self.per_synapse else p2.view(flat_params_batch.shape[0], -1, 6))

    def forward(self, x):
        if self.hidden_dim > 0:
            h = self.layer1(x)
            out = self.layer2(h)
            return h, out
        else:
            out = self.layer1(x)
            return None, out
    
    def update(self, x_in, h_act, out_act):
        if self.hidden_dim > 0:
            self.layer1.update_weights(x_in, h_act)
            self.layer2.update_weights(h_act, out_act)
        else:
            self.layer1.update_weights(x_in, out_act)
        
    def reset_episode(self):
        self.layer1.reset_plastic_weights()
        if self.hidden_dim > 0:
            self.layer2.reset_plastic_weights()

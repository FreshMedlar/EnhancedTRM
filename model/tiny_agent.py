import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TinyRecursiveAgent(nn.Module):
    def __init__(self, num_actions, hidden_dim=64, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.hidden_dim = hidden_dim
        
        # 1. Tiny Encoder (NatureCNN-like but smaller/strided for 84x84)
        # Input: (B, 1, 84, 84)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4), # -> (16, 20, 20)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), # -> (32, 9, 9)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, hidden_dim),
            nn.ReLU()
        )
        
        # 2. The Core Recursive Layer (Shared weights)
        # "Brain" that thinks for multiple steps
        self.core_layer = nn.GRUCell(hidden_dim, hidden_dim)
        
        # 3. Heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state_input, prev_h=None):
        """
        Args:
            state_input: (B, 1, H, W) Observation
            prev_h: (B, hidden_dim) Previous hidden state (from previous game step)
            
        Returns:
            action_logits: (B, num_actions) Final thought
            value: (B, 1) Estimated value
            next_h: (B, hidden_dim) Updated hidden state to carry over to next step
            all_logits: (num_iterations, B, num_actions) For deep supervision
        """
        batch_size = state_input.size(0)
        
        # 1. Encode observation to latent "thought" seed
        z = self.encoder(state_input) # (B, D)
        
        # 2. Initialize internal state if not provided
        if prev_h is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=state_input.device)
        else:
            h = prev_h
            
        # Recursive "Listening/Thinking" Loop
        # At each step, we update h using input z and previous h.
        # But wait, original request said: 
        # "z = h" (hidden state acts as context for next adjustment)
        # "h = core(z, h)"
        # Let's interpret: 
        # Input 'z' stays constant (the observation), 'h' evolves.
        
        all_logits = []
        
        current_thought = h
        
        for _ in range(self.num_iterations):
            # Refine thought based on observation z and current thought
            # We treat 'z' as the input to the cell, 'current_thought' as hidden
            current_thought = self.core_layer(z, current_thought)
            
            # Deep Supervision outputs
            logits = self.action_head(current_thought)
            all_logits.append(logits)
            
        # Stack all intermediate outputs: (Iter, B, Act)
        all_logits = torch.stack(all_logits)
        
        final_logits = all_logits[-1]
        value = self.value_head(current_thought)
        
        return final_logits, value, current_thought, all_logits
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())

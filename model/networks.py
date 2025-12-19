import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, latent_dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, latents):
        # latents: (B, Dim) - we typically take the last token's embedding
        return self.net(latents)

class Critic(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, latents):
        return self.net(latents)

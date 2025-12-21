import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.ops import MLP

class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        # proj: (B, D)
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        # CRITICAL: Scaling by batch size proj.size(-2)
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()

class ResNetEncoder(nn.Module):
    def __init__(self, proj_dim=128, in_channels=3):
        super().__init__()
        self.backbone = timm.create_model(
            "resnet18",
            pretrained=False,
            num_classes=0,
            in_chans=in_channels,
            global_pool="avg"
        )
        self.embed_dim = self.backbone.num_features
        
        # Matching MINIMAL.md capacity: MLP(embed, [2048, 2048, proj_dim])
        self.proj = nn.Sequential(
            nn.Linear(self.embed_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Linear(2048, proj_dim) # Output layer (no norm/act usually at end of projector if following standard SSL)
        )

    def forward(self, x):
        emb = self.backbone(x)
        return emb, self.proj(emb)

class Predictor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, state, action_embed):
        x = torch.cat([state, action_embed], dim=-1)
        return self.net(x)

class JEPAWorldModel(nn.Module):
    def __init__(self, latent_dim, num_actions, in_channels=3):
        super().__init__()
        self.encoder = ResNetEncoder(proj_dim=latent_dim, in_channels=in_channels)
        self.act_embedding = nn.Embedding(num_actions, latent_dim)
        self.predictor_net = Predictor(latent_dim, latent_dim)
    
    def forward(self, x):
        emb, proj = self.encoder(x)
        return proj
    def __init__(self, latent_dim, num_actions, in_channels=3):
        super().__init__()
        self.encoder = ResNetEncoder(proj_dim=latent_dim, in_channels=in_channels)
        self.act_embedding = nn.Embedding(num_actions, latent_dim)
        self.predictor_net = Predictor(latent_dim, latent_dim) 
    
    def forward(self, x):
        emb, proj = self.encoder(x)
        return proj


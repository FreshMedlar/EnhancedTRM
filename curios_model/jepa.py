import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from rwkv_model import RWKVWorldModel, RWKVConfig

class SIGReg(nn.Module):
    """SIGReg loss from LeJEPA - applied to predicted embeddings."""
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
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    """
    Frozen ViT encoder for frame embedding.
    Uses a small ViT variant adapted for 84x84 grayscale images.
    """
    def __init__(self, img_size=84, in_channels=1, embed_dim=128):
        super().__init__()
        # Use a tiny ViT with patch16, adapted for our input size
        self.backbone = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=False,
            num_classes=embed_dim,  # Output dimension
            img_size=img_size,
            in_chans=in_channels,
        )
        self.embed_dim = embed_dim
        
        # Freeze by default
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) or (B, T, C, H, W)
        Returns:
            embeddings: (B, D) or (B, T, D)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            emb = self.backbone(x)  # (B*T, D)
            return emb.view(B, T, -1)  # (B, T, D)
        else:
            return self.backbone(x)  # (B, D)


class TemporalRWKV(nn.Module):
    """
    Lightweight RWKV for temporal prediction.
    Processes one frame embedding per time step.
    """
    def __init__(self, embed_dim=128, n_layer=2):
        super().__init__()
        config = RWKVConfig(
            vocab_size=0,  # We use inputs_embeds
            block_size=64,  # Max sequence length
            n_layer=n_layer,
            n_embd=embed_dim,
            intermediate_size=embed_dim * 2,  # Smaller FFN
        )
        self.rwkv = RWKVWorldModel(config)
        self.rwkv.rwkv.wte = None  # Disable token embedding
        self.n_layer = n_layer
        self.embed_dim = embed_dim
    
    def get_initial_state(self, batch_size, device):
        """Create initial RWKV state."""
        # State shape: (n_layer, batch_size, 5, n_embd)
        # 5 = PREV_X_TIME, NUM_STATE, DEN_STATE, MAX_STATE, PREV_X_CHANNEL
        return torch.zeros(self.n_layer, batch_size, 5, self.embed_dim, device=device)
    
    def forward(self, x, state=None):
        """
        Args:
            x: (B, T, D) frame embeddings
            state: RWKV recurrent state (only used for T=1 inference)
        Returns:
            hidden: (B, T, D) temporal hidden states
            state: Updated RWKV state
        """
        B, T, D = x.shape
        
        if T > 1:
            # Training mode: process full sequence without state (parallel)
            _, hidden, _, _ = self.rwkv(idx=None, inputs_embeds=x, state=None)
            return hidden, None
        else:
            # Inference mode: single step with recurrent state
            if state is None:
                state = self.get_initial_state(B, x.device)
            _, hidden, _, state = self.rwkv(idx=None, inputs_embeds=x, state=state)
            return hidden, state


class Predictor(nn.Module):
    """Predicts next frame embedding from current hidden + action."""
    def __init__(self, embed_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, state, action_embed):
        x = torch.cat([state, action_embed], dim=-1)
        return self.net(x)


class RewardPredictor(nn.Module):
    """Predicts reward from hidden state + action."""
    def __init__(self, embed_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action_embed):
        x = torch.cat([state, action_embed], dim=-1)
        return self.net(x)


class JEPAWorldModel(nn.Module):
    """
    New architecture: ViT (frozen) → RWKV (temporal) → Predictor
    
    Flow:
        pixels → ViT encoder → frame embedding → RWKV → temporal hidden
        temporal hidden + action → Predictor → predicted next embedding
    """
    def __init__(self, latent_dim, num_actions, in_channels=1, img_size=84):
        super().__init__()
        
        # Frozen ViT encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=latent_dim,
        )
        
        # Lightweight RWKV for temporal processing
        self.temporal = TemporalRWKV(
            embed_dim=latent_dim,
            n_layer=2,  # Small for speed
        )
        
        # Action embedding
        self.act_embedding = nn.Embedding(num_actions, latent_dim)
        
        # Predictors
        self.predictor_net = Predictor(latent_dim, latent_dim)
        self.reward_predictor = RewardPredictor(latent_dim, latent_dim)
        
        self.latent_dim = latent_dim
    
    def encode(self, x, state=None):
        """
        Encode pixels to temporal hidden state.
        
        Args:
            x: (B, T, C, H, W) or (B, C, H, W)
            state: RWKV recurrent state
        Returns:
            hidden: (B, T, D) or (B, D) temporal hidden states
            state: Updated RWKV state
        """
        # Get frame embeddings from frozen ViT
        with torch.no_grad():
            frame_emb = self.encoder(x)  # (B, T, D) or (B, D)
        
        # Add temporal dimension if single frame
        if frame_emb.dim() == 2:
            frame_emb = frame_emb.unsqueeze(1)  # (B, 1, D)
        
        # Process through RWKV
        hidden, state = self.temporal(frame_emb, state)
        
        return hidden, state
    
    def forward(self, x, state=None):
        """
        Full forward pass.
        Returns hidden states that can be used for prediction.
        """
        hidden, state = self.encode(x, state)
        return hidden, state

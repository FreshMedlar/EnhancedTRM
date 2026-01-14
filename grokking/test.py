import torch
import torch.nn as nn
import torch.optim as optim
import random
import time

# --- Configuration (The "Fast Grokking" Levers) ---
config = {
    'p': 97,              # Prime number (Modulo)
    'train_frac': 0.4,    # fraction of data for training (40%)
    'steps': 5000,        # Max training steps
    'lr': 1e-3,           # Learning rate
    'weight_decay': 1.0,  # AGGRESSIVE weight decay (Crucial for speed)
    'batch_size': 512,    # Large batch size helps
    'embed_dim': 128,
    'hidden_dim': 128,
    'num_heads': 4,
    'layers': 1,
    'seed': 42
}

# Set seeds for reproducibility
torch.manual_seed(config['seed'])
random.seed(config['seed'])

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, layers):
        super().__init__()
        # Token embedding + Learned Positional embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
        
        # Transformer Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.0,
            batch_first=True,
            norm_first=True # Pre-Norm often stabilizes training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # Output Head
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)

        # Initialize weights small (Rich regime initialization)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # x shape: [batch, 2] -> (a, b)
        emb = self.embedding(x) + self.pos_embedding
        features = self.transformer(emb)
        # We take the mean of the two tokens as the representation
        pooled = features.mean(dim=1) 
        return self.decoder(pooled)

def generate_data(p):
    data = []
    for i in range(p):
        for j in range(p):
            target = (i + j) % p
            data.append(([i, j], target))
    return data

def main():
    print(f"--- Starting Grokking Experiment (Modulo {config['p']}) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Data Prep
    data = generate_data(config['p'])
    random.shuffle(data)
    
    split_idx = int(len(data) * config['train_frac'])
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Convert to tensors
    X_train = torch.tensor([x[0] for x in train_data], device=device)
    y_train = torch.tensor([x[1] for x in train_data], device=device)
    
    X_val = torch.tensor([x[0] for x in val_data], device=device)
    y_val = torch.tensor([x[1] for x in val_data], device=device)

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    
    # 2. Model & Optimizer
    model = GrokkingTransformer(
        vocab_size=config['p'], 
        embed_dim=config['embed_dim'], 
        hidden_dim=config['hidden_dim'], 
        num_heads=config['num_heads'], 
        layers=config['layers']
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.98)
    )

    criterion = nn.CrossEntropyLoss()

    # 3. Training Loop
    start_time = time.time()
    for step in range(config['steps']):
        model.train()
        
        # Full batch or large batch training is preferred for grokking
        # Here we do full batch for simplicity and stability
        indices = torch.randperm(len(X_train))[:config['batch_size']]
        x_batch = X_train[indices]
        y_batch = y_train[indices]

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            model.eval()
            with torch.no_grad():
                # Check Train Acc
                train_logits = model(X_train)
                train_acc = (train_logits.argmax(dim=-1) == y_train).float().mean().item()
                
                # Check Val Acc
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(dim=-1) == y_val).float().mean().item()
                
                # Visual output
                progress = "=" * int(val_acc * 20)
                space = " " * (20 - len(progress))
                print(f"Step {step:04d} | Loss: {loss.item():.4f} | "
                      f"Train Acc: {train_acc*100:.1f}% | "
                      f"Val Acc: [{progress}{space}] {val_acc*100:.1f}%")

            # Early stop if grokked
            if val_acc > 0.99:
                print("\n--- GROKKED! ---")
                print(f"Solved in {step} steps and {time.time()-start_time:.2f} seconds.")
                break

if __name__ == "__main__":
    main()

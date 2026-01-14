import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math

# --- HARD MODE CONFIGURATION ---
config = {
    'task': 'multiplication', # Harder than addition
    'p': 113,                 # Larger Prime
    'train_frac': 0.3,        # Data Starvation (Only 30% seen)
    'steps': 40000,           # Needs much longer to grok
    'lr': 1e-3,
    'weight_decay': 0.1,      # LOW decay = Longer Memorization Phase
    'batch_size': 512,
    'embed_dim': 128,
    'hidden_dim': 256,        # Slightly wider to allow more memorization capacity
    'num_heads': 4,
    'layers': 2,              # Deeper = Harder to optimize for generalization
    'seed': 999
}

# Reproducibility
torch.manual_seed(config['seed'])
random.seed(config['seed'])

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_heads, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.0,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        emb = self.embedding(x) + self.pos_embedding
        features = self.transformer(emb)
        pooled = features.mean(dim=1)
        return self.decoder(pooled)

def generate_data(p, task='addition'):
    data = []
    for i in range(p):
        for j in range(p):
            if task == 'addition':
                target = (i + j) % p
            elif task == 'multiplication':
                target = (i * j) % p
            data.append(([i, j], target))
    return data

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Hard Mode: Modulo {config['p']} {config['task'].upper()} ---")
    print(f"--- Data Fraction: {config['train_frac']*100}% | Weight Decay: {config['weight_decay']} ---")
    print(f"Device: {device}")

    # Data Setup
    data = generate_data(config['p'], config['task'])
    random.shuffle(data)
    
    # Strictly split data (No overlap)
    split_idx = int(len(data) * config['train_frac'])
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    X_train = torch.tensor([x[0] for x in train_data], device=device)
    y_train = torch.tensor([x[1] for x in train_data], device=device)
    X_val = torch.tensor([x[0] for x in val_data], device=device)
    y_val = torch.tensor([x[1] for x in val_data], device=device)

    print(f"Train size: {len(X_train)} | Val size: {len(X_val)}")
    
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
    
    history = {'step': [], 'val_acc': []}
    start_time = time.time()
    
    try:
        for step in range(config['steps']):
            model.train()
            indices = torch.randperm(len(X_train))[:config['batch_size']]
            optimizer.zero_grad()
            logits = model(X_train[indices])
            loss = criterion(logits, y_train[indices])
            loss.backward()
            optimizer.step()

            if step % 500 == 0:
                model.eval()
                with torch.no_grad():
                    train_acc = (model(X_train).argmax(dim=-1) == y_train).float().mean().item()
                    val_acc = (model(X_val).argmax(dim=-1) == y_val).float().mean().item()
                    
                    history['step'].append(step)
                    history['val_acc'].append(val_acc)

                    # Dynamic Progress Bar
                    bar_len = 20
                    filled = int(val_acc * bar_len)
                    bar = "█" * filled + "-" * (bar_len - filled)
                    
                    print(f"Step {step:05d} | Train Acc: {train_acc*100:5.1f}% | "
                          f"Val Acc: {val_acc*100:5.1f}% [{bar}]")

                if val_acc > 0.99:
                    print(f"\n--- GROKKED at step {step}! ---")
                    print(f"Time taken: {time.time()-start_time:.1f}s")
                    break
    except KeyboardInterrupt:
        print("\nStopped by user.")

    # Simple ASCII Plot of the run
    print("\n--- Grokking Curve ---")
    if len(history['val_acc']) > 0:
        for v in history['val_acc']:
            pos = int(v * 40)
            print("|" + " " * pos + "*")
        print("|" + "_" * 41)

if __name__ == "__main__":
    main()

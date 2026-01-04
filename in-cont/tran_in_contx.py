import torch
import torch.nn as nn
import torch.optim as optim
import math
import wandb

# --- Hyperparameters ---
BATCH_SIZE = 64
SEQ_LEN = 50          # Increased from 20 to see longer context learning
X_DIM = 20            # Increased from 10
D_MODEL = 256         # Increased from 128
N_LAYER = 6           # Increased from 4
N_HEAD = 8            # Increased from 4
LR = 0.0001           # Lower LR for more stable training
STEPS = 20000         # Much more steps to see 'emergence'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize WandB
wandb.init(
    project="In-Context-Learning",
    config={
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "x_dim": X_DIM,
        "d_model": D_MODEL,
        "n_layer": N_LAYER,
        "n_head": N_HEAD,
        "lr": LR,
        "steps": STEPS,
    }
)

# --- 1. The Dataset (Linear Regression) ---
def get_batch(batch_size, seq_len, x_dim):
    # 1. Sample random weights w for each batch element (B, x_dim, 1)
    # The model must figure out these weights purely from context!
    w = torch.randn(batch_size, x_dim, 1, device=device)
    
    # 2. Sample random inputs x (B, seq_len, x_dim)
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    
    # 3. Compute y = x @ w (B, seq_len, 1)
    # We add a tiny bit of noise usually, but for pure ICL demonstration, clean is fine.
    y = x @ w 
    
    return x, y

# --- 2. The Model (Decoder-only Transformer) ---
class ICLTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Input embeddings
        self.x_enc = nn.Linear(X_DIM, D_MODEL)
        self.y_enc = nn.Linear(1, D_MODEL)
        
        # Transformer Backbone
        # We use a standard Encoder with causal masking (acts like a Decoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, 
                                                   dim_feedforward=D_MODEL*4, 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYER)
        
        # Prediction Head
        self.predict = nn.Linear(D_MODEL, 1)

    def forward(self, x, y):
        # x: (B, L, D_x), y: (B, L, 1)
        B, L, _ = x.shape
        
        # Project inputs
        x_emb = self.x_enc(x) # (B, L, D_model)
        y_emb = self.y_enc(y) # (B, L, D_model)
        
        # Interleave x and y: [x1, y1, x2, y2, ...]
        # We create a sequence of length 2*L
        combined = torch.stack((x_emb, y_emb), dim=2) # (B, L, 2, D_model)
        combined = combined.view(B, L*2, D_MODEL) # (B, 2L, D_model)
        
        # Create Causal Mask (upper triangular is -inf)
        # This prevents the model from seeing future tokens.
        # Specifically, when at x_i, it shouldn't see y_i.
        mask = torch.triu(torch.full((L*2, L*2), float('-inf'), device=device), diagonal=1)
        
        # Pass through Transformer
        out = self.transformer(combined, mask=mask)
        
        # We only care about predictions at the 'x' positions (even indices)
        # Input: x1 -> Predict: y1
        # Input: y1 -> Predict: ... (we don't care)
        # So we take the output corresponding to the even positions (0, 2, 4...)
        z_x = out[:, 0::2, :] # (B, L, D_model)
        
        # Predict y
        y_pred = self.predict(z_x) # (B, L, 1)
        
        return y_pred

# --- 3. Training Loop ---
model = ICLTransformer().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Starting training...")

for step in range(STEPS):
    x, y = get_batch(BATCH_SIZE, SEQ_LEN, X_DIM)
    
    optimizer.zero_grad()
    y_pred = model(x, y)
    
    # Calculate loss
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    
    # Log to WandB
    wandb.log({"loss": loss.item()}, step=step)
    
    if step % 200 == 0:
        print(f"Step {step}: MSE Loss = {loss.item():.4f}")

# --- 4. The "Proof" (In-Context Learning Test) ---
# To prove ICL, we look at the error at index 0 vs index 19.
# At index 0, the model has seen NO examples. It should guess blindly.
# At index 19, the model has seen 19 pairs of (x,y). It should know w perfectly.

print("\n--- Testing In-Context Learning Capabilities ---")
model.eval()
with torch.no_grad():
    test_x, test_y = get_batch(1000, SEQ_LEN, X_DIM) # Larger batch for stable stats
    preds = model(test_x, test_y)
    
    # Calculate Squared Error for each position in the sequence
    squared_errors = (preds - test_y) ** 2 # (B, L, 1)
    mse_per_position = squared_errors.mean(dim=0).squeeze().cpu().numpy()

# Simple text-based plot
print(f"{'Pos':<5} | {'MSE Loss':<10} | {'Visual'}")
print("-" * 35)
max_loss = mse_per_position[0]
for i, loss in enumerate(mse_per_position):
    # Create a bar proportional to loss
    bar_len = int((loss / max_loss) * 20)
    bar = "█" * bar_len
    print(f"{i:<5} | {loss:.4f}     | {bar}")

print("\nInterpretation:")
print("- Pos 0: High loss (Model hasn't seen 'w' yet, guessing blindly)")
print(f"- Pos {SEQ_LEN-1}: Low loss (Model 'learned' 'w' in-context and applied it)")

# Log final results to WandB
data = [[i, loss] for i, loss in enumerate(mse_per_position)]
table = wandb.Table(data=data, columns=["position", "mse"])
wandb.log({"mse_per_position_plot": wandb.plot.line(table, "position", "mse", title="MSE Loss per Position")})
wandb.finish()
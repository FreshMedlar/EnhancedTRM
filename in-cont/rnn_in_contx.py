import torch
import torch.nn as nn
import torch.optim as optim
import math
import wandb

# --- Hyperparameters ---
# Keeping these consistent with tran_in_contx.py for a fair comparison
BATCH_SIZE = 64
SEQ_LEN = 50          
X_DIM = 20            
HIDDEN_SIZE = 256     # Matches D_MODEL in Transformer
N_LAYER = 4           # RNNs are typically shallower than Transformers
LR = 0.0001           
STEPS = 20000         

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize WandB
wandb.init(
    project="In-Context-Learning",
    name="RNN-ICL-Baseline",
    config={
        "architecture": "RNN (GRU)",
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "x_dim": X_DIM,
        "hidden_size": HIDDEN_SIZE,
        "n_layer": N_LAYER,
        "lr": LR,
        "steps": STEPS,
    }
)

# --- 1. The Dataset (Linear Regression) ---
def get_batch(batch_size, seq_len, x_dim):
    # Sample random weights w for each batch element (B, x_dim, 1)
    w = torch.randn(batch_size, x_dim, 1, device=device)
    # Sample random inputs x (B, seq_len, x_dim)
    x = torch.randn(batch_size, seq_len, x_dim, device=device)
    # Compute y = x @ w (B, seq_len, 1)
    y = x @ w 
    return x, y

# --- 2. The Model (RNN/GRU Baseline) ---
class ICLRNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input embeddings
        self.x_enc = nn.Linear(X_DIM, HIDDEN_SIZE)
        self.y_enc = nn.Linear(1, HIDDEN_SIZE)
        
        # RNN Backbone
        # GRU is inherently causal (it only sees past tokens)
        self.rnn = nn.GRU(input_size=HIDDEN_SIZE, 
                          hidden_size=HIDDEN_SIZE, 
                          num_layers=N_LAYER, 
                          batch_first=True)
        
        # Prediction Head
        self.predict = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, y):
        # x: (B, L, D_x), y: (B, L, 1)
        B, L, _ = x.shape
        
        # Project inputs
        x_emb = self.x_enc(x) # (B, L, HIDDEN_SIZE)
        y_emb = self.y_enc(y) # (B, L, HIDDEN_SIZE)
        
        # Interleave x and y: [x1, y1, x2, y2, ...]
        combined = torch.stack((x_emb, y_emb), dim=2) # (B, L, 2, HIDDEN_SIZE)
        combined = combined.view(B, L*2, HIDDEN_SIZE) # (B, 2L, HIDDEN_SIZE)
        
        # Pass through RNN
        # out: (B, 2L, HIDDEN_SIZE)
        out, _ = self.rnn(combined)
        
        # We extract states corresponding to 'x' positions (0, 2, 4...)
        # Because out[0] is the state after seeing x1
        # out[2] is the state after seeing x1, y1, x2
        z_x = out[:, 0::2, :] # (B, L, HIDDEN_SIZE)
        
        # Predict y
        y_pred = self.predict(z_x) # (B, L, 1)
        
        return y_pred

# --- 3. Training Loop ---
model = ICLRNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Starting RNN training...")

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
print("\n--- Testing RNN In-Context Learning Capabilities ---")
model.eval()
with torch.no_grad():
    test_x, test_y = get_batch(1000, SEQ_LEN, X_DIM) 
    preds = model(test_x, test_y)
    
    # Calculate Squared Error for each position
    squared_errors = (preds - test_y) ** 2 
    mse_per_position = squared_errors.mean(dim=0).squeeze().cpu().numpy()

# Simple text-based plot
print(f"{'Pos':<5} | {'MSE Loss':<10} | {'Visual'}")
print("-" * 35)
max_loss = mse_per_position[0]
for i, loss in enumerate(mse_per_position):
    bar_len = int((loss / max_loss) * 20) if max_loss > 0 else 0
    bar = "█" * bar_len
    print(f"{i:<5} | {loss:.4f}     | {bar}")

# Log final results to WandB
data = [[i, loss] for i, loss in enumerate(mse_per_position)]
table = wandb.Table(data=data, columns=["position", "mse"])
wandb.log({"mse_per_position_plot": wandb.plot.line(table, "position", "mse", title="RNN MSE Loss per Position")})
wandb.finish()

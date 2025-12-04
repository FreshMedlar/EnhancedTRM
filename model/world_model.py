import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class WorldModel(nn.Module):
    def __init__(self,obs_dim, act_dim, hidden_dim: int, output_dim: int):
        """
        A simple 2-layer Feedforward Neural Network (MLP).
        Structure: Input -> Linear -> ReLU -> Linear -> Output
        """
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, obs, act) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        return x

def train_model():
    # --- SETUP ---
    # Dimensions
    input_dim = 10
    output_dim = 1 
    
    model = WorldModel(input_dim, 32, output_dim)

    # Create "Fake" Data 
    # Let's pretend the rule is: output should be the sum of the input vector
    X = torch.randn(100, input_dim)    # 100 samples
    y = X.sum(dim=1, keepdim=True)     # Target labels (ground truth)

    # 2. The Loss Function (Criterion)
    # MSELoss is standard for regression (predicting a number)
    criterion = nn.MSELoss()

    # 3. The Optimizer
    # SGD (Stochastic Gradient Descent) with a learning rate of 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    print(f"Initial Loss: {criterion(model(X), y).item():.4f} (Random weights)")

    # --- THE TRAINING LOOP ---
    epochs = 200
    for epoch in range(epochs):
        # A. Zero the gradients
        # (PyTorch accumulates gradients by default, so we must clear them)
        optimizer.zero_grad()

        # B. Forward Pass
        predictions = model(X)

        # C. Calculate Loss
        loss = criterion(predictions, y)

        # D. Backward Pass (Backpropagation)
        # This calculates the gradient of the loss w.r.t. every weight
        loss.backward()

        # E. Optimizer Step
        # Adjust weights: weight = weight - (learning_rate * gradient)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print(f"Final Loss: {criterion(model(X), y).item():.4f} (Learned weights)")

if __name__ == "__main__":
    train_model()





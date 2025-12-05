import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RewardModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, hidden_dim2: int, output_dim: int):
        """
        A simple 2-layer Feedforward Neural Network (MLP).
        Structure: Input -> Linear -> ReLU -> Linear -> Output
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.fc3(x)
        return x



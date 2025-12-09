# This example shows how to use TRM to solve Sudoku puzzles, the task where TRM achieved 87.4% accuracy with only 5M parameters.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple
from trm_implementation import TinyRecursiveModel, TRMWithACT


class SudokuDataset(Dataset):
    """
    Sudoku dataset for training and evaluation.
    """
    
    def __init__(self, puzzles: np.ndarray, solutions: np.ndarray):
        """
        Args:
            puzzles: Array of shape (n_samples, 81) with values 0-9
                     0 represents empty cells
            solutions: Array of shape (n_samples, 81) with values 1-9
        """
        self.puzzles = torch.tensor(puzzles, dtype=torch.long)
        self.solutions = torch.tensor(solutions, dtype=torch.long)
        
    def __len__(self):
        return len(self.puzzles)
    
    def __getitem__(self, idx):
        return self.puzzles[idx], self.solutions[idx]
    
    @staticmethod
    def augment_sudoku(puzzle: np.ndarray, num_augmentations: int = 100) -> np.ndarray:
        """
        Generate augmentations of a Sudoku puzzle through valid transformations.
        
        Valid Sudoku transformations (preserve solution uniqueness):
        - Shuffle rows within bands
        - Shuffle columns within stacks
        - Permute digits (relabel 1-9)
        - Transpose
        """
        augmented = [puzzle.copy()]
        
        for _ in range(num_augmentations - 1):
            p = puzzle.copy()
            
            # Random digit permutation
            if np.random.rand() < 0.5:
                digit_map = np.arange(10)
                np.random.shuffle(digit_map[1:])  # Keep 0 as empty
                p = digit_map[p]
            
            # Random band shuffle (permute rows within 3x3 bands)
            if np.random.rand() < 0.5:
                bands = [0, 3, 6]
                for band_start in bands:
                    rows = list(range(band_start, band_start + 3))
                    np.random.shuffle(rows)
                    p[band_start:band_start+3] = p[rows]
            
            # Random stack shuffle (permute columns within 3x3 stacks)
            if np.random.rand() < 0.5:
                stacks = [0, 3, 6]
                for stack_start in stacks:
                    cols = list(range(stack_start, stack_start + 3))
                    np.random.shuffle(cols)
                    p[:, stack_start:stack_start+3] = p[:, cols]
            
            # Random transpose
            if np.random.rand() < 0.3:
                p = p.T
            
            augmented.append(p.reshape(-1))
        
        return np.array(augmented)


class SudokuSolver:
    """
    Complete training and evaluation pipeline for Sudoku solving with TRM.
    """
    
    def __init__(
        self,
        model: TinyRecursiveModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.device = device
        
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=weight_decay,
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for puzzle, solution in train_loader:
            puzzle = puzzle.to(self.device)
            solution = solution.to(self.device)
            
            # Forward pass with deep supervision
            logits, _, _ = self.model(puzzle, supervision_steps=16)
            
            # Loss on final supervision step
            loss = self.loss_fn(
                logits.view(-1, 10),
                solution.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Update EMA
            self.model.update_ema()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate model.
        
        Returns:
            (accuracy, loss) where accuracy is % of correctly predicted cells
        """
        self.model.eval()
        
        # Use EMA weights for evaluation
        original_state = self.model.state_dict()
        if self.model.use_ema:
            ema_state = self.model.get_ema_state_dict()
            self.model.load_state_dict(ema_state)
        
        total_loss = 0.0
        total_correct = 0
        total_cells = 0
        
        for puzzle, solution in eval_loader:
            puzzle = puzzle.to(self.device)
            solution = solution.to(self.device)
            
            # Forward pass (use less supervision for speed during eval)
            logits, predictions, _ = self.model(puzzle, supervision_steps=8)
            
            # Compute loss
            loss = self.loss_fn(
                logits.view(-1, 10),
                solution.view(-1)
            )
            total_loss += loss.item()
            
            # Compute accuracy
            correct = (predictions == solution).sum().item()
            total_correct += correct
            total_cells += puzzle.numel()
        
        # Restore original state
        self.model.load_state_dict(original_state)
        
        accuracy = (total_correct / total_cells) * 100.0
        avg_loss = total_loss / len(eval_loader)
        
        return accuracy, avg_loss
    
    def solve_puzzle(self, puzzle: np.ndarray) -> np.ndarray:
        """
        Solve a single Sudoku puzzle.
        
        Args:
            puzzle: Array of shape (81,) with values 0-9
            
        Returns:
            solution: Predicted solution
        """
        self.model.eval()
        
        puzzle_tensor = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0)
        puzzle_tensor = puzzle_tensor.to(self.device)
        
        with torch.no_grad():
            _, predictions, _ = self.model(
                puzzle_tensor,
                supervision_steps=16  # Full supervision at test time
            )
        
        return predictions[0].cpu().numpy()


# Example training script
def main():
    """
    Complete example of training TRM on Sudoku.
    """
    
    # Hyperparameters (from the paper's Sudoku-Extreme setup)
    batch_size = 768
    num_epochs = 50000
    eval_interval = 5000
    
    # Generate dummy Sudoku data for demonstration
    # In practice, load from actual Sudoku datasets
    print("Generating dummy Sudoku dataset...")
    np.random.seed(42)
    n_train_samples = 1000
    n_eval_samples = 100
    
    # Create random valid (but unsolved) puzzles and solutions
    train_puzzles = np.random.randint(0, 10, (n_train_samples, 81))
    train_solutions = np.random.randint(1, 10, (n_train_samples, 81))
    
    eval_puzzles = np.random.randint(0, 10, (n_eval_samples, 81))
    eval_solutions = np.random.randint(1, 10, (n_eval_samples, 81))
    
    # Create datasets with augmentation
    print("Creating training dataset with augmentations...")
    augmented_puzzles = []
    augmented_solutions = []
    
    for puzzle, solution in zip(train_puzzles, train_solutions):
        augmented = SudokuDataset.augment_sudoku(puzzle, num_augmentations=1000)
        augmented_puzzles.extend(augmented)
        augmented_solutions.extend([solution] * len(augmented))
    
    train_dataset = SudokuDataset(
        np.array(augmented_puzzles),
        np.array(augmented_solutions)
    )
    eval_dataset = SudokuDataset(eval_puzzles, eval_solutions)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    
    # Initialize model (7M parameters like the paper)
    print("Initializing TRM model...")
    model = TinyRecursiveModel(
        vocab_size=10,           # Digits 0-9
        embed_dim=512,
        L_layers=2,              # \"Tiny\" network
        H_cycles=3,              # T=3 from paper
        L_cycles=6,              # n=6 from paper
        num_heads=8,
        mlp_dim=2048,
        max_seq_len=81,          # 9x9 Sudoku flattened
        use_ema=True,
        ema_rate=0.999,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # Initialize solver
    solver = SudokuSolver(model, learning_rate=1e-4, weight_decay=1.0)
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(num_epochs):
        train_loss = solver.train_epoch(train_loader)
        
        if (epoch + 1) % eval_interval == 0:
            eval_acc, eval_loss = solver.evaluate(eval_loader)
            print(f\"Epoch {epoch+1:6d}: \"\n f\"Train Loss = {train_loss:.4f}, \"\n  f\"Eval Acc = {eval_acc:.1f}%, \"\n  f\"Eval Loss = {eval_loss:.4f}\")
    
    # Example inference
    print(\"\\nExample inference on single puzzle:\")
    test_puzzle = train_puzzles[0]
    print(f\"Input puzzle (flattened): {test_puzzle[:18]}...\")
    
    solution = solver.solve_puzzle(test_puzzle)
    print(f\"Predicted solution: {solution[:18]}...\")
    
    # Check accuracy on test set
    final_acc, _ = solver.evaluate(eval_loader)
    print(f\"\\nFinal test accuracy: {final_acc:.1f}%\")


if __name__ == \"__main__\":
    main()

# ## Key Points for Sudoku-Specific Implementation
#
# ### 1. Data Augmentation
#
# Sudoku benefits from heavy augmentation because valid transformations don't change solution difficulty:
#
# ```python
# # 1000 augmentations per puzzle create ~1M samples from 1K originals
# augmented = SudokuDataset.augment_sudoku(puzzle, num_augmentations=1000)
# ```
#
# Valid Sudoku transformations:
# - **Band shuffling**: Permute rows within 3×3 horizontal bands
# - **Stack shuffling**: Permute columns within 3×3 vertical stacks
# - **Digit permutation**: Relabel digits 1-9 (but not 0 for empty cells)
# - **Transposition**: Flip across diagonal
#
# ### 2. Loss Function
#
# Use standard cross-entropy on predicted vs. ground truth tokens:
#
# ```python
# loss = CrossEntropyLoss(
#     logits.view(-1, 10),      # Flatten to (batch*81, 10)
#     solution.view(-1)          # Flatten to (batch*81,)
# )
# ```
#
# ### 3. Evaluation Metrics
#
# - **Cell accuracy**: % of correctly predicted digits
# - **Puzzle accuracy**: % of fully correct puzzles (all 81 cells right)
#
# ### 4. Test-Time Settings
#
# - Use **N_sup=16** supervision steps for maximum accuracy at test time
# - Use **N_sup=4-8** during training/evaluation for speed
# - Prediction: `argmax(logits)` for each cell
#
# ## Performance Expectations
#
# With proper implementation:
# - **Sudoku-Extreme (paper setup)**: 87.4% cell accuracy
# - **Training time**: <36 hours on 1×L40S GPU
# - **Model size**: 5M parameters (vs 27M for HRM)
#
# ## Extending Beyond Sudoku
#
# This same architecture works for:
# - **Mazes**: 30×30 grid, 2 classes (path/wall)
# - **ARC-AGI**: Variable sizes, complex transformations
# - **Pattern completion**: Predict missing elements
#
# Key adjustments:
# 1. Change `vocab_size` to output space (e.g., 2 for binary mazes)
# 2. Adjust `max_seq_len` to problem size
# 3. Design domain-specific data augmentations
# 4. Tune `H_cycles` and `L_cycles` for task difficulty
#
# ## References
#
# - **Original implementation**: https://github.com/SamsungSAILMontreal/TinyRecursiveModels
# - **Paper results**: 87.4% on Sudoku-Extreme dataset

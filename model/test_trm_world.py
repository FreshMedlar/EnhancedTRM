
import gymnasium as gym
import torch
import numpy as np
import ale_py
from PIL import Image
from torch.optim import AdamW
import wandb

# Import from local project structure
from trm_lucid import TinyRecursiveModel
from mlp_mixer_1d import MLPMixer1D

def preprocess(obs):
    """
    Resize observation to 32x32 and convert to grayscale.
    Returns: torch.Tensor of shape (1, 1024) with values 0-255 (Long)
    """
    img = Image.fromarray(obs)
    img = img.resize((32, 32), Image.BILINEAR)
    img = img.convert("L") # Grayscale
    
    # Flatten array to (1024,)
    arr = np.array(img).flatten()
    
    # Add batch dim -> (1, 1024)
    return torch.tensor(arr, dtype=torch.long).unsqueeze(0)

def train_world_model():
    # Helper to clean up wandb if it's already running (for repl/notebook usage)
    if wandb.run is not None:
        wandb.finish()
        
    wandb.init(project="EnhancedTRM")
    
    print("Initializing environment...")
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array') # Use rgb_array for speed
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    SEQ_LEN = 32 * 32 + 1 # 1024 (1024 pixels)
    assert SEQ_LEN == 1025
    
    # Initialize Model
    # We use MLPMixer1D as the backbone as in the test_trm.py example
    print("Initializing TRM...")
    network = MLPMixer1D(
        dim = 256,
        depth = 2,
        seq_len = SEQ_LEN
    )
    
    trm = TinyRecursiveModel(
        dim = 256,
        num_tokens = 260, # 0-255 pixel values + 4 actions (256-259)
        network = network
    ).to(device)

    optim = AdamW(trm.parameters(), lr = 1e-4)

    # Initial state
    print("Resetting Env...")
    obs, info = env.reset()
    current_seq_pixels = preprocess(obs).to(device)
    
    # Get initial recurrent states
    outputs, latents = trm.get_initial()
    outputs = outputs.to(device)
    latents = latents.to(device)

    # Training Loop
    episodes = 10
    for ep in range(episodes):
        done = False
        step_count = 0
        total_loss = 0
        
        while not done:
            # Random action for data collection
            action = env.action_space.sample()
            
            # Step
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Prepare inputs with Action
            # Map action 0-3 to 256-259
            action_token = torch.tensor([[256 + action]], device=device)
            input_full = torch.cat([current_seq_pixels, action_token], dim=1) 

            # Prepare next sequence
            next_seq_pixels = preprocess(obs_next).to(device)
            
            # Prepare label with dummy token for action slot (since we predict next state)
            dummy_token = torch.tensor([[0]], device=device)
            # Label should be: [next_frame, dummy] because we output 1025 tokens
            label_full = torch.cat([next_seq_pixels, dummy_token], dim=1)
            
            # TRM Forward Pass
            loss, losses, outputs, latents, pred, halt = trm(
                input_full, 
                outputs, 
                latents, 
                labels = label_full
            )
            
            # Optimization
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Detach states to truncate BPTT through time if needed, 
            # though TRM naturally detaches in forward return usually?
            # Looking at trm_lucid.py: 
            # outputs, latents = outputs.detach(), latents.detach()
            # It ALREADY detaches inside forward before returning.
            
            total_loss += loss.item()
            wandb.log({"step_loss": loss.item()})
            step_count += 1
            current_seq_pixels = next_seq_pixels
            
            if step_count % 50 == 0:
                print(f"Ep {ep} Step {step_count} Loss: {loss.item():.4f}")
                
            if step_count > 1000: # Short episodes for testing
                break
        
        avg_loss = total_loss / step_count
        print(f"Episode {ep} finished. Avg Loss: {avg_loss:.4f}")
        wandb.log({"episode": ep, "avg_loss": avg_loss})
        
        # Save model
        torch.save(trm.state_dict(), "trm_world_only.pth")
        
        # Initial state
        print("Resetting Env...")
        obs, info = env.reset()
        current_seq_pixels = preprocess(obs).to(device)
        outputs, _ = trm.get_initial()
        outputs = outputs.to(device)
        latents = latents.to(device)

    env.close()
    print("Training finished.")

def test():
    print("Initializing environment...")
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array') # Use rgb_array for speed
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    SEQ_LEN = 32 * 32  # 1024
    assert SEQ_LEN == 1024
    
    # Initialize Model
    # We use MLPMixer1D as the backbone as in the test_trm.py example
    print("Initializing TRM...")
    network = MLPMixer1D(
        dim = 256,
        depth = 2,
        seq_len = SEQ_LEN
    )
    
    trm = TinyRecursiveModel(
        dim = 256,
        num_tokens = 260, # 0-255 pixel values + 4 actions
        network = network,
        num_refinement_blocks=2,
        num_latent_refinements=2
    ).to(device)

    optim = AdamW(trm.parameters(), lr = 1e-4)

    # Initial state
    print("Resetting Env...")
    obs, info = env.reset()
    action = env.action_space.sample()

    print(action)
    current_seq = preprocess(obs).to(device)
    


if __name__ == "__main__":
    # test()
    train_world_model()

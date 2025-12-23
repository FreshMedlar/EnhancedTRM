import torch
import sys
import os

# Add model path
sys.path.append(os.path.join(os.getcwd(), 'model'))

from jepa import JEPAWorldModel

def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

model = JEPAWorldModel(latent_dim=128, num_actions=4, in_channels=1, img_size=84)

print(f"=== New Architecture ===")
print(f"Total Trainable Parameters: {count_parameters(model, trainable_only=True) / 1e6:.2f}M")
print(f"Total Parameters (incl. frozen): {count_parameters(model, trainable_only=False) / 1e6:.2f}M")
print(f"ViT Encoder (frozen): {count_parameters(model.encoder, trainable_only=False) / 1e6:.2f}M")
print(f"Temporal RWKV: {count_parameters(model.temporal, trainable_only=True) / 1e6:.2f}M")
print(f"Predictor: {count_parameters(model.predictor_net, trainable_only=True) / 1e6:.2f}M")
print(f"Reward Predictor: {count_parameters(model.reward_predictor, trainable_only=True) / 1e6:.2f}M")

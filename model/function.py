import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from action_model import ActionModel

def select_action(obs):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        # optionally, enrich obs_t with memory; here it's just a pass-through
        logits = action_model(obs_t)          # [1, act_dims]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()
    return action

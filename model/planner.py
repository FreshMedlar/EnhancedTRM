import torch


def plan_action(current_obs, world_model, reward_model, horizon=15, candidates=100):
    """
    Uses the World Model to 'dream' into the future.
    Finds the sequence of actions that leads to the highest Value (Objective).
    """
    world_model.eval()
    reward_model.eval()

    # 1. Expand current observation to match candidates batch size
    # Shape: [candidates, obs_dim]
    curr_states = torch.tensor(current_obs, dtype=torch.float32, device=device).repeat(candidates, 1)
    
    # 2. Generate random action sequences (The "What if?" scenarios)
    # Shape: [candidates, horizon, 1]
    # Random integers between 0 and act_dims
    action_seqs = torch.randint(0, act_dims, (candidates, horizon, 1), device=device).float()
    
    cumulative_rewards = torch.zeros(candidates, device=device)
    
    # 3. Simulate the future (The "Dream")
    with torch.no_grad():
        states = curr_states
        for t in range(horizon):
            actions = action_seqs[:, t] # Action at step t for all candidates
            
            # Predict next state
            next_states = world_model(states, actions)
            
            # Predict Value of this state (Is this a good objective?)
            # We sum up the values along the path
            values = reward_model(next_states).squeeze()
            cumulative_rewards += values
            
            states = next_states

    # 4. Pick the winner
    # Find the candidate index with the highest total value
    best_candidate_idx = torch.argmax(cumulative_rewards)
    
    # 5. Return the FIRST action of the best sequence
    best_first_action = action_seqs[best_candidate_idx, 0].item()
    
    world_model.train()
    reward_model.train()
    
    return int(best_first_action)

import gymnasium as gym
import time  # Optional: for frame rate control
import ale_py


env = gym.make('ALE/Breakout-v5', render_mode='human')  # Key change here
observation, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # Random actions for demo
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Optional: Add small delay for smoother visuals
    time.sleep(1.0 / 60.0)  # ~60 FPS

env.close()  # Always close to free resources

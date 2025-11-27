import gymnasium as gym
env = gym.make("LunarLander-v3")
# observation = (lander_x, lander_y, velocity_x, velocity_y, angle, ang_velocity, leg1, leg2)
observation, info = env.reset(seed=42)
for _ in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()


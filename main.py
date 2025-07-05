import gymnasium as gym
from gantry import Environment
import example

gym_env = gym.make("so-arm100")
env = Environment(gym_env, fps=10)
observation, info = env.reset(seed=42)

with env.render_settings(fps=10):
  for _ in range(2000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()

    if terminated or truncated:
      observation, info = env.reset()


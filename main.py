import gymnasium as gym
from gantry import Environment
import example

gym_env = gym.make("so-arm100")
env = Environment(gym_env)
env.render()


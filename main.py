import gymnasium as gym
from gantry import Environment

env = Environment(gym.make("so-arm100"))
env.render()


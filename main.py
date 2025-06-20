import gymnasium as gym
from gantry import Environment

env = Environment(gym.make("so-100"))
env.render()


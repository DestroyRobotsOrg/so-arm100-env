import gymnasium as gym
from gantry import Environment
import example

env = Environment(gym.make("so-arm100"))
env.render()


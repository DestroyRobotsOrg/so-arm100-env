import gymnasium as gym
from gantry import Environment, render_sphere
import example

gym_env = gym.make("so-arm100")
env = Environment(gym_env)
env.render()


import gymnasium as gym

from .so_100 import SOARM100Env

gym.register(id="so-arm100", entry_point=SOARM100Env)

from typing import Optional
import numpy as np
import mujoco
import gymnasium as gym


class SOARM100Env(gym.Env):
    def __init__(self, render_mode=None):
        self._model = mujoco.MjModel.from_xml_path(
            "mujoco_menagerie/trs_so_arm100/scene.xml"
        )
        self._data = mujoco.MjData(self._model)  # Initialize _data here as well
        self.action_space = gym.spaces.Discrete(6)
        # Define a more appropriate observation space based on the model's qpos
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self._data.qpos.shape, dtype=np.float64
        )
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
        self.render_mode = render_mode

        if self.render_mode == "rgb_array":
            self._render_width = 640
            self._render_height = 480
            self._renderer = mujoco.Renderer(
                self._model
            )  # , self._render_width, self._render_height)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset the data to initial state
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self._data.ctrl = action
        mujoco.mj_step(self._model, self._data)

        return self._get_obs(), 0, False, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            self._renderer.update_scene(self._data)

            return self._renderer.render()
        else:
            return None

    def _get_obs(self):
        return self._data.qpos

    def _get_info(self):
        return {"state": self._data.qpos}

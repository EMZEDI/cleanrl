# sparse_cartpole_wrapper.py
import gymnasium as gym
import numpy as np


class SparseCartpoleRewardWrapper(gym.Wrapper):
    """
    Replaces dm_control cartpole-swingup reward with a threshold-based sparse reward.
    threshold=0.0  → always reward 1 (fully dense approximation)
    threshold=0.95 → original dm_control sparse setting
    """
    def __init__(self, env, threshold: float = 0.95):
        super().__init__(env)
        self.threshold = threshold

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Re-read physics directly from the unwrapped dm_control env
        # obs[0] = cos(pole angle), obs[1] = sin(pole angle), obs[2] = cart pos
        # After FlattenObservation, dm_control cartpole obs order:
        # [position, velocity, cos(angle), sin(angle), angular_velocity]
        # We need cos(angle) for uprightness
        # Get it from the underlying physics via unwrapped env
        try:
            physics = self.env.unwrapped.physics
            upright = (physics.pole_angle_cosine() + 1) / 2
            cart_pos = physics.cart_position()
            # centering tolerance (mirrors dm_control logic)
            margin = 2.0
            centered_raw = max(0.0, 1.0 - abs(cart_pos) / margin)
            centered = (1 + centered_raw) / 2

            if self.threshold <= 0.0:
                # fully dense
                new_reward = upright * centered_raw
            else:
                new_reward = float(upright >= self.threshold and centered >= self.threshold)
        except Exception:
            # fallback: keep original reward
            new_reward = reward

        return obs, new_reward, terminated, truncated, info

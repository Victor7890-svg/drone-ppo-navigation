import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.target = np.array([110, 85.0, -3.0])
        self.max_steps = 2000
        self.current_step = 0
        self.prev_dist = None

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(-3, 2).join()
        self.current_step = 0
        self.prev_dist = None
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        pos = self.client.getMultirotorState().kinematics_estimated.position
        position = np.array([pos.x_val, pos.y_val, pos.z_val])
        to_target = self.target - position
        collision = float(self.client.simGetCollisionInfo().has_collided)
        return np.concatenate([position, to_target, [collision]]).astype(np.float32)

    def step(self, action):
        # 先获取当前距离决定速度
        current_obs = self._get_obs()
        current_pos = current_obs[:3]
        current_dist = np.linalg.norm(self.target[:2] - current_pos[:2])
        
        # 靠近终点时减速
        speed = 8.0 if current_dist > 15.0 else 3.0
        
        vx, vy = 0.0, 0.0
        if action == 0: vx = speed
        if action == 1: vx = -speed
        if action == 2: vy = speed
        if action == 3: vy = -speed

        self.client.moveByVelocityZAsync(
            vx, vy, -3, duration=0.3
        ).join()

        self.current_step += 1
        obs = self._get_obs()
        pos = obs[:3]
        collision = bool(self.client.simGetCollisionInfo().has_collided)
        dist = np.linalg.norm(self.target[:2] - pos[:2])

        # 新奖励函数：靠近目标加分，远离扣分
        reward = -0.1  # 时间惩罚
        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 2  # 靠近加分，远离扣分
        self.prev_dist = dist

        if collision:
            reward -= 100
        if dist < 8.0:
            reward += 200

        terminated = collision or dist < 3.0
        truncated = self.current_step >= self.max_steps

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

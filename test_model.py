from stable_baselines3 import PPO
from drone_env import DroneEnv
import numpy as np

env = DroneEnv()
model = PPO.load("drone_ppo_model", env=env)

n_episodes = 20  # 测试20轮
successes = 0
collisions = 0

for i in range(n_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    if reward > 100:
        successes += 1
        print(f"第{i+1}轮：✅ 成功到达终点")
    elif env.client.simGetCollisionInfo().has_collided:
        collisions += 1
        print(f"第{i+1}轮：💥 碰撞失败")
    else:
        print(f"第{i+1}轮：⏱ 超时未到达")

print(f"\n成功率：{successes}/{n_episodes} = {successes/n_episodes*100:.1f}%")
print(f"碰撞率：{collisions}/{n_episodes} = {collisions/n_episodes*100:.1f}%")

env.close()
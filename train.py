from stable_baselines3 import PPO
from drone_env import DroneEnv

env = DroneEnv()

# 加载上次的模型继续训练
model = PPO.load("drone_ppo_model", env=env)

print("继续训练！")
model.learn(total_timesteps=2000)
model.save("drone_ppo_model")
print("训练完成！")

env.close()
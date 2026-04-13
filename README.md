# Drone Urban Low-Altitude Obstacle Avoidance Navigation



An autonomous drone obstacle avoidance and navigation system based on deep reinforcement learning (PPO algorithm), implemented in AirSim + Unreal Engine 4 simulation environment.



## Results



- Final test success rate: \*\*20/20 = 100%\*\*

- Collision rate: \*\*0%\*\*

- Navigation distance: \~138 meters (including turns)



## Tech Stack



- Simulation: Microsoft AirSim + Unreal Engine 4.27

- RL Algorithm: PPO (Proximal Policy Optimization)

- Framework: Stable-Baselines3 + Gymnasium

- Language: Python 3.8



## File Structure


| Module | Description |
|-------|-------|
| Aerodynamic Core | Blade Element Theory (BET) engine computing thrust, torque, power, and Figure of Merit (FM). |
| Flight Simulator | 6‑DOF physics with waypoint navigation, battery discharge modeling, ground effect & motor lag. |  
| Live Dashboard | Real‑time altitude, RPM, thrust, power gauges with manual RPM override and auto‑flight controls. |  
| Parametric Design | Interactive modification of rotor radius, chord, blade count, mass, etc. with immediate feedback of RPM. |
| Data Recording | Automatic CSV logging of all flight parameters; replay any previous flight with speed control. |  
| 3D Visualization | OpenGL rendering of the eVTOL model, animated rotors, thrust cones, and altitude‑colored trail. |  
| Fault Injection | Simulate single‑rotor failure to evaluate emergency descent behavior (planned). |
| File | Description | 
|------|-------------|  

| `drone\_env.py` | Custom Gym environment with observation space, action space, and reward function |  

| `train.py` | PPO training script, supports loading existing model to continue training   

| `test\_model.py` | Model evaluation script, computes success rate and collision rate |



\## Setup



```bash

conda create -n uav\_rl python=3.8

conda activate uav\_rl

pip install numpy

pip install msgpack-rpc-python

pip install airsim stable-baselines3 gymnasium torch opencv-python

```



\## Usage



\*\*Train the model:\*\*

```bash

conda activate uav\_rl

python train.py

```



\*\*Test the model:\*\*

```bash

python test\_model.py

```



\## Training Strategy



Used \*\*Curriculum Learning\*\* to progressively increase task difficulty:



1\. Short-range straight navigation (20m)

2\. Medium-range straight navigation (40m → 60m)

3\. Added lateral offset (turning along Y-axis)

4\. Final long-range complex path (\~138m with turns)



\## Reward Function



\- Getting closer to target: `+ distance\_reduction × 2`

\- Time penalty per step: `-0.1`

\- Collision penalty: `-100`

\- Reaching target reward: `+200`



\## Requirements



\- GPU: NVIDIA GTX 1080 or above (RTX 3060+ recommended)

\- RAM: 16GB+

\- Storage: 100GB+

\- OS: Windows 10/11 x64


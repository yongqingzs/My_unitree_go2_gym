## NOTE
```bash
conda activate go1_gym

python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/export_alone.py --checkpoint XX
```

## base
- 1223
```txt
原版 go2 奖励
 --resume  --load_run Oct29_10-17-44_ --checkpoint 7500

NOTE:
    原地抖振现象比 policy1.pt 更为明显
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base --resume  --load_run Oct29_10-17-44_ --checkpoint 7500

python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/play.py --headless --task go1_trot --load_run Dec23_02-15-55_base --checkpoint 3800
```

- 1223-1
```txt
like 1223, but:
    randomize_motor_zero_offset = False
 --resume  --load_run Oct29_10-17-44_ --checkpoint 7500

NOTE: 抖振
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base1 --resume  --load_run Oct29_10-17-44_ --checkpoint 7500
```

- 0112
```txt
修复 random 后，重新跑

NOTE: 后期发散
```
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base2
```

- 0113
```txt
like resume, but:
    obs_buf = torch.cat((
        simple_cmd_input,  # 5 = 2D(sin cos) + 3D(vel_x, vel_y, aug_vel_yaw)
        self.obs_imu,#6 角速度，欧拉角XYZ
        self.obs_motor,#24
        self.actions,   # 12
    ), dim=-1)

NOTE: 直接不上升
```
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base3
```

- 0114
```python
class domain_rand:
    randomize_friction = True
    friction_range = [0.2,1.2]

    push_robots = True
    push_interval_s = 4
    max_push_vel_xy = 0.4
    max_push_ang_vel = 0.6

    randomize_base_mass = True
    added_base_mass_range = [-1,2]

    randomize_link_mass = True
    multiplied_link_mass_range = [0.9, 1.1]

    randomize_base_com = True
    added_base_com_range = [-0.05, 0.05]

    randomize_motor_strength = True
    motor_strength_range = [0.9, 1.1]

    randomize_kp = True
    kp_range = [0.9, 1.1]
    
    randomize_kd = True
    kd_range = [0.9, 1.1]

    randomize_motor_zero_offset = False
    motor_zero_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

    # range to contain the real joint armature 
    # old delay
    add_obs_latency = False # no latency for obs_action
    randomize_obs_motor_latency = False
    randomize_obs_imu_latency = False
    range_obs_motor_latency = [1, 3]
    range_obs_imu_latency = [1, 3]
    
    add_cmd_action_latency = False
    randomize_cmd_action_latency = False
    range_cmd_action_latency = [1, 3]

    # Lag timesteps for motor delay simulation (similar to HIMLoco)
    delay = False
    randomize_lag_timesteps = True
    lag_timesteps = 6  # Number of timesteps to delay (buffer size - 1)

class rewards:
    class scales:
        termination = -0.0
        tracking_lin_vel = 2.
        tracking_ang_vel = 2.
        lin_vel_z = -2.
        ang_vel_xy = -0.05
        orientation = -2.
        torques = -0.0001#
        dof_acc = -2.5e-7#-7
        collision = -1.
        action_rate = -0.01
        stand_still = -1.
        base_height=-5.
        trot=0.5
        feet_clearance=0.05 #feet clearance can increase for more
        default_hip_pos=-0.2
        default_pos=-0.1
        contact_without_command=1.

    only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
    tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
    soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
    soft_dof_vel_limit = 1.
    soft_torque_limit = 1.
    base_height_target = 0.29
    max_contact_force = 100. # forces above this value are penalized
    cycle_time=0.5
    target_foot_height=0.06  #feet height

```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go1_trot --max_iterations 5000 --seed 1 --num_envs 4096 --run_name base5

CUDA_VISIBLE_DEVICES=1 python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/export_policy.py --headless --task go1_trot --load_run Jan14_06-00-29_base5 --checkpoint 3400
```


- 0115
```txt
直接使用 go2_trot 训练:
    command_norm = torch.norm(self.commands[:, :3] * self.commands_scale, dim=1)
    mask = command_norm < 0.1
    sin_pos = torch.where(mask.unsqueeze(1), torch.zeros_like(sin_pos), sin_pos)
    cos_pos = torch.where(mask.unsqueeze(1), torch.zeros_like(cos_pos), cos_pos)
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 10000 --seed 1 --num_envs 4096 --run_name mask
```

- 0204
```txt
only origin
max_command = 3

NOTE: 实机还行，机身高度略有下降
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base1
```

- 0204-1
```txt
only origin
max_command = 3
but:
    tracking_ang_vel = 1.
    feet_clearance=0.01
    contact_without_command=0.3

NOTE: 不如 0204
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base2
```

- 0205
```txt
only origin
max_command = 3
but:
    cycle_time=0.333
NOTE: 有效
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base3
```

- 0205-1
```txt
only origin
max_command = 3
but:
    cycle_time=0.333
    feet_clearance=0.05
NOTE: 有效，表现比 0205 好
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base5
```

- 0206
```txt
only origin
max_command = 3
but:
    cycle_time=0.333
    feet_clearance=0.05
    stand_still = -0.5
    feet_clearance=0.03

NOTE: 静止时表现更差
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base6
```

- 0206-1
```txt
only origin
max_command = 3
but:
    cycle_time=0.333
    feet_clearance=0.03

NOTE: 待实机测试
```
```bash
python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base7
```

- 0207
```txt
only origin
max_command = 3
but:
    cycle_time=0.333
    feet_clearance=0.03
    frame_stack = 5 #action stack

NOTE: 启动更慢
```
```bash
CUDA_VISIBLE_DEVICES=1 python3 /workspace/My_unitree_go2_gym/legged_gym/scripts/train.py --headless --task go2_trot --max_iterations 20000 --seed 1 --num_envs 4096 --run_name base8
```

- 0208
```txt
only origin
max_command = 3
but:
    cycle_time=0.5
    feet_clearance=0.05
base9

NOTE: 待实机测试
```

- 0208-1
```txt
only origin
max_command = 3
but:
    cycle_time=0.5
    feet_clearance=0.03
base10

NOTE: 待实机测试
```

- 0210
```txt
only origin
max_command = 3
VelActorCritic
but:
    cycle_time=0.5
    feet_clearance=0.03
base11
```

import time
from isaacgym.torch_utils import *
import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from collections import deque
import math
from scipy.spatial.transform import Rotation as R
x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
x_vel_max, y_vel_max, yaw_vel_max = 1.5, 1.0, 3.0
def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd 
from pynput import keyboard

def key_callback(key):
    global x_vel_cmd, y_vel_cmd, yaw_vel_cmd
    try:
        if key.char == '6':
            x_vel_cmd += 0.3
        elif key.char == '7':
            x_vel_cmd -= 0.3
        elif key.char == '8':
            y_vel_cmd += 0.3
        elif key.char == '9':
            y_vel_cmd -= 0.3
        elif key.char == '-':
            yaw_vel_cmd += 0.5
        elif key.char == '=':
            yaw_vel_cmd -= 0.5
        elif key.char == '1':
            x_vel_cmd=0
            y_vel_cmd=0
            yaw_vel_cmd=0
        print(f"Updated velocities: vx={x_vel_cmd}, vy={y_vel_cmd}, dyaw={yaw_vel_cmd}")
    except AttributeError:
        pass
listener = keyboard.Listener(on_press=key_callback)
listener.start()
def quaternion_to_euler_array(quat):
    # Ensure quaternion is in the correct format [x, y, z, w]
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    # Pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)
    
    # Yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    # Returns roll, pitch, yaw in a NumPy array in radians
    return np.array([roll_x, pitch_y, yaw_z])

def get_obs(data,model):
    '''Extracts an observation from the mujoco data structure
    '''
    # print(data.qpos.astype(np.double).shape,data.qvel.astype(np.double).shape)
    q = data.qpos[7:19].astype(np.double)
    dq = data.qvel[6:].astype(np.double)
    quat = data.qpos[3:7].astype(np.double)[[1, 2, 3, 0]]
    r = R.from_quat(quat)
    v = r.apply(data.qvel[:3], inverse=True).astype(np.double)  # In the base frame
    omega = data.qvel[3:6].astype(np.double)
    gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    base_pos = data.qpos[0:3].astype(np.double)
    foot_positions = []
    # foot_forces = data.cfrc_ext[0][2].copy().astype(np.double)
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        # print(body_name)
        if 'foot' in body_name: 
            # print(body_name)
            foot_positions.append(data.xpos[i][2].copy().astype(np.double))
            foot_forces = data.cfrc_ext[i][2].copy().astype(np.double)
    return (q, dq, quat, v, omega, base_pos, foot_positions,foot_forces)
if __name__ == "__main__":
    # get config file name from command line
    import argparse

    config_file ="go2_MoB.yaml"
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        lin_vel_scale = config["lin_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d,key_callback=key_callback) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        hist_obs = deque()
        for _ in range(10):
                hist_obs.append(np.zeros([1, 47], dtype=np.double))
        while viewer.is_running() and time.time() - start < 1000000:
            step_start = time.time()
            if  (time.time() - start<5):
                tau = pd_control(default_angles, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
            if  (time.time() - start>5):
                tau = pd_control(target_q, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                # print(tau)
            # mj_step can be replaced with code that also evalua
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            q, dq, quat, v, omega, base_pos, foot_positions, foot_forces = get_obs(d,m)
            target_dq = np.zeros((num_actions), dtype=np.double)
            eu_ang = quaternion_to_euler_array(quat)
            eu_ang[eu_ang > math.pi] -= 2 * math.pi
            if counter % control_decimation == 0:
                # Apply control signal here.
                obs = np.zeros([1, 47], dtype=np.float32)
                # create observation
                period = 0.5
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                # print("sin_phase: ", sin_phase)
                obs[0, 0] = sin_phase
                obs[0, 1] = cos_phase
                obs[0, 2] = x_vel_cmd * lin_vel_scale
                obs[0, 3] = y_vel_cmd * lin_vel_scale
                obs[0, 4] = yaw_vel_cmd * ang_vel_scale
                obs[0, 5:8] = omega*ang_vel_scale
                obs[0, 8:11] = eu_ang

                obs[0, 11:23] = (q - default_angles) * dof_pos_scale
                obs[0, 23:35] = dq * dof_vel_scale
                obs[0, 35:47] = action
                hist_obs.append(obs)
                hist_obs.popleft()

                policy_input = np.zeros([1, 470], dtype=np.float32)
                for i in range(10):
                    policy_input[0, i * 47 : (i + 1) * 47] = hist_obs[i][0, :]

                action[:] = policy(torch.tensor(policy_input))[0].detach().numpy()
                action = np.clip(action, -100,100)
                target_q = action * action_scale
                # with open("/home/zju/YuSongmin/RL_Leggedgym/unitree_rl_gym-main/deploy/deploy_mujoco/simulation_data.txt", "a+") as file:
                #     file.write(str(omega[0])+","+str(omega[1])+","+str(omega[2])+"\n")
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

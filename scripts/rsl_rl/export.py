import os
import torch
import numpy as np
import pandas as pd
import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

# 配置参数
TASK_NAME = "sk-Robot1-v0"
NUM_STEPS = 500  # 500步 * 0.02s = 10秒
CMD_VX = 0.5     # 恒定测试速度 0.5m/s

def run_evaluation(mode, env, env_wrapper, policy=None):
    """
    mode: "ours" (RL), "passive" (被动), "active_pid" (传统主动)
    """
    obs, _ = env_wrapper.get_observations()
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    
    data_log = {
        "time": [], "pitch": [], "roll": [],
        "cmd_vx": [], "actual_vx": [], "z_accel": []
    }
    
    # 强制覆盖指令 (0.5m/s 前进，0转角，0航向)
    fixed_cmd = torch.tensor([[CMD_VX, 0.0, 0.0]], device=env.unwrapped.device)
    
    for step in range(NUM_STEPS):
        # 强制下发测试指令
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        if hasattr(cmd_term, 'command'):
            cmd_term.command[:] = fixed_cmd
        else:
            cmd_term.vel_command_b[:] = fixed_cmd
            
        # ---------------------------------------------------------
        # 动作计算路由 (全身协同 vs 解耦控制)
        # ---------------------------------------------------------
        if mode == "ours":
            # 1. 我们的方法：RL 神经网络端到端输出 (轮速 + 腿长)
            with torch.inference_mode():
                actions = policy(obs)
        else:
            # 2. 传统解耦基线方法
            actions = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device)
            # 2.1 传统轮速运动学控制 (直接透传期望速度)
            # 因为 base_scale=(1.0, 1.0)，动作 0.5 直接对应 0.5m/s
            actions[:, 0] = CMD_VX 
            actions[:, 1] = 0.0
            
            # 2.2 传统悬挂控制
            if mode == "passive":
                # 被动：全部输出 0，代表腿部保持在默认零位
                actions[:, 2:6] = 0.0 
                
            elif mode == "active_pid":
                # 传统主动：基于 Pitch 的启发式 PID
                root_quat = robot.data.root_quat_w
                _, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
                
                Kp = 3.0 # PID 比例系数 (可根据你的机器人微调)
                # 车头抬起 (Pitch > 0) 时，前腿缩短(负动作)，后腿伸长(正动作)
                actions[:, 2] = torch.clamp(-pitch * Kp, -1.0, 1.0) # LF
                actions[:, 3] = torch.clamp(-pitch * Kp, -1.0, 1.0) # RF
                actions[:, 4] = torch.clamp(pitch * Kp, -1.0, 1.0)  # LB
                actions[:, 5] = torch.clamp(pitch * Kp, -1.0, 1.0)  # RB
                
        # 执行动作
        obs, _, _, _ = env_wrapper.step(actions)
        
        # ---------------------------------------------------------
        # 记录数据
        # ---------------------------------------------------------
        root_quat = robot.data.root_quat_w
        roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
        z_accel = robot.data.root_lin_acc_w[:, 2] 
        actual_vx = robot.data.root_lin_vel_b[:, 0]
        
        data_log["time"].append(step * dt)
        data_log["pitch"].append(torch.rad2deg(pitch[0]).item())
        data_log["roll"].append(torch.rad2deg(roll[0]).item())
        data_log["cmd_vx"].append(CMD_VX)
        data_log["actual_vx"].append(actual_vx[0].item())
        data_log["z_accel"].append(z_accel[0].item())
        
    df = pd.DataFrame(data_log)
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_run", type=str, required=True, help="你的 RL 模型文件夹名")
    parser.add_argument("--checkpoint", type=str, default="model.pt")
    args = parser.parse_args()

    # 1. 环境初始化
    env_cfg = parse_env_cfg(TASK_NAME, device="cuda:0", num_envs=1)
    env_cfg.scene.terrain.terrain_generator.seed = 42 # ★ 极度关键：固定地形种子，确保对比公平
    env_cfg.scene.terrain.terrain_generator.curriculum = False # 关闭课程，强制最高难度波浪
    
    env = gym.make(TASK_NAME, cfg=env_cfg)
    env_wrapper = RslRlVecEnvWrapper(env)
    
    # 2. 加载 RL 模型
    agent_cfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    checkpoint_path = os.path.join(os.path.abspath(log_root_path), args.load_run, args.checkpoint)
    
    ppo_runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=None, device=env_cfg.sim.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    print("===========================================")
    print("Running Baseline 1: Passive + Kinematic...")
    df_passive = run_evaluation("passive", env, env_wrapper)
    df_passive["method"] = "Passive Decoupled"
    
    # 必须重置环境，确保每次从同一个起点出发
    env_wrapper.reset()
    
    print("Running Baseline 2: Active PID + Kinematic...")
    df_pid = run_evaluation("active_pid", env, env_wrapper)
    df_pid["method"] = "Traditional Active PID"
    
    env_wrapper.reset()
    
    print("Running Ours: Integrated RL Whole-Body Control...")
    df_ours = run_evaluation("ours", env, env_wrapper, policy)
    df_ours["method"] = "Ours (Integrated RL)"

    env.close()

    # 3. 合并数据并保存
    final_df = pd.concat([df_passive, df_pid, df_ours])
    save_path = "/home/wjc/robot1/data/paper_experiments_data1.csv"
    final_df.to_csv(save_path, index=False)
    print(f"\n✅ All experiments finished! Data saved to {save_path}")

if __name__ == "__main__":
    main()
import argparse
import os
import torch
import numpy as np
import pandas as pd

# ==========================================
# 1. 必须先启动 AppLauncher，然后才能导入其他 Isaac Lab 模块
# ==========================================
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Paper Evaluation Baseline")
parser.add_argument("--task", type=str, default="sk-Robot1-v0", help="Task name")
parser.add_argument("--load_run", type=str, required=True, help="你的 RL 模型文件夹名")
parser.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint filename")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True  # 后台静默运行，跑数据更快

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==========================================
# 2. 引擎启动完毕，现在可以安全导入依赖了
# ==========================================
import gymnasium as gym
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry

# 导入你的自定义任务注册 (确保 sk-Robot1-v0 能被找到)
import robot1.tasks

import rsl_rl.runners.on_policy_runner
from robot1.tasks.manager_based.MY_EVN.agents.cnn import CNNActorCriticRecurrent
rsl_rl.runners.on_policy_runner.CNNActorCriticRecurrent = CNNActorCriticRecurrent

# 测试常量
CMD_VX = 0.5
NUM_STEPS = 500  # 10秒

def run_evaluation(mode, env, env_wrapper, policy=None):
    """运行测试并返回数据字典"""
    obs, _ = env_wrapper.get_observations()
    robot = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    
    data_log = {
        "time": [], "pitch": [], "roll": [],
        "cmd_vx": [], "actual_vx": [], "z_accel": []
    }
    
    # 强制覆盖指令 (0.5m/s 前进)
    fixed_cmd = torch.tensor([[CMD_VX, 0.0, 0.0]], device=env.unwrapped.device)
    
    # ==========================================
    # 🌟 新增：为传统 Baseline 初始化 PI 控制器的积分项
    # ==========================================
    integral_error_v = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    
    for step in range(NUM_STEPS):
        # 强制下发测试指令
        cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        if hasattr(cmd_term, 'command'):
            cmd_term.command[:] = fixed_cmd
        else:
            cmd_term.vel_command_b[:] = fixed_cmd
            
        # --- 动作计算路由 ---
        if mode == "ours":
            with torch.inference_mode():
                actions = policy(obs)
        else:
            actions = torch.zeros((env.unwrapped.num_envs, 6), device=env.unwrapped.device)
            
            # 获取当前实际机身速度
            actual_vx = robot.data.root_lin_vel_b[:, 0]
            
            if mode == "passive":
                # ==========================================
                # 🌟 改进场景 1：强大的闭环 PI 速度追踪器 (Strong Baseline)
                # ==========================================
                # 被动悬挂：腿部保持在默认零位
                actions[:, 2:6] = 0.0 
                
                # 计算速度误差
                error_v = CMD_VX - actual_vx
                integral_error_v += error_v * dt
                
                # PI 参数 (你可以微调这两个值)
                Kp_v = 1.5  # 比例增益：误差越大，油门踩得越深
                Ki_v = 0.5  # 积分增益：消除持续的稳态打滑误差
                
                # 前馈 (CMD_VX) + PI 反馈补偿
                v_out = CMD_VX + Kp_v * error_v + Ki_v * integral_error_v
                
                # 限制输出范围，防止电机指令爆炸 (假设你的 base_scale 是 1.0)
                actions[:, 0] = torch.clamp(v_out, -1.0, 1.0)
                actions[:, 1] = 0.0
            elif mode == "active_pid":
                # ==========================================
                # 场景 2：传统主动 PD 悬挂 + 速度 PI 控制 (Strong Baseline)
                # ==========================================
                
                # 1. 速度 PI 控制 (追踪 0.5m/s)
                error_v = CMD_VX - actual_vx
                integral_error_v += error_v * dt
                v_out = CMD_VX + 1.5 * error_v + 0.5 * integral_error_v
                actions[:, 0] = torch.clamp(v_out, -1.0, 1.0)
                actions[:, 1] = 0.0
                
                # 2. 悬挂姿态 PD 控制
                root_quat = robot.data.root_quat_w
                _, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
                
                # 获取机身在局部坐标系下的角速度 (Y轴对应 Pitch Rate)
                pitch_rate = robot.data.root_ang_vel_b[:, 1]
                
                # PD 参数 (你可以根据实际震荡情况微调 Kd)
                Kp_pitch = 3.0   # 比例增益 (P)：对抗当前的倾斜角度
                Kd_pitch = 0.5   # 微分增益 (D)：对抗倾斜的趋势(角速度)，起到减震器/阻尼的作用
                
                # PD 联合输出计算
                pd_output = pitch * Kp_pitch + pitch_rate * Kd_pitch
                
                # 车头抬起 (Pitch > 0) 时，前腿缩短(负动作)，后腿伸长(正动作)
                actions[:, 2] = torch.clamp(-pd_output, -1.0, 1.0) # LF (左前)
                actions[:, 3] = torch.clamp(-pd_output, -1.0, 1.0) # RF (右前)
                actions[:, 4] = torch.clamp(pd_output, -1.0, 1.0)  # LB (左后)
                actions[:, 5] = torch.clamp(pd_output, -1.0, 1.0)  # RB (右后)
                
        # 执行动作
        obs, _, _, _ = env_wrapper.step(actions)
        
        # --- 记录数据 ---
        root_quat = robot.data.root_quat_w
        roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
        
        # 🌟 修复：使用 body_lin_acc_w，并取第 0 个 body（底盘）的 Z 轴 (索引 2)
        z_accel = robot.data.body_lin_acc_w[:, 0, 2] 
        
        actual_vx = robot.data.root_lin_vel_b[:, 0]
        
        data_log["time"].append(step * dt)
        data_log["pitch"].append(torch.rad2deg(pitch[0]).item())
        data_log["roll"].append(torch.rad2deg(roll[0]).item())
        data_log["cmd_vx"].append(CMD_VX)
        data_log["actual_vx"].append(actual_vx[0].item())
        data_log["z_accel"].append(z_accel[0].item())
        
    return pd.DataFrame(data_log)


def main():
    # 1. 环境初始化
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1)
    env_cfg.scene.terrain.terrain_generator.seed = 42 # 固定地形
    env_cfg.scene.terrain.terrain_generator.curriculum = False 
    
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_wrapper = RslRlVecEnvWrapper(env)
    
    # 2. 加载 RL 模型
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    checkpoint_path = os.path.join(os.path.abspath(log_root_path), args_cli.load_run, args_cli.checkpoint)
    
    ppo_runner = OnPolicyRunner(env_wrapper, agent_cfg.to_dict(), log_dir=None, device=env_cfg.sim.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    print("===========================================")
    print("Running Baseline 1: Passive + Kinematic...")
    df_passive = run_evaluation("passive", env, env_wrapper)
    df_passive["method"] = "Passive Decoupled"
    
    env_wrapper.reset()
    
    print("Running Baseline 2: Active PID + Kinematic...")
    df_pid = run_evaluation("active_pid", env, env_wrapper)
    df_pid["method"] = "Traditional Active PID"
    
    env_wrapper.reset()
    
    print("Running Ours: Integrated RL Whole-Body Control...")
    df_ours = run_evaluation("ours", env, env_wrapper, policy)
    df_ours["method"] = "Ours (Integrated RL)"

    # ==========================================
    # 🌟 关键修复：在关闭环境前，先保存数据！
    # ==========================================
        #==========================================
    # 🌟 关键修复：在关闭环境前，先保存数据到指定目录！
    # ==========================================
    print("\n[INFO] Saving data before closing simulation...")
    final_df = pd.concat([df_passive, df_pid, df_ours])
    
    # 指定你的绝对保存路径
    save_dir = "/home/wjc/robot1/data"
    
    # 安全检查：如果 data 文件夹不存在，就自动创建它
    os.makedirs(save_dir, exist_ok=True)
    
    # 拼接最终的文件路径
    save_path = os.path.join(save_dir, "paper_experiments_data.csv")
    
    # 保存为 CSV
    final_df.to_csv(save_path, index=False)
    print(f"✅ All experiments finished! Data strictly saved to: {save_path}")
    
    # ==========================================
    # 数据保存安全后，再执行危险的关闭操作
    # ==========================================
    try:
        env.close()
        simulation_app.close()
    except Exception as e:
        print(f"[Warning] Environment close error (can be ignored): {e}")

if __name__ == "__main__":
    main()
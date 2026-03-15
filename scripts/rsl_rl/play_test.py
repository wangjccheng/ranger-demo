"""
Keyboard Control + Data Logging + Plotting
Fixed: Independent Leg Subplots with Dual Y-Axis for Action vs Actual Position (rad)
"""
import argparse
import sys
import os
import torch
import numpy as np
import carb
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime

from isaaclab.app import AppLauncher

# 1. 启动 Isaac Sim
parser = argparse.ArgumentParser(description="Keyboard Control & Plotting")
parser.add_argument("--task", type=str, default="sk-Robot1-v0", help="Task name")
parser.add_argument("--load_run", type=str, required=True, help="Run folder or timestamp")
parser.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint filename")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the generated plot")
parser.add_argument("--random_cmd", action="store_true", help="Enable random command mode")
parser.add_argument("--cmd_interval", type=float, default=5.0, help="Interval (seconds) to change random commands")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import isaaclab.utils.math as math_utils
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
import robot1.tasks  

from robot1.tasks.manager_based.MY_EVN.agents.cnn import CNNActorCriticRecurrent
import rsl_rl.runners.on_policy_runner as on_policy_runner
on_policy_runner.CNNActorCriticRecurrent = CNNActorCriticRecurrent

# --- 键盘控制器 ---
class KeyboardController:
    def __init__(self, speed_scale=1.0, rot_scale=1.0):
        self.input = carb.input.acquire_input_interface()
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        self.sub = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key_event)
        
        self.current_vel = np.array([0.0, 0.0, 0.0]) 
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.speed_scale = speed_scale
        self.rot_scale = rot_scale
        self.stop_requested = False 
        self.reset_requested = False

    def _on_key_event(self, event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            if event.input == carb.input.KeyboardInput.W: self.target_vel[0] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.S: self.target_vel[0] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.A: self.target_vel[2] = self.rot_scale
            elif event.input == carb.input.KeyboardInput.D: self.target_vel[2] = -self.rot_scale
            elif event.input == carb.input.KeyboardInput.Q: self.target_vel[1] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.E: self.target_vel[1] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.ESCAPE: self.stop_requested = True
            elif event.input == carb.input.KeyboardInput.R: self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]: self.target_vel[0] = 0.0
            elif event.input in [carb.input.KeyboardInput.Q, carb.input.KeyboardInput.E]: self.target_vel[1] = 0.0
            elif event.input in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]: self.target_vel[2] = 0.0

def main():
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # 强制关闭域随机化与航向角控制
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        for attr in dir(env_cfg.events):
            if not attr.startswith("__"):  
                setattr(env_cfg.events, attr, None)
                
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0  
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0   
        env_cfg.commands.base_velocity.resampling_time_range = (99999.0, 99999.0) 
        
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if hasattr(agent_cfg, "to_dict"): agent_cfg_dict = agent_cfg.to_dict()
    else: agent_cfg_dict = agent_cfg

    if os.path.exists(args_cli.load_run):
        resume_path = os.path.abspath(args_cli.load_run)
    else:
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg_dict["experiment_name"])
        resume_path = os.path.join(os.path.abspath(log_root_path), args_cli.load_run)
    
    checkpoint_path = os.path.join(resume_path, args_cli.checkpoint)
    print(f"Loading checkpoint: {checkpoint_path}")

    ppo_runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=env_cfg.sim.device)
    ppo_runner.load(checkpoint_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
    
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    keyboard = KeyboardController(speed_scale=1.0, rot_scale=1.0)
    
    logs = {
        "time": [], "roll": [], "pitch": [], 
        "cmd_vx": [], "cmd_wz": [],
        "actual_vx": [], "actual_wz": [],  
        "action_vx": [], "action_wz": [],
        "action_lf": [], "action_rf": [], "action_lh": [], "action_rh": [],
        "pos_lf": [], "pos_rf": [], "pos_lh": [], "pos_rh": [] ,
        # 轮子的 Action (网络输出) 和 实际速度 (rad/s)
        "action_w_lf": [], "action_w_rf": [], "action_w_lh": [], "action_w_rh": [],
        "vel_w_lf": [], "vel_w_rf": [], "vel_w_lh": [], "vel_w_rh": []
    }
    
    obs, _ = env.reset()
    if hasattr(policy_nn, "reset"):
        with torch.inference_mode():
            force_dones = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
            policy_nn.reset(force_dones)

    robot_entity = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt
    sim_time = 0.0
    
    # ================= 修复版：强制精准查找真实的关节索引 =================
   # ================= 修复版：强制精准查找真实的关节索引 =================
    joint_names = robot_entity.data.joint_names 
    print("\n" + "🔥"*25)
    print(f"[DEBUG] 底层实际关节顺序:\n{joint_names}")
    print("🔥"*25 + "\n")
    
    # 腿部名字
    target_leg_names = ['g_lb', 'g_lf', 'g_rb', 'g_rf'] 
    # 🌟 轮子名字 (请根据终端打印的列表确认这些名字，如果不一致请修改这里)
    target_wheel_names = ['w_lb', 'w_lf', 'w_rb', 'w_rf'] 
    
    leg_indices = []
    wheel_indices = []
    
    for name in target_leg_names:
        if name in joint_names: leg_indices.append(joint_names.index(name))
        else: raise ValueError(f"🚨 找不到腿关节: {name}")
            
    for name in target_wheel_names:
        if name in joint_names: wheel_indices.append(joint_names.index(name))
        else: print(f"⚠️ [警告] 找不到轮子关节 '{name}'，请检查名称！图表将缺少轮子真实数据。")

    print(f"✅ 成功匹配腿关节: {leg_indices} | 轮关节: {wheel_indices}")
    # ==============================================================
    # ==============================================================
    # ================= 预热机制 (Warm-up) =================
    # 先空跑 60 步（约 1.2 秒），强制输入 0 指令，不记录数据，度过落地猛烈颠簸期
    print("\n[INFO] 正在执行防摔预热 (Warm-up phase)... 让机器人平稳落地")
    for _ in range(60):
        with torch.inference_mode():
            # 强制速度指令为0
            try:
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                if hasattr(cmd_term, 'command'): cmd_term.command[:] = 0.0
                if hasattr(cmd_term, 'vel_command_b'): cmd_term.vel_command_b[:] = 0.0
            except: pass
            
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"): policy_nn.reset(dones)
    print("✅ 预热完成，开始正式接受指令并记录数据！\n")
    
    sim_time = 0.0 # 重置计时器，让图表从 0 秒开始算
    while simulation_app.is_running():
        if keyboard.stop_requested:
            break
            
        if keyboard.reset_requested:
            print("\n[INFO] 环境已重置！")
            obs, _ = env.reset()
            if hasattr(policy_nn, "reset"):
                force_dones = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                with torch.inference_mode(): 
                    policy_nn.reset(force_dones)
            keyboard.reset_requested = False
            keyboard.current_vel = np.array([0.0, 0.0, 0.0])
            continue 

        with torch.inference_mode():
            smooth_factor = 0.05 
            keyboard.current_vel += (keyboard.target_vel - keyboard.current_vel) * smooth_factor
            user_vel = torch.tensor(keyboard.current_vel, dtype=torch.float32, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            
            try:
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                if hasattr(cmd_term, 'command'):
                    cmd_term.command[:, 0] = user_vel[:, 0]  
                    cmd_term.command[:, 1] = user_vel[:, 1]  
                    cmd_term.command[:, 2] = user_vel[:, 2]  
                if hasattr(cmd_term, 'vel_command_b'):
                    cmd_term.vel_command_b[:, 0] = user_vel[:, 0]
                    cmd_term.vel_command_b[:, 1] = user_vel[:, 1]
                    cmd_term.vel_command_b[:, 2] = user_vel[:, 2]
            except Exception:
                pass
                
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        # 收集物理状态
        root_quat = robot_entity.data.root_quat_w
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
        r_deg = torch.rad2deg(roll[0]).item()
        p_deg = torch.rad2deg(pitch[0]).item()
        
        actual_vx = robot_entity.data.root_lin_vel_b[0, 0].item()
        actual_wz = robot_entity.data.root_ang_vel_b[0, 2].item()
        
        #actual_joint_pos = robot_entity.data.joint_pos[0, 4:8].cpu().numpy()
        # 【修正】：使用动态匹配到的准确索引，只抓取真正的悬挂腿！
        actual_joint_pos = robot_entity.data.joint_pos[0, leg_indices].cpu().numpy()
        # 获取轮子的实际角速度 (rad/s)
        if len(wheel_indices) == 4:
            actual_wheel_vel = robot_entity.data.joint_vel[0, wheel_indices].cpu().numpy()
        else:
            actual_wheel_vel = [0.0, 0.0, 0.0, 0.0]
            
                # 第二步：计算理想的 2D 基础转速 (基础运动学)
        wl = (actions[0, 0].item() - actions[0, 1].item()* 0.5 * (0.68 / 2.0)) / 0.19
        wr = (actions[0, 0].item() + actions[0, 1].item() * 0.5 * (0.68 / 2.0)) / 0.19   
        action_scale = 0.3
        action_bias = 0.05
        # 记录数据
        logs["time"].append(sim_time)
        logs["roll"].append(r_deg)
        logs["pitch"].append(p_deg)
        logs["cmd_vx"].append(keyboard.current_vel[0]) 
        logs["cmd_wz"].append(keyboard.current_vel[2])
        logs["actual_vx"].append(actual_vx)
        logs["actual_wz"].append(actual_wz)
        # 轮动作 (假设 2,3,4,5 是轮子)
        logs["action_w_lh"].append(actions[0, 3].item()*0.1+wl)
        logs["action_w_lf"].append(actions[0, 2].item()*0.1+wl)
        logs["action_w_rh"].append(actions[0, 5].item()*0.1+wr)
        logs["action_w_rf"].append(actions[0, 4].item()*0.1+wr)
        
        logs["action_vx"].append(actions[0, 0].item())
        logs["action_wz"].append(actions[0, 1].item())
        logs["action_lf"].append(actions[0, 6].item()*action_scale+action_bias)
        logs["action_rf"].append(actions[0, 7].item()*action_scale+action_bias)
        logs["action_lh"].append(actions[0, 8].item()*action_scale+action_bias)
        logs["action_rh"].append(actions[0, 9].item()*action_scale+action_bias)

        logs["pos_lf"].append(actual_joint_pos[1])
        logs["pos_rf"].append(actual_joint_pos[3])
        logs["pos_lh"].append(actual_joint_pos[0])
        logs["pos_rh"].append(actual_joint_pos[2])
        # 轮实际速度 (顺序与 target_wheel_names 一致: w_lb, w_lf, w_rb, w_rf)
        logs["vel_w_lh"].append(actual_wheel_vel[0]) # lb
        logs["vel_w_lf"].append(actual_wheel_vel[1]) # lf
        logs["vel_w_rh"].append(actual_wheel_vel[2]) # rb
        logs["vel_w_rf"].append(actual_wheel_vel[3]) # rf
        
        sim_time += dt
        print(f"\r[Rec] T:{sim_time:.1f}s | Cmd Vx:{keyboard.current_vel[0]:4.1f} | Act Vx:{actual_vx:4.1f} | Pitch:{p_deg:4.1f}° | Act Pos(rad):{actual_joint_pos[0]:.2f}", end="")

    env.close()
    '''
    # ==========================================
    # 5. 绘图逻辑 (共 6 个子图，双 Y 轴对比)
    # ==========================================
    print("\n\nGenerating plots...")
    
    plt.style.use('ggplot')
    # 创建 6 行 1 列的画布，加长整体高度，共享 X 轴
    fig, axes = plt.subplots(6, 1, figsize=(10, 20), sharex=True)
    ax_att, ax_vel, ax_lf, ax_rf, ax_lh, ax_rh = axes
    
    # 子图1: 姿态角
    ax_att.plot(logs["time"], logs["pitch"], label='Pitch (deg)', color='orange', linewidth=1.5)
    ax_att.plot(logs["time"], logs["roll"], label='Roll (deg)', color='green', linewidth=1.5)
    ax_att.set_ylabel('Angle (deg)')
    ax_att.set_title('Robot Attitude Response')
    ax_att.legend(loc='upper right', ncol=2)
    ax_att.grid(True)

    # 子图2: 速度追踪
    ax_vel.plot(logs["time"], logs["cmd_vx"], label='Cmd Vx (m/s)', color='blue', linestyle='--', linewidth=2.0)
    ax_vel.plot(logs["time"], logs["cmd_wz"], label='Cmd Wz (rad/s)', color='red', linestyle='--', linewidth=2.0)
    ax_vel.plot(logs["time"], logs["actual_vx"], label='Actual Vx', color='dodgerblue', linewidth=1.5)
    ax_vel.plot(logs["time"], logs["actual_wz"], label='Actual Wz', color='salmon', linewidth=1.5)
    ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Velocity Tracking Performance')
    ax_vel.legend(loc='upper right', ncol=4, fontsize='small') 
    ax_vel.grid(True)
    
    # 定义画单条腿的函数（双 Y 轴）
    def plot_single_leg(ax, time, action, pos, title):
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # 左侧 Y 轴：神经网络输出的 Action
        color_act = 'tab:red'
        ax.plot(time, action, label='NN Action [-1, 1]', color=color_act, linewidth=1.5)
        ax.set_ylabel('Action [-1, 1]', color=color_act)
        ax.tick_params(axis='y', labelcolor=color_act)
        ax.grid(True, alpha=0.3)
        
        # 右侧 Y 轴：真实的物理旋转角度 (rad)
        ax_twin = ax.twinx()
        color_pos = 'tab:blue'
        # 物理位置用虚线（-.）表示，防止两根线完美重合时看不清
        ax_twin.plot(time, pos, label='Actual Pos (rad)', color=color_pos, linewidth=2.0, linestyle='-.')
        ax_twin.set_ylabel('Position (rad)', color=color_pos)
        ax_twin.tick_params(axis='y', labelcolor=color_pos)
        
        # 合并左右 Y 轴的图例
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        ax_twin.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize='small')

    # 子图3~6: 分别将 4 条腿独立渲染
    plot_single_leg(ax_lf, logs["time"], logs["action_lf"], logs["pos_lf"], 'Left Front (LF) Leg')
    plot_single_leg(ax_rf, logs["time"], logs["action_rf"], logs["pos_rf"], 'Right Front (RF) Leg')
    plot_single_leg(ax_lh, logs["time"], logs["action_lh"], logs["pos_lh"], 'Left Hind (LH) Leg')
    plot_single_leg(ax_rh, logs["time"], logs["action_rh"], logs["pos_rh"], 'Right Hind (RH) Leg')
    
    ax_rh.set_xlabel('Time (s)')
    '''
    # ==========================================
    # 5. 绘图逻辑 (5x2 并排双列布局)
    # ==========================================
    print("\n\nGenerating plots...")
    
    plt.style.use('ggplot')
    # 创建 5 行 2 列的画布。第1行用合并的方式画整体姿态和速度，下面4行左腿右轮
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(5, 2)
    
    ax_att = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    
    axes_legs = [fig.add_subplot(gs[i, 0]) for i in range(1, 5)]
    axes_wheels = [fig.add_subplot(gs[i, 1]) for i in range(1, 5)]
    
    # --- 子图1: 姿态角 ---
    ax_att.plot(logs["time"], logs["pitch"], label='Pitch (deg)', color='orange', linewidth=1.5)
    ax_att.plot(logs["time"], logs["roll"], label='Roll (deg)', color='green', linewidth=1.5)
    ax_att.set_ylabel('Angle (deg)')
    ax_att.set_title('Robot Attitude')
    ax_att.legend(loc='upper right', ncol=2)
    ax_att.grid(True)

    # --- 子图2: 速度追踪 ---
    ax_vel.plot(logs["time"], logs["cmd_vx"], label='Cmd Vx', color='blue', linestyle='--', linewidth=2.0)
    ax_vel.plot(logs["time"], logs["cmd_wz"], label='Cmd Wz', color='red', linestyle='--', linewidth=2.0)
    ax_vel.plot(logs["time"], logs["actual_vx"], label='Act Vx', color='dodgerblue', linewidth=1.5)
    ax_vel.plot(logs["time"], logs["actual_wz"], label='Act Wz', color='salmon', linewidth=1.5)
    ax_vel.set_ylabel('Velocity')
    ax_vel.set_title('Base Velocity Tracking')
    ax_vel.legend(loc='upper right', ncol=4, fontsize='small') 
    ax_vel.grid(True)
    
    # 辅助画图函数：双Y轴
    # === 专门为同单位数据准备的画图函数（单 Y 轴） ===
    def plot_tracking(ax, time, target, actual, title, ylabel):
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # 画目标值 (经过后处理的 Action)
        ax.plot(time, target, label='Target Pos', color='tab:red', linewidth=1.5, linestyle='--')
        
        # 画实际值 (真实物理反馈)
        ax.plot(time, actual, label='Actual Pos', color='tab:blue', linewidth=1.5, alpha=0.8)
        
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(True, alpha=0.3)

    # --- 左列：腿部位置 (统一为 rad，直接同轴对比！) ---
    plot_tracking(axes_legs[0], logs["time"], logs["action_lf"], logs["pos_lf"], 'Leg LF', 'Pos (rad)')
    plot_tracking(axes_legs[1], logs["time"], logs["action_rf"], logs["pos_rf"], 'Leg RF', 'Pos (rad)')
    plot_tracking(axes_legs[2], logs["time"], logs["action_lh"], logs["pos_lh"], 'Leg LH', 'Pos (rad)')
    plot_tracking(axes_legs[3], logs["time"], logs["action_rh"], logs["pos_rh"], 'Leg RH', 'Pos (rad)')
    axes_legs[3].set_xlabel('Time (s)')

    # --- 右列：轮子速度 (Action vs 真实角速度rad/s) ---
    plot_tracking(axes_wheels[0], logs["time"], logs["action_w_lf"], logs["vel_w_lf"], 'Wheel LF', 'Velocity (rad/s)')
    plot_tracking(axes_wheels[1], logs["time"], logs["action_w_rf"], logs["vel_w_rf"], 'Wheel RF', 'Velocity (rad/s)')
    plot_tracking(axes_wheels[2], logs["time"], logs["action_w_lh"], logs["vel_w_lh"], 'Wheel LH', 'Velocity (rad/s)')
    plot_tracking(axes_wheels[3], logs["time"], logs["action_w_rh"], logs["vel_w_rh"], 'Wheel RH', 'Velocity (rad/s)')
    axes_wheels[3].set_xlabel('Time (s)')
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args_cli.output_dir:
        os.makedirs(args_cli.output_dir, exist_ok=True)
        save_dir = args_cli.output_dir
    else:
        save_dir = resume_path
        
    save_path = os.path.join(save_dir, f"robot_states_{timestamp}.png")
    
    plt.tight_layout() 
    plt.savefig(save_path)
    print(f"\n[INFO] Plot saved to: {os.path.abspath(save_path)}")
    
    plt.show()
    simulation_app.close()
    
if __name__ == "__main__":
    main()

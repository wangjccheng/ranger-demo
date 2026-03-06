"""
Keyboard Control + Data Logging + Plotting
"""
import argparse
import sys
import os
import torch
import numpy as np
import carb
import gymnasium as gym
import matplotlib.pyplot as plt  # 引入绘图库
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
import robot1.tasks  # 注册任务

# 假设你的自定义网络保存在 robot1 里的某个文件，比如 robot1.modules
    # 请把下面这行的路径替换为你实际存放 CNNActorCriticRecurrent 的路径！
from robot1.tasks.manager_based.MY_EVN.agents.cnn import CNNActorCriticRecurrent
# 将自定义类塞入 RSL-RL runner 的作用域，使其可以通过字符串名字实例化
    # 核心修复逻辑：把你的自定义类强行塞进 rsl_rl 的 runner 命名空间里
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
        self.cmd_vel = np.array([0.0, 0.0, 0.0]) 
        self.speed_scale = speed_scale
        self.rot_scale = rot_scale
        self.stop_requested = False # 添加退出标志

    def _on_key_event(self, event, *args):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            if event.input == carb.input.KeyboardInput.W: self.cmd_vel[0] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.S: self.cmd_vel[0] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.A: self.cmd_vel[2] = self.rot_scale
            elif event.input == carb.input.KeyboardInput.D: self.cmd_vel[2] = -self.rot_scale
            elif event.input == carb.input.KeyboardInput.Q: self.cmd_vel[1] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.E: self.cmd_vel[1] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.ESCAPE: self.stop_requested = True # ESC 退出
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]: self.cmd_vel[0] = 0.0
            elif event.input in [carb.input.KeyboardInput.Q, carb.input.KeyboardInput.E]: self.cmd_vel[1] = 0.0
            elif event.input in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]: self.cmd_vel[2] = 0.0

def main():
    # 配置与环境初始化
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 路径处理
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

    keyboard = KeyboardController(speed_scale=1.0, rot_scale=0.5)
    
# === 数据记录容器 ===
    logs = {
        "time": [],
        "roll": [],
        "pitch": [],
        "cmd_vx": [],
        "cmd_wz": [],
        # 👇 新增：实际速度记录容器
        "actual_vx": [],  
        "actual_wz": [],  
        # 🌟 动作记录容器
        "action_vx": [],
        "action_wz": [],
        "action_lf": [],
        "action_rf": [],
        "action_lh": [],
        "action_rh": []
    }
    
    obs, _ = env.get_observations()
    robot_entity = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt # 获取仿真步长 (通常 0.02s 或 0.04s)
    sim_time = 0.0
    # 👇 新增：用于跟踪随机指令的计时器和当前指令
    last_random_time = -args_cli.cmd_interval # 设置为负数，确保在第 0 秒立刻生成第一条指令
    current_random_cmd = np.array([0.0, 0.0, 0.0])

    print("\n" + "="*50)
    print("Recording Data... Press ESC or Close Window to Finish")
    print("="*50 + "\n")

    while simulation_app.is_running():
        if keyboard.stop_requested:
            break
            
        with torch.inference_mode():
            if args_cli.random_cmd:
                # 如果开启了随机模式，检查是否到达了切换指令的时间点
                if sim_time - last_random_time >= args_cli.cmd_interval:
                    # 分别限制 vx 和 wz 的随机范围 (这里设为前向 +-1.5, 转向 +-1.0，可自调)
                    rand_vx = np.random.uniform(-1.0, 1.0)
                    rand_wz = np.random.uniform(-0.5, 0.5)
                    current_random_cmd = np.array([rand_vx, 0.0, rand_wz])
                    last_random_time = sim_time
                    # 在终端打印出新生成的指令
                    print(f"\n[Auto Command] T:{sim_time:.1f}s | New Target -> Vx: {rand_vx:.2f} m/s, Wz: {rand_wz:.2f} rad/s")
                
                active_cmd = current_random_cmd
            else:
                # 否则继续使用键盘输入的指令
                active_cmd = keyboard.cmd_vel
                
            # 1. 覆盖指令 (★ 把这里传入的 keyboard.cmd_vel 换成 active_cmd)
            user_vel = torch.tensor(active_cmd, dtype=torch.float32, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            try:
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                # 兼容 Isaac Lab 的不同版本命名
                if hasattr(cmd_term, 'command'):
                    cmd_term.command[:, 0] = user_vel[:, 0]  # vx
                    cmd_term.command[:, 1] = user_vel[:, 1]  # vy
                    
                    # 只有当有转向输入时，才覆盖系统的 wz 指令
                    if active_cmd[2] != 0.0:  # ★ 改为 active_cmd
                        cmd_term.command[:, 2] = user_vel[:, 2] 
                        
                elif hasattr(cmd_term, 'vel_command_b'):
                    cmd_term.vel_command_b[:, 0] = user_vel[:, 0]
                    cmd_term.vel_command_b[:, 1] = user_vel[:, 1]
                    if active_cmd[2] != 0.0:  # ★ 改为 active_cmd
                        cmd_term.vel_command_b[:, 2] = user_vel[:, 2]
                else:
                    print(f"警告: 找不到 command 属性。可用属性为: {dir(cmd_term)}")
                    
            except Exception as e:
                print(f"覆盖指令时发生错误: {e}")
                
            # 2. 推理与步进
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            # 3. 获取数据
            root_quat = robot_entity.data.root_quat_w
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
            r_deg = torch.rad2deg(roll[0]).item()
            p_deg = torch.rad2deg(pitch[0]).item()
            
            # 👇 新增：获取基座标系下的实际前进线速度 (Vx) 和实际偏航角速度 (Wz)
            actual_vx = robot_entity.data.root_lin_vel_b[0, 0].item()
            actual_wz = robot_entity.data.root_ang_vel_b[0, 2].item()
            
            # 4. === 记录数据 ===
            logs["time"].append(sim_time)
            logs["roll"].append(r_deg)
            logs["pitch"].append(p_deg)
            
            # ★ 记录实际生效的指令，而不是固定记录键盘指令
            logs["cmd_vx"].append(active_cmd[0]) 
            logs["cmd_wz"].append(active_cmd[2])
            
            logs["actual_vx"].append(actual_vx)
            logs["actual_wz"].append(actual_wz)
            
            # 👇 补回上一轮不小心漏掉的动作记录代码
            logs["action_vx"].append(actions[0, 0].item())
            logs["action_wz"].append(actions[0, 1].item())
            logs["action_lf"].append(actions[0, 2].item())
            logs["action_rf"].append(actions[0, 3].item())
            logs["action_lh"].append(actions[0, 4].item())
            logs["action_rh"].append(actions[0, 5].item())
            
            sim_time += dt

            # 👇 修改：将终端打印信息升级，加入目标与实际的对比
            print(f"\r[Rec] T:{sim_time:.1f}s | Cmd (Vx:{active_cmd[0]:5.2f} Wz:{active_cmd[2]:5.2f})| Act (Vx:{actual_vx:5.2f} Wz:{actual_wz:5.2f}) | Pitch:{p_deg:5.1f}°", end="")

    env.close()
    
    # === 绘图逻辑 (在仿真关闭后运行) ===
    print("\n\nGenerating plots...")
    
    # 🌟 修改绘图布局：变为 3 个子图
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 子图1: 姿态角
    ax1.plot(logs["time"], logs["pitch"], label='Pitch (deg)', color='orange', linewidth=1.5)
    ax1.plot(logs["time"], logs["roll"], label='Roll (deg)', color='green', linewidth=1.5)
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Robot Attitude Response')
    ax1.legend()
    ax1.grid(True)

# 子图2: 键盘/随机指令、实际速度与模型输出对比
    # 指令用粗虚线表示
    ax2.plot(logs["time"], logs["cmd_vx"], label='Cmd Vx (m/s)', color='blue', linestyle='--', linewidth=2.0)
    ax2.plot(logs["time"], logs["cmd_wz"], label='Cmd Wz (rad/s)', color='red', linestyle='--', linewidth=2.0)
    
    # 实际速度用鲜艳的实线表示
    ax2.plot(logs["time"], logs["actual_vx"], label='Actual Vx', color='dodgerblue', linewidth=1.5)
    ax2.plot(logs["time"], logs["actual_wz"], label='Actual Wz', color='salmon', linewidth=1.5)
    
    # 动作输出作为参考，用细线+半透明度防止画面太乱
    ax2.plot(logs["time"], logs["action_vx"], label='Action Vx', color='cyan', linewidth=1.0, alpha=0.6)
    ax2.plot(logs["time"], logs["action_wz"], label='Action Wz', color='magenta', linewidth=1.0, alpha=0.6)
    
    ax2.set_ylabel('Velocity / Action')
    ax2.set_title('Velocity Tracking Performance (Command vs Actual)')
    # 把图例分成 3 列放在右上方，避免挡住曲线
    ax2.legend(loc='upper right', ncol=3, fontsize='small') 
    ax2.grid(True)
    
    # 子图3: 🌟 腿部悬挂动作 (LF, RF, LH, RH)
    ax3.plot(logs["time"], logs["action_lf"], label='Leg LF', color='red', linewidth=1.0)
    ax3.plot(logs["time"], logs["action_rf"], label='Leg RF', color='blue', linewidth=1.0)
    ax3.plot(logs["time"], logs["action_lh"], label='Leg LH', color='green', linewidth=1.0)
    ax3.plot(logs["time"], logs["action_rh"], label='Leg RH', color='purple', linewidth=1.0)
    ax3.set_ylabel('Leg Position Action')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Active Suspension Outputs')
    ax3.legend()
    ax3.grid(True)
    
    # 保存图片

    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 👇 新增路径判断逻辑
    if args_cli.output_dir:
        # 如果指定了路径，先确保该文件夹存在，不存在则自动创建
        os.makedirs(args_cli.output_dir, exist_ok=True)
        save_dir = args_cli.output_dir
    else:
        # 如果没指定，默认保存在 load_run 所在的模型文件夹里
        save_dir = resume_path
        
    save_path = os.path.join(save_dir, f"robot_states_{timestamp}.png")
    
    plt.tight_layout() # 防止图表重叠
    plt.savefig(save_path)
    print(f"[INFO] Plot saved to: {os.path.abspath(save_path)}")
    
    
    
    
    # 显示图片
    plt.show()
    
    simulation_app.close()
    
    

if __name__ == "__main__":
    main()
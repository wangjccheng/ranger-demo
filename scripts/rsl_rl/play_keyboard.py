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
        #self.cmd_vel = np.array([0.0, 0.0, 0.0]) 
        # 【修改】区分当前实际下发的指令和键盘按下的目标指令
        self.current_vel = np.array([0.0, 0.0, 0.0]) 
        self.target_vel = np.array([0.0, 0.0, 0.0])
        self.speed_scale = speed_scale
        self.rot_scale = rot_scale
        self.stop_requested = False # 添加退出标志
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
    # 配置与环境初始化
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    # 【新增 1】：关闭域随机化干扰
    # 替换为以下代码：
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        for attr in dir(env_cfg.events):
            if not attr.startswith("__"):  # 忽略内置属性
                setattr(env_cfg.events, attr, None)
    # 【新增 2】：强制关闭航向角控制，确保我们通过键盘塞入的是纯粹的线速度/角速度
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False
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
    try:
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        policy_nn = ppo_runner.alg.actor_critic

    keyboard = KeyboardController(speed_scale=1.0, rot_scale=1.0)
    
    # === 数据记录容器 ===
    logs = {
        "time": [],
        "roll": [],
        "pitch": [],
        "cmd_vx": [],
        "cmd_wz": []
    }
    
    obs, _ = env.get_observations()
    robot_entity = env.unwrapped.scene["robot"]
    dt = env.unwrapped.step_dt # 获取仿真步长 (通常 0.02s 或 0.04s)
    sim_time = 0.0

    print("\n" + "="*50)
    print("Recording Data... Press ESC or Close Window to Finish")
    print("="*50 + "\n")

    while simulation_app.is_running():
        if keyboard.stop_requested:
            break
            
        if keyboard.reset_requested:
            print("\n[INFO] 收到手动重置指令！正在重置机器人...")
            obs, _ = env.reset()
            if hasattr(policy_nn, "reset"):
                force_dones = torch.ones(env.unwrapped.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                policy_nn.reset(force_dones)
            keyboard.reset_requested = False
            continue

        with torch.inference_mode():
            # 【新增】平滑过渡逻辑 (Exponential Moving Average)
            # smooth_factor 控制加速度大小。值越小（如 0.02），加速越柔和；值越大（如 0.5），加速越猛。
            # 你可以根据实际操控手感修改这个系数
            smooth_factor = 0.05 
            keyboard.current_vel += (keyboard.target_vel - keyboard.current_vel) * smooth_factor
            
            # 1. 覆盖指令 (注意这里使用的是平滑后的 current_vel，而不是直接用按键目标值)
            user_vel = torch.tensor(keyboard.current_vel, dtype=torch.float32, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
            
            try:
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                if hasattr(cmd_term, 'command'):
                    # 无条件覆盖指令
                    cmd_term.command[:, 0] = user_vel[:, 0]  
                    cmd_term.command[:, 1] = user_vel[:, 1]  
                    cmd_term.command[:, 2] = user_vel[:, 2]  
                elif hasattr(cmd_term, 'vel_command_b'):
                    cmd_term.vel_command_b[:, 0] = user_vel[:, 0]
                    cmd_term.vel_command_b[:, 1] = user_vel[:, 1]
                    cmd_term.vel_command_b[:, 2] = user_vel[:, 2]
            except Exception as e:
                pass
            
            # 2. 推理与步进
            actions = policy(obs)
            # 【重要修复】：提取 dones 信号
            obs, _, dones, _ = env.step(actions)
            
            # 【重要修复】：当环境重置时，同步重置 GRU 隐藏状态
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

            # 3. 获取数据
            root_quat = robot_entity.data.root_quat_w
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_quat)
            r_deg = torch.rad2deg(roll[0]).item()
            p_deg = torch.rad2deg(pitch[0]).item()
            
            # 4. === 记录数据 ===
            logs["time"].append(sim_time)
            logs["roll"].append(r_deg)
            logs["pitch"].append(p_deg)
            
            # 【修复这里】：把 cmd_vel 改为 current_vel
            logs["cmd_vx"].append(keyboard.current_vel[0])
            logs["cmd_wz"].append(keyboard.current_vel[2])
            
            sim_time += dt

            print(f"\r[Rec] T:{sim_time:.1f}s | Pitch:{p_deg:6.2f}° | Roll:{r_deg:6.2f}°", end="")

    env.close()
    
    # === 绘图逻辑 (在仿真关闭后运行) ===
    print("\n\nGenerating plots...")
    
    # 设置风格
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 子图1: 姿态角
    ax1.plot(logs["time"], logs["pitch"], label='Pitch (deg)', color='orange', linewidth=1.5)
    ax1.plot(logs["time"], logs["roll"], label='Roll (deg)', color='green', linewidth=1.5)
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Robot Attitude Response')
    ax1.legend()
    ax1.grid(True)

    # 子图2: 键盘指令
    ax2.plot(logs["time"], logs["cmd_vx"], label='Command Vx (m/s)', color='blue', linestyle='--')
    ax2.plot(logs["time"], logs["cmd_wz"], label='Command YawRate (rad/s)', color='red', linestyle='--')
    ax2.set_ylabel('Command Input')
    ax2.set_xlabel('Time (s)')
    ax2.legend()
    ax2.grid(True)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(resume_path, f"robot_states_{timestamp}.png")
    plt.savefig(save_path)
    print(f"[INFO] Plot saved to: {os.path.abspath(save_path)}")
    
    # 显示图片
    plt.show()
    
    simulation_app.close()

if __name__ == "__main__":
    main()
"""
Sim2Sim Verification Script (Official Pattern)
- Adapted from Isaac Lab official play.py
- Includes: Keyboard Control, RSL-RL Wrapper, Correct Config Loading
"""

import argparse
import sys
import os
import torch
import numpy as np
import gymnasium as gym
import carb

# [Official] 1. Launch App first
from isaaclab.app import AppLauncher

# Argument Parsing
parser = argparse.ArgumentParser(description="Sim2Sim Verification with Keyboard")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-v0", help="Task name")
parser.add_argument("--load_run", type=str, required=True, help="Run folder name")
parser.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint filename")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
# Add standard AppLauncher args (headless, video, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Force GUI for keyboard interaction
args_cli.headless = False

# Launch the Simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# [Official] 2. Imports after AppLauncher
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # [Critical Fix] Wrapper import
from rsl_rl.runners import OnPolicyRunner

# [Critical Fix] Import your custom task package to register 'sk-Robot1'
# If your package is named 'robot1', keep this. If different, change it.
try:
    import robot1.tasks  # noqa: F401
    print("[INFO] Successfully imported 'robot1.tasks'")
except ImportError:
    print("[WARNING] Could not import 'robot1.tasks'. Ensure your python package is installed.")

# --- Keyboard Controller ---
# --- Keyboard Controller (修复版) ---
class KeyboardController:
    def __init__(self):
        # 1. 获取底层输入接口
        self.input = carb.input.acquire_input_interface()
        
        # 2. [核心修复] 通过 AppWindow 获取键盘设备句柄
        # 旧方法 self.input.get_keyboard(0) 已被废弃
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        
        # 3. 订阅键盘事件
        self.sub = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key_event)
        
        # 状态变量
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.speed_scale = 1.0
        self.rot_scale = 1.0

    def _on_key_event(self, event, *args):
        # 保持原有逻辑不变
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            if event.input == carb.input.KeyboardInput.W: self.cmd_vel[0] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.S: self.cmd_vel[0] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.A: self.cmd_vel[1] = self.speed_scale
            elif event.input == carb.input.KeyboardInput.D: self.cmd_vel[1] = -self.speed_scale
            elif event.input == carb.input.KeyboardInput.Q: self.cmd_vel[2] = self.rot_scale
            elif event.input == carb.input.KeyboardInput.E: self.cmd_vel[2] = -self.rot_scale
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input in [carb.input.KeyboardInput.W, carb.input.KeyboardInput.S]: self.cmd_vel[0] = 0.0
            if event.input in [carb.input.KeyboardInput.A, carb.input.KeyboardInput.D]: self.cmd_vel[1] = 0.0
            if event.input in [carb.input.KeyboardInput.Q, carb.input.KeyboardInput.E]: self.cmd_vel[2] = 0.0

    def get_command(self):
        return torch.tensor(self.cmd_vel, dtype=torch.float32, device="cuda:0")

def main():
    # --- A. Load & Override Env Config ---
    # [Fix] Use 'device' instead of 'use_gpu'
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    
    # Disable randomization for Sim2Sim stability
    #env_cfg.events = None 
    
    # 1. 关闭随机事件 (Sim2Sim 稳定性)
    #env_cfg.events = None 

    # ================= [核心修复：让机器人默认静止] =================
    # 找到指令管理器配置
    # ==========================================================
        # ================= [核心修复：强制默认静止] =================
    # 遍历配置中的所有指令项，将它们的随机范围设置为 0
    if hasattr(env_cfg, "commands"):
        print("[INFO] Zeroing out default command ranges to prevent auto-movement...")
        for term_name, term_cfg in env_cfg.commands.__dict__.items():
            # 检查是否有 'ranges' 属性 (标准指令配置都有这个)
            if hasattr(term_cfg, "ranges"):
                # 将线性速度 (x, y) 和 角速度 (heading) 的范围都设为 0
                # 这样环境生成的“默认背景指令”就是静止
                if hasattr(term_cfg.ranges, "lin_vel_x"): term_cfg.ranges.lin_vel_x = (0.0, 0.0)
                if hasattr(term_cfg.ranges, "lin_vel_y"): term_cfg.ranges.lin_vel_y = (0.0, 0.0)
                if hasattr(term_cfg.ranges, "ang_vel_z"): term_cfg.ranges.ang_vel_z = (0.0, 0.0)
                if hasattr(term_cfg.ranges, "heading"):   term_cfg.ranges.heading   = (0.0, 0.0)
    # ==========================================================
    # --- B. Create & Wrap Environment ---
    # --- B. Create & Wrap Environment ---
    # 1. Create Gym Env
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 2. [Critical Fix] Wrap with RslRlVecEnvWrapper
    # This provides the .get_observations() method required by OnPolicyRunner
    env = RslRlVecEnvWrapper(env)

    # --- C. Load Agent Config & Model ---
    # [Critical Fix] Load RSL-RL config separately
    print(f"[INFO] Loading RSL-RL agent config for: {args_cli.task}")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    
    # Convert to dict if necessary
    if hasattr(agent_cfg, "to_dict"):
        agent_cfg = agent_cfg.to_dict()
    
    '''
    # Determine Log Path
    log_root = os.path.join("logs", "rsl_rl", args_cli.task.split("-")[-2] if "Isaac" in args_cli.task else args_cli.task)
    if not os.path.exists(log_root): log_root = os.path.join("logs", "rsl_rl", args_cli.task)
    run_dir = os.path.join(log_root, args_cli.load_run)
    '''
        # ================= [智能路径修复] =================
    # 1. 首先检查用户传入的 load_run 是否已经是一个存在的绝对路径或相对路径
    possible_direct_path = args_cli.load_run
    if os.path.exists(possible_direct_path):
        run_dir = os.path.abspath(possible_direct_path)
        print(f"[INFO] Using direct path provided in --load_run: {run_dir}")
    
    else:
        # 2. 如果不是直接路径，则尝试自动构建标准路径
        # 加载 experiment_name (例如 "youxia_manager")
        experiment_name = ""
        if isinstance(agent_cfg, dict):
            experiment_name = agent_cfg.get("experiment_name", "")
        
        # 如果没找到，回退到使用任务名
        if not experiment_name:
            experiment_name = args_cli.task.split("-")[-2] if "Isaac" in args_cli.task else args_cli.task
            
        # 构建默认路径: logs/rsl_rl/{experiment_name}/{load_run}
        # 假设当前工作目录是 robot1 根目录
        run_dir = os.path.join("logs", "rsl_rl", experiment_name, args_cli.load_run)
        run_dir = os.path.abspath(run_dir)
        print(f"[INFO] Constructed run directory: {run_dir}")

    # 3. 最终检查
    if not os.path.exists(run_dir):
        print(f"[ERROR] Run directory does not exist: {run_dir}")
        print(f"[HINT] Try providing the FULL ABSOLUTE PATH to --load_run")
        sys.exit(1)
    # ===============================================

    # Initialize Runner
    # Now passing the correct 'agent_cfg' instead of 'env_cfg'
    runner = OnPolicyRunner(env, agent_cfg, log_dir=run_dir, device="cuda:0")
    runner.load(os.path.join(run_dir, args_cli.checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")

    # Keyboard Init
    key_controller = KeyboardController()
    
    # --- D. Simulation Loop ---
    # [Official Pattern] Use env.get_observations()
    obs, _ = env.get_observations()
    
    print(f"[INFO] Sim2Sim Running. Controls: W/S (X-vel), A/D (Y-vel), Q/E (Yaw).")

    while simulation_app.is_running():
        # 1. Get Keyboard Command
        target_vel = key_controller.get_command()
        
        # 2. Inject Command (Hack into CommandManager)
        # We need to access env.unwrapped to penetrate the RslRlVecEnvWrapper
        try:
            # Depending on wrappers, might need multiple .unwrapped
            base_env = env.unwrapped 
            cmd_manager = base_env.command_manager
            if "base_velocity" in cmd_manager._active_command_terms:
                cmd_manager._active_command_terms["base_velocity"].vel_command_b[:] = target_vel
        except AttributeError:
            pass # Handle cases where command manager isn't accessible directly

        # 3. Inference
        with torch.inference_mode():
            actions = policy(obs)
            # [Official Pattern] env.step returns obs directly in RslRlVecEnvWrapper
            obs, rews, dones, infos = env.step(actions)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
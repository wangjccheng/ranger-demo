"""
Sim2Sim Verification: Hard Override Mode (Verified)
"""
import argparse
import sys
import os
import torch
import numpy as np
import gymnasium as gym
import carb

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sim2Sim Hard Override")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-v0", help="Task name")
parser.add_argument("--load_run", type=str, required=True, help="Run folder name")
parser.add_argument("--checkpoint", type=str, default="model.pt", help="Checkpoint filename")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

try:
    import robot1.tasks
except ImportError:
    pass

# --- Keyboard Controller ---
class KeyboardController:
    def __init__(self):
        self.input = carb.input.acquire_input_interface()
        import omni.appwindow
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        self.sub = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key_event)
        self.cmd_vel = np.array([0.0, 0.0, 0.0])
        self.speed_scale = 1.0
        self.rot_scale = 1.0

    def _on_key_event(self, event, *args):
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
    # 1. 配置加载
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=args_cli.num_envs)
    env_cfg.events = None # 关闭随机化
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 2. 加载模型
    print(f"[INFO] Loading Config: {args_cli.task}")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if hasattr(agent_cfg, "to_dict"): agent_cfg = agent_cfg.to_dict()

    # 处理路径 (直接使用绝对路径逻辑)
    if os.path.exists(args_cli.load_run):
        run_dir = os.path.abspath(args_cli.load_run)
    else:
        # 回退逻辑
        exp_name = agent_cfg.get("experiment_name", args_cli.task)
        run_dir = os.path.join("logs", "rsl_rl", exp_name, args_cli.load_run)
    
    runner = OnPolicyRunner(env, agent_cfg, log_dir=run_dir, device="cuda:0")
    runner.load(os.path.join(run_dir, args_cli.checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")

    key_controller = KeyboardController()
    obs, _ = env.get_observations()
    
    print("=" * 60)
    print("[INFO] Sim2Sim READY. Click window to focus.")
    print("[INFO] Index 15 (vx) and 16 (wz) will be OVERWRITTEN.")
    print("=" * 60)

    step_cnt = 0

        
    # === 调试代码：强制停车测试 ===
    print("[INFO] TEST MODE: Forcing command to 0.0 to stop robot.")
    
    while simulation_app.is_running():
        # 1. 忽略键盘，直接给 0
        target_vel = torch.tensor([0.0, 0.0, 0.0], device="cuda:0")
        
        with torch.inference_mode():
            # 2. 暴力覆写 (Hard Override)
            # 你的 Debug 数据证明索引就是 15 和 16
            obs[:, 15] = 0.0   # 强制设为 0
            obs[:, 16] = 0.0   # 强制设为 0
            
            # 3. 打印核对 (这一步非常重要，请看控制台输出)
            # 应该看到输出全是 0.000
            print(f"DEBUG OBS: v_x={obs[0, 15]:.3f}, w_z={obs[0, 16]:.3f}")
            print(f"DEBUG OBS: v_x={obs[0:50,:]:.3f}")

            actions = policy(obs)
            
        obs, _, _, _ = env.step(actions)
            
        step_cnt += 1

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
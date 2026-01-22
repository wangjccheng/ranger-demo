"""
主动悬架对比实验脚本 (Passive vs PID)
"""

import torch
import hydra
import numpy as np
import matplotlib.pyplot as plt
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat

# 导入你的环境配置
# 注意：确保这一行能正确导入，如果报错请检查 robot1 包是否安装 (pip install -e source/robot1)
from robot1.tasks.manager_based.MY_EVN.robot1_env_cfg import ROBOT1RoughEnvCfg

class PIDController:
    """简单的姿态保持 PID 控制器"""
    def __init__(self, kp=2.0, kd=0.1):
        self.kp = kp
        self.kd = kd
        
    def compute(self, roll, pitch, roll_vel, pitch_vel):
        # 目标是 roll=0, pitch=0
        # 计算所需的姿态修正力矩/位置调整量
        
        # Pitch 修正：车头抬起 (pitch > 0) -> 需要压低车头
        action_pitch = self.kp * pitch + self.kd * pitch_vel
        
        # Roll 修正：车身右倾 (roll > 0) -> 需要抬高右侧
        action_roll = self.kp * roll + self.kd * roll_vel
        
        return action_pitch, action_roll

@hydra.main(config_path=None, version_base=None)
def main(cfg=None):
    # ---------------------------------------------------------
    # 1. 实验设置 (在这里修改模式)
    # ---------------------------------------------------------
    # MODE = "passive"  # 模式 1: 被动悬架 (关节不动)
    MODE = "pid"      # 模式 2: PID 主动控制
    
    SIM_TIME = 10.0   # 仿真时长 (秒)
    PLOT_RESULTS = True
    
    # PID 参数 (请根据实际情况微调)
    # 如果车身震荡过大，减小 kp；如果反应太慢，增大 kp
    pid = PIDController(kp=5.0, kd=0.2)
    
    # ---------------------------------------------------------
    # 2. 初始化环境
    # ---------------------------------------------------------
    env_cfg = ROBOT1RoughEnvCfg()
    env_cfg.scene.num_envs = 1  # 强制只跑 1 个环境
    
    # 关闭域随机化，确保对比公平 (可选)
    if hasattr(env_cfg, "events"):
        env_cfg.events = None 
        
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print(f"[INFO] Starting experiment: Mode = {MODE}")
    print(f"[INFO] Simulation Time: {SIM_TIME} s")

    # ---------------------------------------------------------
    # 3. 仿真循环
    # ---------------------------------------------------------
    obs, _ = env.reset()
    robot = env.scene["robot"]
    
    # 数据日志
    log_data = {
        "time": [],
        "pitch": [],
        "roll": [],
        "height": [],
        "cmd_vel": []
    }
    
    dt = env.step_dt
    num_steps = int(SIM_TIME / dt)

    for i in range(num_steps):
        # ----------------------------------------
        # A. 生成底盘速度指令 (模拟定速巡航)
        # ----------------------------------------
        # 设定：以 1.0 m/s 前进，无转向
        base_action = torch.tensor([[1.0, 0.0]], device=env.device) 
        
        # ----------------------------------------
        # B. 获取真实状态 (Ground Truth)
        # ----------------------------------------
        quat = robot.data.root_quat_w
        ang_vel = robot.data.root_ang_vel_w
        pos = robot.data.root_pos_w
        
        # 转欧拉角
        r, p, y = euler_xyz_from_quat(quat)
        
        # 提取标量值 (取第 0 个环境)
        roll_val = r[0].item()
        pitch_val = p[0].item()
        roll_vel = ang_vel[0, 0].item()
        pitch_vel = ang_vel[0, 1].item()
        height_val = pos[0, 2].item()
        
        # ----------------------------------------
        # C. 计算腿部动作
        # ----------------------------------------
        if MODE == "passive":
            # 被动模式：保持原长 (0.0)
            leg_action = torch.zeros((env.num_envs, 4), device=env.device)
            
        elif MODE == "pid":
            # PID 计算
            act_p, act_r = pid.compute(roll_val, pitch_val, roll_vel, pitch_vel)
            
            # 分配动作到 4 条腿
            # 假设关节顺序为: [LF, RF, LB, RB] (左前, 右前, 左后, 右后)
            # 符号定义: action > 0 代表 "抬高车身" (伸长腿)
            
            # LF (左前): 需抵抗前倾(-p) 和 左倾(+r) -> 实际上左倾roll>0意味着右边低左边高? 
            # 通常 Roll>0 是右倾。
            # 修正逻辑：
            # Pitch > 0 (抬头) -> 后腿抬高(+)，前腿降低(-)
            # Roll > 0 (右倾) -> 右腿抬高(+)，左腿降低(-)
            
            lf = -act_p - act_r
            rf = -act_p + act_r
            lb = +act_p - act_r
            rb = +act_p + act_r
            
            leg_action = torch.tensor([[lf, rf, lb, rb]], device=env.device)
            
            # 截断到 [-1, 1] 动作范围
            leg_action = torch.clamp(leg_action, -1.0, 1.0)

        # ----------------------------------------
        # D. 执行动作
        # ----------------------------------------
        # 拼接：[v_x, omega_z, leg_1, leg_2, leg_3, leg_4]
        full_actions = torch.cat([base_action, leg_action], dim=1)
        
        # Isaac Lab Step
        obs, rew, terminated, truncated, info = env.step(full_actions)
        
        # ----------------------------------------
        # E. 记录数据
        # ----------------------------------------
        log_data["time"].append(i * dt)
        log_data["pitch"].append(pitch_val)
        log_data["roll"].append(roll_val)
        log_data["height"].append(height_val)
        
        # 重置处理
        if terminated.any() or truncated.any():
            obs, _ = env.reset()

    print("[INFO] Simulation finished.")
    
    # ---------------------------------------------------------
    # 4. 绘图与保存
    # ---------------------------------------------------------
    if PLOT_RESULTS:
        plt.figure(figsize=(10, 8))
        
        # Pitch 曲线
        plt.subplot(3, 1, 1)
        plt.plot(log_data["time"], log_data["pitch"], label=f'Pitch ({MODE})', color='blue')
        plt.title(f"Vehicle Attitude Stability - {MODE.upper()}")
        plt.ylabel("Pitch (rad)")
        plt.grid(True)
        plt.legend()
        
        # Roll 曲线
        plt.subplot(3, 1, 2)
        plt.plot(log_data["time"], log_data["roll"], label=f'Roll ({MODE})', color='orange')
        plt.ylabel("Roll (rad)")
        plt.grid(True)
        plt.legend()
        
        # Height 曲线
        plt.subplot(3, 1, 3)
        plt.plot(log_data["time"], log_data["height"], label=f'Height ({MODE})', color='green')
        plt.ylabel("Height (m)")
        plt.xlabel("Time (s)")
        plt.grid(True)
        plt.legend()
        
        save_path = f"experiment_result_{MODE}.png"
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
        plt.show()

if __name__ == "__main__":
    main()
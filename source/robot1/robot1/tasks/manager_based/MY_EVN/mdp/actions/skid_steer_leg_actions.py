# 文件路径: .../MY_EVN/mdp/actions/skid_steer_leg_actions.py

from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm

class SkidSteerLegAction(ActionTerm):
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # 1. 获取关节索引
        self._left_ids, _  = self._asset.find_joints(cfg.left_wheel_joint_names,  preserve_order=True)
        self._right_ids, _ = self._asset.find_joints(cfg.right_wheel_joint_names, preserve_order=True)
        self._leg_ids, _   = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._all_wheel_ids = list(self._left_ids) + list(self._right_ids)

        # 2. 映射系数
        self._wheel_scale = getattr(cfg, "wheel_scale", 10.0)
        self._wheel_offset = getattr(cfg, "wheel_offset", 0.0)
        self._leg_vel_scale = getattr(cfg, "leg_vel_scale", 3.0)
        
        self.actuator_lag_alpha = getattr(cfg, "actuator_lag_alpha", 0.8) 
        self.eha_lag_alpha      = getattr(cfg, "eha_lag_alpha", 0.6)

        # ★ 方案B：获取配置的最大允许动作变化率
        self.max_action_delta = getattr(cfg, "max_action_delta", 0.1)

        # 3. 动作维度 = 4(轮速) + 4(腿速) = 8
        self.num_wheels = len(self._all_wheel_ids)
        self.num_legs = len(self._leg_ids)
        self._action_dim = self.num_wheels + self.num_legs
        
        # 4. 初始化缓存
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # 底层物理 LPF 缓存
        self._prev_wheel_vel_cmd = torch.zeros(self.num_envs, self.num_wheels, device=self.device)
        self._prev_leg_vel_cmd   = torch.zeros(self.num_envs, self.num_legs, device=self.device)

        # ★ 方案B：用于记录上一帧最终通过限幅的 Raw Action
        self._last_raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            
        # 重置物理底层缓存
        self._prev_wheel_vel_cmd[env_ids] = 0.0
        self._prev_leg_vel_cmd[env_ids] = 0.0

        # ★ 方案B：重置时，上一帧动作归零
        self._last_raw_actions[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor):
        # 1. 确保网络直接输出在 [-1, 1] 之间
        actions = torch.clamp(actions.detach(), -1.0, 1.0)
        
        # ==========================================================
        # ★ 方案B 核心逻辑：计算并硬性截断指令变化率
        # ==========================================================
        # 计算当前指令与上一帧实际指令的差值
        delta = actions - self._last_raw_actions
        
        # 将差值强行截断在允许的最大变化跨度内
        delta = torch.clamp(delta, -self.max_action_delta, self.max_action_delta)
        
        # 得到平滑且安全的新指令，并更新缓存
        smoothed_actions = self._last_raw_actions + delta
        self._last_raw_actions[:] = smoothed_actions
        
        # 更新 self._raw_actions 以供 observation 或 rewards 读取 (它们读到的是平滑后的值)
        self._raw_actions[:] = smoothed_actions
        
        # ==========================================================
        # 2. 解析为实际的物理目标转速
        # ==========================================================
        # 提取前 4 个维度给轮子
        wheel_raw = smoothed_actions[:, :self.num_wheels]
        wheel_cmd = wheel_raw * self._wheel_scale + self._wheel_offset

        # 提取后 4 个维度给腿部 EHA
        leg_raw = smoothed_actions[:, self.num_wheels:]
        leg_cmd = leg_raw * self._leg_vel_scale

        self._processed_actions[:, :self.num_wheels] = wheel_cmd
        self._processed_actions[:, self.num_wheels:] = leg_cmd
        
    def apply_actions(self):
        wheel_vel_target = self._processed_actions[:, :self.num_wheels]
        leg_vel_target   = self._processed_actions[:, self.num_wheels:]

        # 物理迟滞模拟 (由于方案B已经对 Raw Action 做了强平滑，这里的 alpha 即使设为 1.0 也很安全)
        wheel_vel_cmd = (self.actuator_lag_alpha * wheel_vel_target + 
                         (1 - self.actuator_lag_alpha) * self._prev_wheel_vel_cmd).detach()
        
        leg_vel_cmd   = (self.eha_lag_alpha * leg_vel_target + 
                         (1 - self.eha_lag_alpha) * self._prev_leg_vel_cmd).detach()
        
        self._prev_wheel_vel_cmd[:] = wheel_vel_cmd
        self._prev_leg_vel_cmd[:]   = leg_vel_cmd
        
        # 统一下发速度指令
        self._asset.set_joint_velocity_target(wheel_vel_cmd, joint_ids=self._all_wheel_ids)
        self._asset.set_joint_velocity_target(leg_vel_cmd, joint_ids=self._leg_ids)
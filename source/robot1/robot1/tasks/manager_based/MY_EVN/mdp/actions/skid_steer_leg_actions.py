from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

class SkidSteerLegAction(ActionTerm):
    """
    二合一动作项：
    - 输入动作: [v, omega, q_leg_0, ..., q_leg_{M-1}]
        v [m/s], omega [rad/s], 余下为调距关节的归一化指令
    - 输出命令:
        1) 四个驱动轮的角速度目标 [rad/s]
        2) M 个调距关节的位置目标 [rad]
    - 需要配置:
        - 车辆几何: base_width (W), wheel_radius (r)
        - 轮子关节名: left_wheel_joint_names, right_wheel_joint_names
        - 调距关节名: leg_joint_names (列表或正则)
        - 标定参数: base_scale/offset, bounding_strategy, no_reverse
        - 调距映射: leg_rescale_to_limits（优先），或 leg_scale/leg_offset
    """
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # 轮子关节索引（保序）
        self._left_ids, _  = self._asset.find_joints(cfg.left_wheel_joint_names,  preserve_order=True)
        self._right_ids, _ = self._asset.find_joints(cfg.right_wheel_joint_names, preserve_order=True)
        if len(self._left_ids) == 0 or len(self._right_ids) == 0:
            raise ValueError("No wheel joints found. Check left_wheel_joint_names/right_wheel_joint_names.")

        # 调距关节索引（保序）
        self._leg_ids, _ = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        if len(self._leg_ids) == 0:
            raise ValueError("No leg joints found. Check leg_joint_names (e.g., ['g_lf','g_rf','g_lb','g_rb'] 或正则).")

        # 底盘几何
        self.W: float = float(cfg.base_width)
        self.r: float = float(cfg.wheel_radius)

        # 底盘标定与约束
        self._base_scale = torch.tensor(cfg.base_scale, device=self.device).view(1, 2)     # [v_scale, omega_scale]
        self._base_offset = torch.tensor(cfg.base_offset, device=self.device).view(1, 2)   # [v_off, omega_off]
        self._bounding_strategy = cfg.bounding_strategy
        self._no_reverse = bool(cfg.no_reverse)

        # 调距关节映射
        self._leg_rescale_to_limits = bool(cfg.leg_rescale_to_limits)
        self._leg_scale = float(cfg.leg_scale)
        self._leg_offset = float(cfg.leg_offset)

        # 动作维度：2（v,omega）+ M（leg joints）
        self._n_leg = len(self._leg_ids)
        self._action_dim = 2 + self._n_leg

        # 运行中缓存
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # +++ 【新增 1：初始化低通滤波器缓存】 +++
        self._action_alpha = getattr(cfg, "action_alpha", 1.0)
        self._prev_wheel_speeds = torch.zeros(self.num_envs, len(self._left_ids) + len(self._right_ids), device=self.device)
        self._prev_leg_vels = torch.zeros(self.num_envs, self._n_leg, device=self.device)
    
    # +++ 【新增 2：极其关键的防坑重置】 +++
    # 如果不写这个，机器人在摔倒重置后，会继承上一个回合死亡前的疯狂速度，导致“幽灵冲刺”
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_wheel_speeds[env_ids] = 0.0
        self._prev_leg_vels[env_ids] = 0.0
    
    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _bound_base_cmd(self, cmd: torch.Tensor) -> torch.Tensor:
        # cmd: (..., 2)
        if self._bounding_strategy == "clip":
            return torch.clamp(cmd, -1.0, 1.0)
        elif self._bounding_strategy == "tanh":
            return torch.tanh(cmd)
        else:
            return cmd
    '''
    def process_actions(self, actions: torch.Tensor):
        # 保存原始动作
        self._raw_actions[:] = actions

        # 1) 底盘两维: 线性变换 + 有界
        base_raw = actions[:, :2]                                # [N,2]
        base_cmd = base_raw * self._base_scale + self._base_offset
        base_cmd = self._bound_base_cmd(base_cmd)
        if self._no_reverse:
            base_cmd[:, 0] = torch.clamp(base_cmd[:, 0], min=0.0)  # 禁止倒车：v>=0

        # 2) 调距关节 M 维：归一化到实际关节目标（优先使用软限反归一化）
        leg_raw = actions[:, 2:]                                 # [N,M]
        if self._leg_rescale_to_limits:
            leg_raw = torch.clamp(leg_raw, -1.0, 1.0)
            low  = self._asset.data.soft_joint_pos_limits[:, self._leg_ids, 0]  # [N,M]
            high = self._asset.data.soft_joint_pos_limits[:, self._leg_ids, 1]  # [N,M]
            leg_cmd = math_utils.unscale_transform(leg_raw, low, high)          # [-1,1] -> [low, high]
        else:
            leg_cmd = leg_raw * self._leg_scale + self._leg_offset

        # 拼接回 processed 动作缓存（便于日志/可视化）
        self._processed_actions[:, :2] = base_cmd
        self._processed_actions[:, 2:] = leg_cmd
    '''
    def process_actions(self, actions: torch.Tensor):
        # 保存原始动作
        self._raw_actions[:] = actions

        # 1) 底盘两维: 保持不变
        base_raw = actions[:, :2]
        base_cmd = base_raw * self._base_scale + self._base_offset
        base_cmd = self._bound_base_cmd(base_cmd)
        if self._no_reverse:
            base_cmd[:, 0] = torch.clamp(base_cmd[:, 0], min=0.0)

        # 2) 调距关节 M 维: 修改为速度指令处理
        # ---------------- 修改开始 ----------------
        leg_raw = actions[:, 2:]  # [N,M]
        
        # 直接使用 scale 和 offset。
        # scale 现在代表 "最大目标角速度" (例如 10 rad/s)
        # offset 通常设为 0.0
        leg_cmd = leg_raw * self._leg_scale + self._leg_offset
        
        # (可选) 如果你想限制最大速度，可以加 clamp
        # leg_cmd = torch.clamp(leg_cmd, -self._max_leg_vel, self._max_leg_vel)
        # ---------------- 修改结束 ----------------

        # 拼接回 processed 动作缓存
        self._processed_actions[:, :2] = base_cmd
        self._processed_actions[:, 2:] = leg_cmd
        
    def apply_actions(self):
        # 从 processed 中取底盘命令
        v     = self._processed_actions[:, 0]  # [N]
        omega = self._processed_actions[:, 1]  # [N]

        # 差速（skid-steer）解算左右轮角速度（rad/s）
        wl = (v - omega * (self.W / 2.0)) / self.r  # 左侧
        wr = (v + omega * (self.W / 2.0)) / self.r  # 右侧

        # 组装四轮速度矩阵 [N, #wheels]
        nL, nR = len(self._left_ids), len(self._right_ids)
        wheel_speeds = torch.zeros(self.num_envs, nL + nR, device=self.device)
        wheel_speeds[:, :nL] = wl.view(-1, 1).expand(-1, nL)
        wheel_speeds[:, nL:] = wr.view(-1, 1).expand(-1, nR)
        # 轮子：执行纯速度控制
        all_wheel_ids = list(self._left_ids) + list(self._right_ids)
        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=all_wheel_ids)
# ==========================================
        # 2. 腿部：执行“低通滤波位置控制” (完美模拟液压)
        # ==========================================
        # 网络解析出的期望绝对位置
        leg_pos_targets = self._processed_actions[:, 2:]  
        
        # 获取液压缸“当前的真实物理位置”作为基准
        current_leg_pos = self._asset.data.joint_pos[:, self._leg_ids]

        # ★ 核心公式：液压建压延迟模拟
        # 物理引擎下一帧的目标位置 = 真实当前位置 + (期望位置 - 真实当前位置) * alpha
        # alpha 取 0.05 ~ 0.1 左右。
        # 这意味着：即便网络瞬间要求腿伸长 10cm，实际下发的指令也只会在当前位置基础上前进 0.5cm。
        
        smoothed_leg_pos = current_leg_pos + (leg_pos_targets - current_leg_pos) * self._action_alpha
        
        # 下发平滑后的【位置目标】给物理引擎
        self._asset.set_joint_position_target(smoothed_leg_pos, joint_ids=self._leg_ids)
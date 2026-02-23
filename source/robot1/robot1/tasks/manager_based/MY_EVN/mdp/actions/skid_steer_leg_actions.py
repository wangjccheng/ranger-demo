
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
import isaaclab.utils.math as math_utils

class SkidSteerLegAction(ActionTerm):
    """
    Sim2Real 增强版动作项 (解决第二个问题: 驱动器建模):
    - 轮毂电机: 速度闭环 -> 转换为阻尼力矩，模拟电机反电动势与扭矩饱和
    - EHA 腿部: 阻抗控制 -> 模拟主动悬架的刚度(弹簧)、阻尼、液压建压延迟和压力上限
    """
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        # 1. 获取关节索引 (保序)
        self._left_ids, _  = self._asset.find_joints(cfg.left_wheel_joint_names,  preserve_order=True)
        self._right_ids, _ = self._asset.find_joints(cfg.right_wheel_joint_names, preserve_order=True)
        self._leg_ids, _   = self._asset.find_joints(cfg.leg_joint_names, preserve_order=True)
        self._all_wheel_ids = list(self._left_ids) + list(self._right_ids)

        if len(self._all_wheel_ids) == 0 or len(self._leg_ids) == 0:
            raise ValueError("未找到轮子或腿部关节，请检查配置。")

        # 2. 几何与映射参数
        self.W = float(cfg.base_width)
        self.r = float(cfg.wheel_radius)
        self._base_scale = torch.tensor(cfg.base_scale, device=self.device).view(1, 2)
        self._base_offset = torch.tensor(cfg.base_offset, device=self.device).view(1, 2)
        self._leg_scale = float(cfg.leg_scale)
        self._leg_offset = float(cfg.leg_offset)
        self._bounding_strategy = getattr(cfg, "bounding_strategy", "clip")
        self._no_reverse = bool(getattr(cfg, "no_reverse", False))

        # ==========================================================
        # 3. [Sim2Real 核心] 驱动器真实物理参数
        # ==========================================================
        # A. 物理扭矩极限 (Nm)
        # 轮毂电机峰值扭矩 (150kg车通常需 60-100Nm)
        self.max_wheel_torque = getattr(cfg, "max_wheel_torque", 100.0) 
        # EHA 最大推力换算的力矩 (假设推力 15000N * 力臂 0.2m = 3000Nm，这里给一个足够大的安全上限)
        self.max_eha_torque   = getattr(cfg, "max_eha_torque", 500.0) 
        
        # B. 驱动增益 (相当于虚拟 PD)
        # 轮子速度环增益：假设误差 2rad/s 时输出 60Nm，则 Kv = 30
        self.wheel_damping = getattr(cfg, "wheel_damping", 50.0)       
        
        # EHA 悬架弹簧刚度 (60000 N/m 转换为旋转刚度)
        self.eha_stiffness = getattr(cfg, "eha_stiffness", 2400.0)     
        # EHA 悬架减震阻尼 (3000 N/s/m 转换为旋转阻尼)
        self.eha_damping   = getattr(cfg, "eha_damping", 120.0)        
        
        # C. 延迟模拟 (低通滤波器系数 alpha)
        self.actuator_lag_alpha = getattr(cfg, "actuator_lag_alpha", 0.4) 
        self.eha_lag_alpha      = getattr(cfg, "eha_lag_alpha", 0.2)

        # ==========================================================
        # 4. 运行时缓存 (包含延迟历史)
        # ==========================================================
        self._action_dim = 2 + len(self._leg_ids)
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # 记录上一帧的指令，用于计算低通滤波 (LPF)
        self._prev_wheel_vel_cmd = torch.zeros(self.num_envs, len(self._all_wheel_ids), device=self.device)
        self._prev_leg_pos_cmd   = torch.zeros(self.num_envs, len(self._leg_ids), device=self.device)

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
        if self._bounding_strategy == "clip":
            return torch.clamp(cmd, -1.0, 1.0)
        elif self._bounding_strategy == "tanh":
            return torch.tanh(cmd)
        return cmd

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        
        # 1. 解析底盘指令 (V, Omega)
        base_raw = actions[:, :2]
        base_cmd = base_raw * self._base_scale + self._base_offset
        base_cmd = self._bound_base_cmd(base_cmd)
        if self._no_reverse:
            base_cmd[:, 0] = torch.clamp(base_cmd[:, 0], min=0.0)

        # 2. 解析腿部指令
        # 注意：这里我们把它解析为 EHA 期望到达的“目标位置 (Position)”
        leg_raw = actions[:, 2:]
        leg_cmd = leg_raw * self._leg_scale + self._leg_offset

        self._processed_actions[:, :2] = base_cmd
        self._processed_actions[:, 2:] = leg_cmd

    def apply_actions(self):
        """
        [Sim2Real 核心] 物理执行层：完全接管力矩计算
        """
        # ===============================================
        # 第一步：运动学解算 (算出理想目标值)
        # ===============================================
        v, omega = self._processed_actions[:, 0], self._processed_actions[:, 1]
        
        # 轮子：差速模型算出左右轮理想速度
        wl = (v - omega * (self.W / 2.0)) / self.r
        wr = (v + omega * (self.W / 2.0)) / self.r
        nL, nR = len(self._left_ids), len(self._right_ids)
        wheel_vel_target = torch.cat([wl.view(-1, 1).expand(-1, nL), wr.view(-1, 1).expand(-1, nR)], dim=1)
        
        # 腿部：期望到达的位置
        leg_pos_target = self._processed_actions[:, 2:]

        # ===============================================
        # 第二步：延迟模拟 (Low Pass Filter)
        # 模拟电机通讯延迟和 EHA 液压油建压的缓慢过程
        # ===============================================
        wheel_vel_cmd = self.actuator_lag_alpha * wheel_vel_target + (1 - self.actuator_lag_alpha) * self._prev_wheel_vel_cmd
        leg_pos_cmd   = self.eha_lag_alpha * leg_pos_target + (1 - self.eha_lag_alpha) * self._prev_leg_pos_cmd
        
        # 更新历史缓存
        self._prev_wheel_vel_cmd[:] = wheel_vel_cmd
        self._prev_leg_pos_cmd[:]   = leg_pos_cmd

        # ===============================================
        # 第三步：计算实际输出力矩 (Explicit Torque)
        # ===============================================
        # 获取机器人当前真实的物理状态
        current_wheel_vel = self._asset.data.joint_vel[:, self._all_wheel_ids]
        current_leg_pos = self._asset.data.joint_pos[:, self._leg_ids]
        current_leg_vel = self._asset.data.joint_vel[:, self._leg_ids]

        # A. 轮子计算：纯阻尼模式 (Torque = Kd * Error_V)
        wheel_torques = self.wheel_damping * (wheel_vel_cmd - current_wheel_vel)
        # 强制截断，模拟电机物理极限
        wheel_torques = torch.clamp(wheel_torques, -self.max_wheel_torque, self.max_wheel_torque)

        # B. 腿部计算：阻抗/悬架模式 (Torque = Kp * Error_P - Kd * V)
        # 这赋予了腿部弹簧一样的支撑力和减震效果
        leg_torques = self.eha_stiffness * (leg_pos_cmd - current_leg_pos) - self.eha_damping * current_leg_vel
        # 强制截断，模拟液压缸最大压力
        leg_torques = torch.clamp(leg_torques, -self.max_eha_torque, self.max_eha_torque)

        # ===============================================
        # 第四步：下发力矩
        # 放弃使用 velocity_target，全面改为 effort_target
        # ===============================================
        self._asset.set_joint_effort_target(wheel_torques, joint_ids=self._all_wheel_ids)
        self._asset.set_joint_effort_target(leg_torques, joint_ids=self._leg_ids)
'''



















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

        # 下发到轮子（速度控制）
        all_wheel_ids = list(self._left_ids) + list(self._right_ids)
        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids=all_wheel_ids)
# ---------------- 修改开始 ----------------
        # 调距目标（速度控制）
        # 注意：这里我们把 leg_targets 当作速度指令
        leg_vel_targets = self._processed_actions[:, 2:]  # [N,M]
        
        # ★ 关键修改：使用 set_joint_velocity_target
        # 同时，为了防止物理引擎残留位置目标，建议将位置目标设为 None (Isaac Lab底层通常会自动处理)
        # 但显式调用 set_joint_velocity_target 是必须的。
        self._asset.set_joint_velocity_target(leg_vel_targets, joint_ids=self._leg_ids)
        # ---------------- 修改结束 ----------------
        # 调距目标（位置控制）
        #leg_targets = self._processed_actions[:, 2:]  # [N,M]
        #self._asset.set_joint_position_target(leg_targets, joint_ids=self._leg_ids)

# 文件路径: source/robot1/robot1/tasks/manager_based/MY_EVN/mdp/actions/skid_steer_leg_actions.py

'''
from __future__ import annotations
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
import isaaclab.utils.math as math_utils

class SkidSteerLegAction(ActionTerm):
    """
    Sim2Real 增强版动作项:
    - 轮毂电机: 速度控制，在 Python 层模拟低通延迟，底层配置阻尼和力矩上限
    - EHA 腿部: 位置控制，在 Python 层模拟液压建压延迟，底层配置高刚度、阻尼和力矩上限
    - 域随机化增强: 引入环形队列(Ring Buffer)模拟真实物理系统的随机通讯延迟
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

        # 3. 物理迟滞模拟参数 (低通滤波器系数 alpha)
        # 数值越小，物理惯性/建压延迟越明显；1.0代表没有延迟
        self.actuator_lag_alpha = getattr(cfg, "actuator_lag_alpha", 0.8) 
        self.eha_lag_alpha      = getattr(cfg, "eha_lag_alpha", 0.6)

        # 4. 运行时缓存
        self._action_dim = 2 + len(self._leg_ids)
        self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # 记录上一帧的物理指令，用于计算低通滤波 (LPF)
        self._prev_wheel_vel_cmd = torch.zeros(self.num_envs, len(self._all_wheel_ids), device=self.device)
        self._prev_leg_pos_cmd   = torch.zeros(self.num_envs, len(self._leg_ids), device=self.device)

        # ==========================================================
        # 5. [新增] 通信延迟模拟参数与队列 (Ring Buffer)
        # ==========================================================
        # 默认延迟 1~3 个 RL step (以 50Hz 计，相当于 20ms~60ms 随机死区)
        self.min_delay = getattr(cfg, "delay_steps_min", 1)
        self.max_delay = getattr(cfg, "delay_steps_max", 3)
        
        # 队列形状: [最大延迟帧数 + 1, 环境数量, 动作维度]
        self._action_history = torch.zeros(
            (self.max_delay + 1, self.num_envs, self._action_dim), 
            device=self.device
        )
        # 为每个 env 随机生成一个当前的通信延迟帧数
        self._current_delays = torch.randint(
            self.min_delay, self.max_delay + 1, 
            (self.num_envs,), 
            device=self.device
        )

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

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """回合重置时，彻底清空历史指令缓存和延迟队列"""
        if env_ids is None:
            env_ids = slice(None)
            
        # [原有逻辑] 低通滤波器缓存清零/重置
        self._prev_wheel_vel_cmd[env_ids] = 0.0
        current_leg_positions = self._asset.data.joint_pos[env_ids]
        self._prev_leg_pos_cmd[env_ids] = current_leg_positions[:, self._leg_ids]

        # [新增逻辑] 清空通信延迟队列
        self._action_history[:, env_ids, :] = 0.0
        # 重新为这些环境随机分配延迟帧数，增强域随机化
        self._current_delays[env_ids] = torch.randint(
            self.min_delay, self.max_delay + 1, 
            (len(env_ids),), 
            device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        actions = torch.clamp(actions.detach(), -1.0, 1.0)
        self._raw_actions[:] = actions
        
        # ==========================================================
        # [新增核心] 拦截动作，送入延迟队列
        # ==========================================================
        # 1. 队列全体往后挪一格 (丢弃最旧的)
        self._action_history[1:] = self._action_history[:-1].clone()
        # 2. 最新动作放队头
        self._action_history[0] = actions.clone()

        # 3. 根据每个 env 自己的延迟设定，提取对应的滞后动作
        delayed_actions = torch.gather(
            self._action_history, 
            dim=0, 
            index=self._current_delays.view(1, self.num_envs, 1).expand(1, self.num_envs, self.action_dim)
        ).squeeze(0)

        # ==========================================================
        # 解析指令 (使用 delayed_actions 替代原来的 actions)
        # ==========================================================
        # 1. 解析底盘指令 (V, Omega)
        base_raw = delayed_actions[:, :2]
        base_cmd = base_raw * self._base_scale + self._base_offset
        base_cmd = self._bound_base_cmd(base_cmd)
        if self._no_reverse:
            base_cmd[:, 0] = torch.clamp(base_cmd[:, 0], min=0.0)

        # 2. 解析腿部指令 (解析为期望到达的目标位置 Position)
        leg_raw = delayed_actions[:, 2:]
        leg_cmd = leg_raw * self._leg_scale + self._leg_offset

        self._processed_actions[:, :2] = base_cmd
        self._processed_actions[:, 2:] = leg_cmd

    def apply_actions(self):
        """
        [Sim2Real 核心] 物理执行层：结合隐式控制与低通延迟
        此时接收到的 self._processed_actions 已经是经过“通讯延迟”之后的滞后指令了
        """
        # 第一步：运动学解算 (算出理想目标值)
        v, omega = self._processed_actions[:, 0], self._processed_actions[:, 1]
        
        wl = (v - omega * (self.W / 2.0)) / self.r
        wr = (v + omega * (self.W / 2.0)) / self.r
        nL, nR = len(self._left_ids), len(self._right_ids)
        wheel_vel_target = torch.cat([wl.view(-1, 1).expand(-1, nL), wr.view(-1, 1).expand(-1, nR)], dim=1)
        
        leg_pos_target = self._processed_actions[:, 2:]

        # 第二步：物理迟滞模拟 (Low Pass Filter)
        # 模拟电机发力缓慢和 EHA 液压油建压的过程
        wheel_vel_cmd = (self.actuator_lag_alpha * wheel_vel_target + 
                         (1 - self.actuator_lag_alpha) * self._prev_wheel_vel_cmd).detach()
        leg_pos_cmd   = (self.eha_lag_alpha * leg_pos_target + 
                         (1 - self.eha_lag_alpha) * self._prev_leg_pos_cmd).detach()
        
        # 更新历史缓存
        self._prev_wheel_vel_cmd[:] = wheel_vel_cmd
        self._prev_leg_pos_cmd[:]   = leg_pos_cmd

        # 第三步：下发给物理引擎底层 
        self._asset.set_joint_velocity_target(wheel_vel_cmd, joint_ids=self._all_wheel_ids)
        self._asset.set_joint_position_target(leg_pos_cmd, joint_ids=self._leg_ids)



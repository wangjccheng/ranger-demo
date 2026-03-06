import torch
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

# ---------------------------
# 自定义奖励项（与动作对齐）
# ---------------------------

def leg_pos_center_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节“偏离默认姿态”惩罚：惩罚偏离默认待机姿态(0.05)的平方和。"""
    asset = env.scene[asset_cfg.name]
    
    # 1. 获取当前真实物理角度
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # 2. ★ 核心修改：不再计算软限位中点，而是直接读取硬件配置里的默认角度（也就是你刚设的 0.05）
    default_q = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    # 获取软限位宽度的一半（依然用来做归一化，保证扣分的量级和以前一样）
    low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    high= asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    half= torch.clamp(0.5 * (high - low), min=1e-6)
    
    # 3. 计算偏离“默认角度”的归一化误差
    qn = torch.clamp((q - default_q) / half, -1.0, 1.0)
    
    return torch.sum(qn**2, dim=1)

def leg_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节速度惩罚（平方和）。"""
    return torch.sum(mdp.joint_vel(env, asset_cfg)**2, dim=1)
'''
def slip_consistency_l2(
    env,
    wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="w_.*"),
    base_width: float = 0.5,
    wheel_radius: float = 0.05,
) -> torch.Tensor:
    """
    打滑一致性惩罚：由左右轮平均角速度反推 (v_hat, w_hat)，与真实 (v_x, w_z) 的差的平方和。
    注：wheel_cfg 匹配的次序应与动作项一致，形如 [左若干, 右若干]。
    """
    v_b = mdp.base_lin_vel(env)[:, 0]     # 真实 v_x（机体系）
    w_b = mdp.base_ang_vel(env)[:, 2]     # 真实 ω_z（机体系）
    w_all = mdp.joint_vel(env, wheel_cfg) # [N, nL+nR]
    n = w_all.shape[1] // 2
    wl = torch.mean(w_all[:, :n], dim=1)  # 左平均
    wr = torch.mean(w_all[:, n:], dim=1)  # 右平均
    v_hat = wheel_radius * 0.5 * (wr + wl)
    w_hat = wheel_radius / base_width * (wr - wl)
    dv2 = (v_hat - v_b)**2
    dw2 = (w_hat - w_b)**2
    return dv2 + dw2
'''
# 可选：姿态/重力投影的“水平”奖励已由内置 flat_orientation_l2/ang_vel_xy_l2 覆盖 [4]。
def feet_air_time_l2(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["w_lf", "w_rf", "w_lb", "w_rb"],  # 直接用你四个轮子的 link 名称
    ),
    max_air_time: float = 0.1,
) -> torch.Tensor:
    """足端离地惩罚: 对超过允许离地时间的部分做 L2 累加.

    - 使用 ContactSensorData.current_air_time, 而不是不存在的 air_time 字段。
    - 通过 SceneEntityCfg.body_ids 自动拿到这几个轮子的索引。
    """
    # 取接触传感器
    sensor = env.scene[sensor_cfg.name]      # "contact_forces"
    data = sensor.data                       # ContactSensorData

    if data.current_air_time is None:
        raise RuntimeError(
            "contact_forces 传感器未开启 track_air_time=True, "
            "请在 ContactSensorCfg 中打开该选项。"
        )

    # sensor_cfg.body_ids 在 env 初始化时由 SceneManager 填好
    foot_ids = sensor_cfg.body_ids           # [n_feet]

    # current_air_time: [N, num_bodies] → 取足端列 → [N, n_feet]
    air_time = data.current_air_time[:, foot_ids]

    # 只惩罚超过 max_air_time 的那部分，并做 L2
    excess = torch.clamp(air_time - max_air_time, min=0.0)
    return torch.sum(excess**2, dim=1)       # [N]

def log_base_pitch(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """记录车身俯仰角(Pitch)的绝对值 (rad)"""
    # 获取欧拉角 (Roll, Pitch, Yaw)
    # 注意：需确保引入了 math_utils: import isaaclab.utils.math as math_utils
    rot = mdp.root_quat_w(env, asset_cfg)
    _, pitch, _ = math_utils.euler_xyz_from_quat(rot)
    pitch_abs = torch.abs(pitch)

    return pitch_abs
# ---------------------------
# 奖励配置（与动作/命令对齐）
# ---------------------------
def residual_mean_penalty(env, residual_indices: list = [2, 3, 4, 5]) -> torch.Tensor:
    """
    零和残差约束：惩罚残差的均值不为0。
    强迫神经网络把残差纯粹当作“差速器”来用，不允许用它来整体提速或减速。
    """
    # 提取四个残差
    residuals = env.action_manager.action[:, residual_indices]
    
    # 计算均值
    mean_residual = torch.mean(residuals, dim=1)
    
    # 惩罚均值的平方
    return mean_residual**2


def flat_orientation_with_tolerance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance_deg: float = 2.0  # 容忍度（度）
) -> torch.Tensor:
    """
    带死区的平稳奖励：
    在 tolerance_deg 范围内不惩罚，超过后施加平方惩罚。
    这允许车身随地形有轻微起伏，但防止剧烈倾斜。
    """
    # 1. 获取投影重力 (0, 0, -1) 在机体系下的向量 [N, 3]
    # 如果车身水平，proj_grav 约为 (0, 0, -1)
    proj_grav = env.scene[asset_cfg.name].data.projected_gravity_b
    
    # 2. 提取 x, y 分量 (对应 Roll 和 Pitch 的倾斜程度)
    grav_xy = proj_grav[:, :2]  # [N, 2]
    
    # 3. 计算当前的倾斜幅度 (约为 sin(theta))
    tilt_magnitude = torch.norm(grav_xy, dim=1) # [N]
    
    # 4. 计算死区阈值 (sin(tolerance))
    threshold = torch.sin(torch.tensor(tolerance_deg * 3.14159 / 180.0, device=env.device))
    
    # 5. 计算超出部分
    excess_tilt = torch.clamp(tilt_magnitude - threshold, min=0.0)
    
    # 6. 返回 L2 惩罚 (平方)
    #return torch.sum(excess_tilt**2, dim=0) # 返回形状通常需要匹配 [N] 或标量，这里建议直接返回 excess_tilt 的平方
    # 注意：IsaacLab 的 RewTerm 会自动处理 batch 维度，通常返回 [N]
    return excess_tilt ** 2

def true_wheel_slip_l2_with_deadzone(
    env,
    wheel_body_names: list = ["w_lf", "w_rf", "w_lb", "w_rb"],
    wheel_radius: float = 0.19,
    slip_tolerance: float = 0.1,  # 允许的合法滑移死区 (m/s)
) -> torch.Tensor:
    """
    终极物理打滑惩罚 (致敬你的物理直觉！)：
    通过提取机身坐标系下 X-Z 平面的合成速度，完美还原轮子沿地形切线的真实滚动速度。
    """
    asset = env.scene["robot"]
    
    body_ids, _ = asset.find_bodies(wheel_body_names)
    joint_ids, _ = asset.find_joints(wheel_body_names)
    
    # 1. 理想前向线速度 (如果之前确认方向同向，直接相乘)
    omega = asset.data.joint_vel[:, joint_ids]
    ideal_forward_vel = omega * wheel_radius
    
    # 2. 获取轮芯在世界系下的真实线速度 [N, 4, 3]
    vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    
   # 3. 将世界速度转换到【机身坐标系(Base Frame)】
    base_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    
    # 使用官方推荐的高效新函数 quat_apply_inverse 替换 quat_rotate_inverse
    vel_b = math_utils.quat_apply_inverse(
        base_quat.reshape(-1, 4), 
        vel_w.reshape(-1, 3)
    ).reshape(-1, len(body_ids), 3)
    
    # ★ 4. 核心修正：计算 X-Z 平面的合成速度 (即沿地形切线的真实滚动速度)
    # 提取 X 分量和 Z 分量
    vel_b_x = vel_b[:, :, 0]
    vel_b_z = vel_b[:, :, 2]
    
    # 勾股定理求合成速度的大小，并用 sign(vel_b_x) 赋予其正确的前进/后退方向
    real_forward_vel = torch.sign(vel_b_x) * torch.sqrt(vel_b_x**2 + vel_b_z**2)
    
    # 5. 计算速度差的绝对值
    slip_error = torch.abs(real_forward_vel - ideal_forward_vel)
    
    # 6. 引入死区：只惩罚超出 tolerance 的打滑部分 (包容正常转弯时的横向拖拽导致的速度损耗)
    excess_slip = torch.clamp(slip_error - slip_tolerance, min=0.0)
    
    # 7. 平方惩罚
    slip_penalty = torch.sum(excess_slip**2, dim=1)
    
    return slip_penalty
    

def lateral_slip_l2_with_kinematic_deadzone(
    env,
    wheel_body_names: list = ["w_lf", "w_rf", "w_lb", "w_rb"],
    base_tolerance: float = 0.02,      # 基础死区
    kinematic_coef: float = 0.5,       # 运动学补偿：直接填入机器人的半轴距 (Half Wheelbase，单位：米)
) -> torch.Tensor:
    """
    针对低速速差转向的横向侧滑惩罚：
    仅补偿纯运动学旋转带来的强制侧滑，严厉打击任何不合理的横向滑动。
    """
    asset = env.scene["robot"]
    body_ids, _ = asset.find_bodies(wheel_body_names)
    
    # 1. 获取速度并转到机身坐标系
    vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    base_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    vel_b = math_utils.quat_apply_inverse(
        base_quat.reshape(-1, 4), 
        vel_w.reshape(-1, 3)
    ).reshape(-1, len(body_ids), 3)
    
    # 2. 提取横向侧滑速度 (Y轴)
    vel_b_y = vel_b[:, :, 1]
    lateral_slip_error = torch.abs(vel_b_y)
    
    # 3. 提取机身偏航角速度 (Yaw Rate)
    base_ang_vel_w = asset.data.root_ang_vel_w
    base_ang_vel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, base_ang_vel_w)
    yaw_rate = torch.abs(base_ang_vel_b[:, 2])  # [N]
    
    # 4. 计算纯运动学动态死区
    # 死区 = 基础容忍度 + (半轴距 * 旋转角速度)
    dynamic_tolerance = base_tolerance + kinematic_coef * yaw_rate
    dynamic_tolerance = dynamic_tolerance.unsqueeze(1) # [N, 1] 以对齐 4 个轮子
    
    # 5. 应用死区并进行平方惩罚
    excess_lateral_slip = torch.clamp(lateral_slip_error - dynamic_tolerance, min=0.0)
    lateral_slip_penalty = torch.sum(excess_lateral_slip**2, dim=1)
    
    return lateral_slip_penalty


@configclass
class SkidSteerLegRewardsCfg:
    """
    与 SkidSteerLegAction 对齐的奖励配置。
    - 命令名请与 CommandManager 中保持一致（默认 "base_vel"；如你用 "base_velocity" 请改成相同名字）。
    - wheel_cfg/几何参数需与动作项一致（轮距 W、半径 r、wheel 关节正则）。
    """

    # 1） 速度跟踪（指数核）[4]
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.rewards.track_lin_vel_xy_exp,
        params={"command_name": "base_velocity", "std": 0.5},  # std 越小，偏差罚得越快
        weight=2.0,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.rewards.track_ang_vel_z_exp,
        params={"command_name": "base_velocity", "std": 0.5},
        weight=2.0,
    )

    # 2) 车身稳定/抑制弹跳 [4]
    flat_orientation_l2 = RewTerm(func=flat_orientation_with_tolerance, weight=-1000.0)
    ang_vel_xy_l2       = RewTerm(func=mdp.rewards.ang_vel_xy_l2,       weight=-0.1)
    lin_vel_z_l2        = RewTerm(func=mdp.rewards.lin_vel_z_l2,        weight=-0.1)

    # 3) 调距关节使用与平滑（自定义）
    leg_center_l2 = RewTerm(
        func=leg_pos_center_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")},
        weight=-0.05,
    )
    leg_speed_l2 = RewTerm(
        func=leg_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")},
        weight=-0.02,
    )
    '''
    # 4) 打滑一致性（自定义）=
    slip_consistency = RewTerm(
        func=slip_consistency_l2,
        params={
            "wheel_cfg": SceneEntityCfg("robot", joint_names="w_.*"),
            "base_width": 0.5,     # 与动作项 base_width 一致
            "wheel_radius": 0.05,  # 与动作项 wheel_radius 一致
        },
        weight=-20,
    )
    '''
    
    # ★ 新增：真实物理打滑/空转惩罚
    true_wheel_slip = RewTerm(
        func=true_wheel_slip_l2_with_deadzone,
        params={
            "wheel_body_names": ["w_lf", "w_rf", "w_lb", "w_rb"],
            "wheel_radius": 0.19, 
            "slip_tolerance": 0.2
        },
        weight=-1.0,  # 这里的权重不需要像之前 -20 那么极端，-2 到 -5 之间就足够让它学会踩地了
    )
    
    true_wheel_slip2 = RewTerm(
        func=lateral_slip_l2_with_kinematic_deadzone,
        params={
            "wheel_body_names": ["w_lf", "w_rf", "w_lb", "w_rb"],
            "base_tolerance": 0.02, 
            "kinematic_coef": 0.5
        },
        weight=-1.0,  # 这里的权重不需要像之前 -20 那么极端，-2 到 -5 之间就足够让它学会踩地了
    )
    
    residual_mean_pen = RewTerm(
        func=residual_mean_penalty,
        params={"residual_indices": [2, 3, 4, 5]},
        weight=-1.0, 
    )

    # 5) 能耗与控制平滑 [4]
    dof_torques_l2 = RewTerm(func=mdp.rewards.joint_torques_l2, weight=-2.5e-6)
    dof_acc_l2     = RewTerm(func=mdp.rewards.joint_acc_l2,     weight=-5.0e-4)
    action_rate_l2 = RewTerm(func=mdp.rewards.action_rate_l2,   weight=-0.05)

    # 6) 可选：卡住终止的惩罚（依赖 TerminationManager 的 "stuck" 条目）[4][5]
    #stuck_penalty = RewTerm(
    #    func=mdp.rewards.is_terminated_term,
    #    params={"term_keys": "time_out"},
    #    weight=-5.0,
    #)
    contact_penalty = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": "base_contact"},
        weight=-10.0,
    )
    
    # 新增: 足端离地惩罚=
    feet_air_time = RewTerm(
        func=feet_air_time_l2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["w_lf", "w_rf", "w_lb", "w_rb"],  # 四个轮子 link
            ),
            "max_air_time": 0.5,
        },
        weight=-0.1,   # 先给一个比较温和的权重，后面看效果再调
    )

    #log_pitch_monitor = RewTerm(
    #    func=log_base_pitch, # 指向上面定义的函数
    #    weight=-0.000573,                      # 权重为 0，不影响训练
    #)
    
    
    #可以通过观察机器人训练过程中的奖励数量级来分析权重值是否合理
import torch
import torch.nn.functional as F
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg

# ---------------------------
# 自定义奖励项
# ---------------------------

def leg_pos_center_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节“偏离默认姿态”惩罚"""
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_q = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    
    low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    high= asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    half= torch.clamp(0.5 * (high - low), min=1e-6)
    
    qn = torch.clamp((q - default_q) / half, -1.0, 1.0)
    return torch.sum(qn**2, dim=1)

def leg_vel_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节速度惩罚（平方和）。"""
    return torch.sum(mdp.joint_vel(env, asset_cfg)**2, dim=1)

def feet_air_time_l2(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "contact_forces",
        body_names=["w_lf", "w_rf", "w_lb", "w_rb"],  
    ),
    max_air_time: float = 0.1,
) -> torch.Tensor:
    """足端离地惩罚"""
    sensor = env.scene[sensor_cfg.name]      
    data = sensor.data                       

    if data.current_air_time is None:
        raise RuntimeError("contact_forces 传感器未开启 track_air_time=True")

    foot_ids = sensor_cfg.body_ids           
    air_time = data.current_air_time[:, foot_ids]

    excess = torch.clamp(air_time - max_air_time, min=0.0)
    return torch.sum(excess**2, dim=1)       

def log_base_pitch(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """记录车身俯仰角(Pitch)的绝对值 (rad)"""
    rot = mdp.root_quat_w(env, asset_cfg)
    _, pitch, _ = math_utils.euler_xyz_from_quat(rot)
    return torch.abs(pitch)

def flat_orientation_with_tolerance(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance_deg: float = 2.0, 
    beta: float = 0.1             
) -> torch.Tensor:
    """带死区的平稳奖励 (Smooth L1)"""
    proj_grav = env.scene[asset_cfg.name].data.projected_gravity_b
    grav_xy = proj_grav[:, :2]  
    tilt_magnitude = torch.norm(grav_xy, dim=1) 
    
    threshold = torch.sin(torch.tensor(tolerance_deg * 3.14159 / 180.0, device=env.device))
    excess_tilt = torch.clamp(tilt_magnitude - threshold, min=0.0)
    
    penalty = F.smooth_l1_loss(excess_tilt, torch.zeros_like(excess_tilt), reduction='none', beta=beta)
    return penalty

def true_wheel_slip_l2_with_smart_deadzone(
    env,
    wheel_body_names: list = ["w_lf", "w_rf", "w_lb", "w_rb"],
    wheel_radius: float = 0.19,
    base_tolerance: float = 0.1,  
    turn_allowance: float = 0.8,   
    beta=0.2
) -> torch.Tensor:
    """真实物理打滑惩罚 (带智能死区)"""
    asset = env.scene["robot"]
    body_ids, _ = asset.find_bodies(wheel_body_names)
    joint_ids, _ = asset.find_joints(wheel_body_names)
    
    omega = asset.data.joint_vel[:, joint_ids]
    ideal_forward_vel = omega * wheel_radius
    
    vel_w = asset.data.body_lin_vel_w[:, body_ids, :]
    base_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, len(body_ids), -1)
    vel_b = math_utils.quat_apply_inverse(
        base_quat.reshape(-1, 4), 
        vel_w.reshape(-1, 3)
    ).reshape(-1, len(body_ids), 3)
    
    real_forward_vel = torch.sign(vel_b[:, :, 0]) * torch.sqrt(vel_b[:, :, 0]**2 + vel_b[:, :, 2]**2)
    slip_error = torch.abs(real_forward_vel - ideal_forward_vel)
    
    base_ang_vel_w = asset.data.root_ang_vel_w
    base_ang_vel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, base_ang_vel_w)
    yaw_rate = torch.abs(base_ang_vel_b[:, 2]) 
    
    smart_tolerance = base_tolerance + turn_allowance * yaw_rate
    smart_tolerance = smart_tolerance.unsqueeze(1) 
    
    excess_slip = torch.clamp(slip_error - smart_tolerance, min=0.0)
    penalty = F.smooth_l1_loss(excess_slip, torch.zeros_like(excess_slip), reduction='none', beta=beta)
    
    return torch.sum(penalty, dim=1)

def action_rate_l1(env) -> torch.Tensor:
    """动作帧间变化率的 L1 惩罚 (绝对值)"""
    current_action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    return torch.sum(torch.abs(current_action - prev_action), dim=1)

# ---------------------------
# 奖励配置
# ---------------------------

@configclass
class SkidSteerLegRewardsCfg:
    """与 SkidSteerLegAction 对齐的奖励配置。"""

    # 1) 速度跟踪
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.rewards.track_lin_vel_xy_exp,
        params={"command_name": "base_velocity", "std": 0.6},  
        weight=3.0,
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.rewards.track_ang_vel_z_exp,
        params={"command_name": "base_velocity", "std": 0.6},
        weight=3.0,
    )
    
    # 动作变化率平滑 (取代之前的残差平滑)
    action_rate_l1_pen = RewTerm(func=action_rate_l1, weight=-0)
    action_rate_l2 = RewTerm(func=mdp.rewards.action_rate_l2, weight=-0)

    # 2) 车身稳定
    flat_orientation_l2 = RewTerm(func=flat_orientation_with_tolerance, weight=0)
    ang_vel_xy_l2       = RewTerm(func=mdp.rewards.ang_vel_xy_l2,       weight=0)
    lin_vel_z_l2        = RewTerm(func=mdp.rewards.lin_vel_z_l2,        weight=-0)

    # 3) 调距关节使用与平滑
    leg_center_l2 = RewTerm(
        func=leg_pos_center_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")},
        weight=-0.0,
    )
    leg_speed_l2 = RewTerm(
        func=leg_vel_l2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")},
        weight=-0.0,
    )
    
    # 4) 物理打滑/空转惩罚
    true_wheel_slip = RewTerm(
        func=true_wheel_slip_l2_with_smart_deadzone,
        params={
            "wheel_body_names": ["w_lf", "w_rf", "w_lb", "w_rb"],
            "wheel_radius": 0.19, 
            "base_tolerance": 0.08,
            "turn_allowance": 0.3
        },
        weight=-0,  
    )

    # 5) 能耗与控制平滑
    dof_torques_l2 = RewTerm(func=mdp.rewards.joint_torques_l2, weight=0)
    dof_acc_l2     = RewTerm(func=mdp.rewards.joint_acc_l2,     weight=-0)

    # 6) 终止惩罚
    contact_penalty = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": "base_contact"},
        weight=-10.0,
    )
    
    feet_air_time = RewTerm(
        func=feet_air_time_l2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["w_lf", "w_rf", "w_lb", "w_rb"],  
            ),
            "max_air_time": 0.05,
        },
        weight=-1,   
    )
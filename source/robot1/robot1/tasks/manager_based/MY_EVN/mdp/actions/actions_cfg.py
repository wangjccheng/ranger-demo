from dataclasses import MISSING
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import skid_steer_leg_actions

# 文件路径: .../MY_EVN/mdp/actions/actions_cfg.py

@configclass
class SkidSteerLegActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = skid_steer_leg_actions.SkidSteerLegAction

    asset_name: str = "robot"

    # ==========================================
    # ★ 修改点：删除运动学参数 base_width, wheel_radius, base_scale 等
    # ★ 新增：直接控制轮子的转速映射系数
    # 假设你的轮子期望最大转速是 10 rad/s (对应约 1.9 m/s)
    # ==========================================
    wheel_scale: float = 40.0   
    wheel_offset: float = 0.0  

    left_wheel_joint_names: list[str] = MISSING 
    right_wheel_joint_names: list[str] = MISSING 
    leg_joint_names: list[str] | str = MISSING

    bounding_strategy: str | None = "clip"      

    leg_rescale_to_limits: bool = False           
    leg_scale: float = 0.30                      
    leg_offset: float = 0.05                      
    
    eha_lag_alpha: float = 0.5
    actuator_lag_alpha: float = 0.8
    
    # ==========================================
    # ★ 方案B 新增：每帧动作(Raw Action)的最大允许变化量
    # 假设控制频率是 50Hz (0.02s)，如果设为 0.1，
    # 意味着从 0 加速到 1 (满速) 至少需要 10 帧 (0.2秒)，非常安全平滑。
    # ==========================================
    max_action_delta: float = 0.1


@configclass
class ActionsCfg:
    skid_steer_leg = SkidSteerLegActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["w_lf", "w_lb"],
        right_wheel_joint_names=["w_rf", "w_rb"],
        leg_joint_names=["g_lf", "g_rf", "g_lb", "g_rb"], 
        
        # ★ 传入新的参数
        wheel_scale=40.0,
        
        bounding_strategy="clip",
        leg_rescale_to_limits=False,  
        eha_lag_alpha=1.0,
        actuator_lag_alpha=1.0,
        max_action_delta = 0.2
    )
from dataclasses import MISSING
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import skid_steer_leg_actions

@configclass
class SkidSteerLegActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = skid_steer_leg_actions.SkidSteerLegAction

    # 资产
    asset_name: str = "robot"

    # 底盘几何
    base_width: float = 0.5     # 轮距 W（左右轮中心距）
    wheel_radius: float = 0.05  # 轮半径 r

    # 轮子关节名（保序）：建议显式列出，或提供能唯一匹配的正则
    left_wheel_joint_names: list[str] = MISSING  # 例如 ["front_left_wheel_throttle", "back_left_wheel_throttle"]
    right_wheel_joint_names: list[str] = MISSING # 例如 ["front_right_wheel_throttle","back_right_wheel_throttle"]

    # 调距关节名：列表或正则表达式（如 "g_.*"）
    leg_joint_names: list[str] | str = MISSING

    # 底盘标定/约束
    base_scale: tuple[float, float] = (5.0, 3.0)  # 分别为 v 与 omega 的缩放
    base_offset: tuple[float, float] = (0.0, 0.0)
    bounding_strategy: str | None = "clip"       # "clip"/"tanh"/None
    no_reverse: bool = False                     # True 则 v>=0

    # 调距关节映射：二选一
    leg_rescale_to_limits: bool = True           # True 时用软限把 [-1,1] 反归一化到实际范围
    leg_scale: float = 1.0                       # False 时，线性映射的缩放
    leg_offset: float = 0.0                      # False 时，线性映射的偏置
    # +++ 【新增】: 低通滤波系数 (模拟响应延迟) +++
    # alpha 范围 (0, 1]。1.0 代表理想执行器(无延迟)，数值越小延迟/惯性越大
    action_alpha: float = 0.3
    
    delay_steps_min: int = 1
    delay_steps_max: int = 3

@configclass
class ActionsCfg:
    skid_steer_leg = SkidSteerLegActionCfg(
        asset_name="robot",
        left_wheel_joint_names=["w_lf", "w_lb"],
        right_wheel_joint_names=["w_rf", "w_rb"],
        leg_joint_names=["g_lf", "g_rf", "g_lb", "g_rb"],  # 或 "g_.*"
        base_width=0.5,
        wheel_radius=0.05,
        base_scale=(1.0, 1.0),
        bounding_strategy="clip",
        no_reverse=False,
        leg_rescale_to_limits=True,  # 建议用软限反归一化
        # +++ 【新增】 +++
        action_alpha=0.3, # 建议从 0.3 或 0.4 开始训练
    )
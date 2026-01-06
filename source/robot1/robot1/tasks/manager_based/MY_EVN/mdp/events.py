from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
import isaaclab.envs.mdp as mdp

@configclass
class SkidSteerLegEventsCfg:
    # ---------- startup：域随机化（只在启动生效） ----------
    wheel_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # 物理材质“桶”随机分配 [2]
        mode="startup",
        params={
            # 匹配全部“车轮刚体”，示例正则请按你的模型改：如 ".*wheel.*link" 或 "w_.*"
            "asset_cfg": SceneEntityCfg("robot", body_names=["w_.*"]),
            "static_friction_range": (0.5, 0.8),
            "dynamic_friction_range": (0.4, 0.7),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 16,
            "make_consistent": True,  # 动摩擦<=静摩擦 [2]
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,  # 质量扰动并重算惯量 [2]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (-2.0, 3.0),  # 加法扰动
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,  # 质心微扰 [2]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.01, 0.01)},
        },
    )
    leg_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,  # 执行器增益扰动（调距关节）[2]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="g_.*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    leg_joint_params = EventTerm(
        func=mdp.randomize_joint_parameters,  # 关节模型参数扰动（调距关节）[2]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="g_.*"),
            "friction_distribution_params": (0.0, 0.05),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # ---------- reset：每回合重置 ----------
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,  # 根位姿/速度随机 [2]
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    # 若使用“跑道重置”或“地形合法点重置”，可把上面 reset_base 换成你自定义的 reset_root_state_along_track 或 mdp.reset_root_state_from_terrain [2]

    reset_leg_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # 调距关节回中心小偏置 [2]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="g_.*"),
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.0, 0.0),
        },
    )
    reset_wheel_joints = EventTerm(
        func=mdp.reset_joints_by_offset,  # 轮速归零/极小偏置 [2]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="w_.*"),
            "position_range": (0.0, 0.0),    # 通常轮子是速度控制，位置不变
            "velocity_range": (-0.0, 0.0),   # 清零
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,  # 可留为 0（占位），随时改成小推力 [2]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )

    # ---------- interval：周期扰动 ----------
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,  # 速度脉冲（比力更简洁稳定）[2]
        mode="interval",
        interval_range_s=(8.0, 12.0),      # 每 8–12 s 触发一次（每个 env 独立计时）[18]
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.4, 0.4), "yaw": (-0.5, 0.5)}},
    )

    # 可选：全局重力轻微随机化（不分 env；建议仅在鲁棒性实验中打开）[2]
    # randomize_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     interval_range_s=(20.0, 30.0),
    #     params={
    #         "gravity_distribution_params": ([-0.2, -0.2, -9.91], [0.2, 0.2, -9.69]),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

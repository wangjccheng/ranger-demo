

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/wjc/robot1/urdf/usd/urdf/urdf copy.usd", 
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=10000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            # [Sim2Real 修改] 提升解算器迭代次数
            # EHA 控制需要极高的物理保真度，建议提升迭代次数防止关节“拉脱”
            solver_position_iteration_count=8,  
            solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "w_lb": 0.0,
            "g_lf": 0.25,
            "w_lf": 0.0,
            "g_rf": 0.25,
            "g_rb": 0.25,
            "w_rf": 0.0,
            "w_rb": 0.0,
            "g_lb": 0.25,
        },
        joint_vel={".*": 0.0},
    ),
    
    # [Sim2Real 修改] 软限位放开
    # 放开物理引擎的软限位约束，让前面我们自己写的 Reward 函数 (0.9 阈值) 发挥作用
    soft_joint_pos_limit_factor=0.95, 
    
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "g_lb", "g_lf", "g_rf", "g_rb",
            ],
            # [Sim2Real 修改] 匹配 EHA 的真实力矩 (通常比轮子大)
            effort_limit=500.0,  
            velocity_limit=3.0,
            
            # [极其关键的修改] 必须清零！
            # 控制权已完全移交至 skid_steer_leg_actions.py 里的 EHA 阻抗模型
            stiffness={".*": 0.0},
            damping={".*": 0.0},
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "w_lb", "w_lf", "w_rf", "w_rb",
            ],
            # [Sim2Real 修改] 匹配轮毂电机的真实力矩，1000太大了
            effort_limit=80.0,  
            velocity_limit=100.0,
            
            # [极其关键的修改] 必须清零！
            # 这里的阻尼会和 actions.py 里的 wheel_damping 严重打架导致锁死
            stiffness={".*": 0.0},
            damping={".*": 0.0},
        ),
    },
)





'''







import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg,DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ROBOT1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/wjc/robot1/urdf/usd/urdf/urdf copy.usd",          # 替换为你前面导出的机器人usd文件路径
        activate_contact_sensors=True,          #地面接触检测
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            #rigid_body_enabled=True, # 开启刚体
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=10000.0,
            max_angular_velocity=10000.0,
            max_depenetration_velocity=10000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
  
        pos=(0.0, 0.0, 0.7), # 机器人初始位置，具体高度可由脚本check_op3.py查看

        # 机器人各关节初始角度，这里对应urdf中的joint,所有可动的joint都需要写进来

        joint_pos={
            "w_lb": 0.0,
            "g_lf": 0.25,
            "w_lf": 0.0,
            "g_rf": 0.25,
            "g_rb": 0.25,
            "w_rf": 0.0,
            "w_rb": 0.0,
            "g_lb": 0.25,
        },
        joint_vel={".*": 0.0},
    ),
    
    soft_joint_pos_limit_factor=0.7,
    
    ## 同样的，将下面的关节名替换为你的机器人的关节joint名，其中，比如你的机器人没有手，可以将"arms"去掉

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
            "g_lb",
            "g_lf",
            "g_rf",
            "g_rb",
            ],
            effort_limit_sim=100,
            velocity_limit_sim=50.0,
            stiffness={
            "g_lb": 0.001,
            "g_lf": 0.001,
            "g_rf": 0.001,
            "g_rb": 0.001,
            },
            #200
            damping={
            "g_lb": 40.0,
            "g_lf": 40.0,
            "g_rf": 40.0,
            "g_rb": 40.0,
            },
            #30
                ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
            "w_lb",
            "w_lf",
            "w_rf",
            "w_rb",
            ],
            #saturation_effort=1000.0,
            effort_limit=1000,
            velocity_limit=1000.0,
            stiffness={
            "w_lb": 0.001,
            "w_lf": 0.001,
            "w_rf": 0.001,
            "w_rb": 0.001,
            },
            damping={
            "w_lb": 80.0,
            "w_lf": 80.0,
            "w_rf": 80.0,
            "w_rb": 80.0,
            },
                ),
    },
)   
'''




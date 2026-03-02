import torch
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise, AdditiveUniformNoiseCfg as Unoise

# ==============================================================================
# 自定义观测辅助函数 (Helper Functions)
# ==============================================================================

def masked_height_scan(env, sensor_cfg: SceneEntityCfg, mask_region: str = "sides"):
    """
    带遮罩的高度扫描：模拟传感器盲区或遮挡。
    """
    sensor = env.scene[sensor_cfg.name]

    # ray_hits_w 形状: [N, num_rays, 3] -> 取 Z 轴
    raw_heights = sensor.data.ray_hits_w[:, :, 2] - env.scene["robot"].data.root_pos_w[:, 2:3]
    
    # 将 NaN 替换为 -2.0 (深坑值)
    raw_heights = torch.nan_to_num(raw_heights, nan=-2.0)

    # Reshape 为 2D 网格结构
    N = raw_heights.shape[0]
    num_rays = raw_heights.shape[1]
    grid_size = int(torch.sqrt(torch.tensor(num_rays)).item())  # 20
    grid_h = raw_heights.view(N, grid_size, grid_size)
    
    # 中心点索引 (10)
    center = grid_size // 2

    if mask_region == "sides":
        # --- 参数定义区（可根据真实机器人的长宽微调） ---
        # 1. 前后轮之间的范围 (X轴，行)
        # 假设前后轮之间总长 0.8m，即中心前后各 0.4m (4个格子)
        x_start = center - 4  # 索引 6
        x_end = center + 4    # 索引 14
        
        # 2. 从两侧轮子向外的范围 (Y轴，列)
        # 轮距是 0.5m，即中心左右各 0.25m (2.5个格子)。我们从第3个格子开始向外抹黑
        y_right_end = center - 6  # 索引 0 到 7 (右侧向外)
        y_left_start = center + 6 # 索引 13 到 20 (左侧向外)

        # --- 执行切片抹黑 (赋值为 -2.0) ---
        # 抹黑一侧的方块
        grid_h[:, x_start:x_end, :y_right_end] = -2.0
        
        # 抹黑另一侧的方块
        grid_h[:, x_start:x_end, y_left_start:] = -2.0

    elif mask_region == "front":
        # 保留原有的前方遮挡选项
        grid_h[:, center:, center-5:center+5] = -2.0 
        
    elif mask_region == "random_dropout":
        # 随机丢失噪点
        mask = torch.rand_like(grid_h) < 0.1
        grid_h[mask] = -2.0

    return grid_h.view(N, -1)

def slip_features(env,
                  wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="w.*"),
                  base_width: float = 0.5,
                  wheel_radius: float = 0.05) -> torch.Tensor:
    """
    打滑特征检测：计算“理论速度”与“真实速度”的偏差。
    【注意】包含特权信息 v_b，仅供 Critic 使用。
    """
    # 真实基座速度 (机体系) - 这是真机上未知的特权信息
    v_b = mdp.base_lin_vel(env)[:, 0]          
    w_b = mdp.base_ang_vel(env)[:, 2]          
    
    # 轮子角速度
    w_all = mdp.joint_vel(env, wheel_cfg)      
    n = w_all.shape[1] // 2                    
    
    wl = torch.mean(w_all[:, :n], dim=1)       
    wr = torch.mean(w_all[:, n:], dim=1)       
    
    v_hat = wheel_radius * 0.5 * (wr + wl)     
    w_hat = wheel_radius / base_width * (wr - wl) 
    
    dv = v_hat - v_b
    dw = w_hat - w_b
    
    return torch.stack([v_hat, w_hat, dv, dw, dv.abs(), dw.abs()], dim=-1)

def cmd_vel_2d(env, command_name: str = "base_velocity") -> torch.Tensor:
    """提取 2D 速度指令 (v_x, ω_z)"""
    cmd = mdp.generated_commands(env, command_name)
    return torch.stack((cmd[:, 0], cmd[:, 2]), dim=-1)

def wheel_joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取轮子关节速度"""
    return mdp.joint_vel(env, asset_cfg)

def leg_joint_pos(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取腿部关节位置"""
    return mdp.joint_pos(env, asset_cfg)

def leg_joint_vel(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """获取腿部关节速度"""
    return mdp.joint_vel(env, asset_cfg)

'''
def leg_pos_normalized(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """将关节位置归一化到 [-1, 1] (基于软限位)"""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    high = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low) + 1e-6
    return torch.clamp((pos - mid) / half, -1.0, 1.0)
'''

# ==============================================================================
# 观测配置类 (Observation Config)
# ==============================================================================

@configclass
class SkidSteerLegObsCfg:
    """
    非对称 Actor-Critic 观测配置（适配 GRU 架构）。
    """

    # --------------------------------------------------------------------------
    # 1. Policy (Actor) Group: 纯净的本体/外界感知，无特权信息
    # --------------------------------------------------------------------------
    @configclass
    class Policy(ObsGroup):
        # --- 本体感知 (Proprioception) ---
        # 【改进】移除了 base_lin_vel。逼迫 GRU 从角速度、重力和 last_action 中隐式推断线速度
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05), clip=(-10,10))
        projected_grav = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))

        # 指令
        cmd_vw = ObsTerm(func=cmd_vel_2d, params={"command_name": "base_velocity"})

        # 关节状态
        wheel_vel = ObsTerm(func=wheel_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="w_.*")}, clip=(-200, 200))
        leg_pos = ObsTerm(func=leg_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_vel = ObsTerm(func=leg_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        #leg_pos_norm = ObsTerm(func=leg_pos_normalized, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})

        # 【改进】移除了 slip_feat，因为它包含不可知的特权速度信息

        # 上一步动作 (对于 GRU 隐式建立系统动力学极其关键)
        last_action = ObsTerm(func=mdp.last_action, clip=(-1, 1))

        # --- 外界感知 (Exteroception) ---
        height_scan = ObsTerm(
            func=masked_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "mask_region": "sides"},
            clip=(-2.0, 2.0),
            noise=Gnoise(std=0.01),  # 适度噪声，增强鲁棒性
            
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True   # 开启噪声

    # --------------------------------------------------------------------------
    # 2. Critic Group: 特权观测 (Privileged Info)，上帝视角辅助 Value 评估
    # --------------------------------------------------------------------------
    @configclass
    class Critic(ObsGroup):
        # --- 真实状态 (Ground Truth) ---
        # Critic 拥有环境的全部真实物理状态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_grav = ObsTerm(func=mdp.projected_gravity, noise=None)
        
        # 关节信息
        wheel_vel = ObsTerm(func=wheel_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="w_.*")})
        leg_pos = ObsTerm(func=leg_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_vel = ObsTerm(func=leg_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        #leg_pos_norm = ObsTerm(func=leg_pos_normalized, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        
        cmd_vw = ObsTerm(func=cmd_vel_2d, params={"command_name": "base_velocity"})
        last_action = ObsTerm(func=mdp.last_action)

        # 【改进】特权物理特征：精确的打滑数据，只给 Critic 用来算 Value
        slip_feat = ObsTerm(func=slip_features, params={"wheel_cfg": SceneEntityCfg("robot", joint_names="w_.*"), "base_width": 0.5, "wheel_radius": 0.05})

        # --- 特权外界感知 ---
        # Critic 看到无盲区、无噪声的高度图
        full_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-2.0, 2.0),
            noise=None
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False  # 关闭噪声

    # 注册组
    policy: Policy = Policy()
    critic: Critic = Critic()

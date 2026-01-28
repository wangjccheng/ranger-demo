import torch
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise, AdditiveUniformNoiseCfg as Unoise

# --- 基础工具 ---

import torch
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise, AdditiveUniformNoiseCfg as Unoise

# ==============================================================================
# 自定义观测辅助函数 (Helper Functions)
# ==============================================================================

def masked_height_scan(env, sensor_cfg: SceneEntityCfg, mask_region: str = "front"):
    """
    带遮罩的高度扫描：模拟传感器盲区或遮挡。
    Actor 只能看到这个版本，必须学会处理视野丢失的情况。
    """
    # 1. 获取原始扫描数据 (相对于机器人根节点的 Z 轴高度差)
    sensor = env.scene[sensor_cfg.name]
    # data.pos_w [N, num_rays, 3] -> 取 Z 轴; robot.data.root_pos_w [N, 3] -> 取 Z 轴
    # 结果: [N, num_rays]
    raw_heights = sensor.data.pos_w[:, :, 2] - env.scene.robot.data.root_pos_w[:, 2:3]
    
    # 2. Reshape 为 2D 网格结构以便进行空间操作
    # 假设 GridPattern 分辨率 0.1m, 范围 2.0x2.0m -> 20x20 网格
    # 注意：这里的 20x20 是基于假设，实际需与 robot1_env_cfg.py 中的 RayCaster 分辨率匹配
    N = raw_heights.shape[0]
    # 这里为了通用性，最好根据实际 ray 数计算，或者硬编码如果你确定尺寸
    # 假设 num_rays = 400
    grid_size = int(torch.sqrt(torch.tensor(raw_heights.shape[1])).item()) # e.g., 20
    grid_h = raw_heights.view(N, grid_size, grid_size)
    
    # 3. 应用遮罩 (Masking)
    if mask_region == "front":
        # 假设机器人面向 x 正方向，网格索引可能根据 pattern 定义有所不同
        # 这里假设上半部分是前方
        # 将前方 1m 范围内的感知数据抹零 (设为 0 或 -1 表示未知)
        center = grid_size // 2
        grid_h[:, center:, center-5:center+5] = 0.0 
    elif mask_region == "random_dropout":
        # 随机丢失 10% 的点
        mask = torch.rand_like(grid_h) < 0.1
        grid_h[mask] = 0.0

    # 4. 展平并返回
    return grid_h.view(N, -1)

def slip_features(env,
                  wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="w.*"),
                  base_width: float = 0.5,
                  wheel_radius: float = 0.05) -> torch.Tensor:
    """
    打滑特征检测：计算“理论速度”与“真实速度”的偏差。
    这对滑移转向机器人至关重要，能让网络感知到是否被卡住或在打滑。
    """
    # 真实基座速度 (机体系)
    v_b = mdp.base_lin_vel(env)[:, 0]          # v_x
    w_b = mdp.base_ang_vel(env)[:, 2]          # ω_z
    
    # 轮子角速度
    w_all = mdp.joint_vel(env, wheel_cfg)      # [N, 4]
    n = w_all.shape[1] // 2                    # 假设左右轮数量相等
    
    # 计算左右侧平均轮速
    wl = torch.mean(w_all[:, :n], dim=1)       # 左侧
    wr = torch.mean(w_all[:, n:], dim=1)       # 右侧
    
    # 根据运动学公式反推“理论速度”
    v_hat = wheel_radius * 0.5 * (wr + wl)     # 理论线速度
    w_hat = wheel_radius / base_width * (wr - wl) # 理论角速度
    
    # 计算偏差 (Delta)
    dv = v_hat - v_b
    dw = w_hat - w_b
    
    # 返回组合特征: [理论v, 理论w, 偏差v, 偏差w, 绝对偏差v, 绝对偏差w]
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

def leg_pos_normalized(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """将关节位置归一化到 [-1, 1] (基于软限位)"""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    high = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    
    mid = 0.5 * (low + high)
    half = 0.5 * (high - low) + 1e-6
    return torch.clamp((pos - mid) / half, -1.0, 1.0)


# ==============================================================================
# 观测配置类 (Observation Config)
# ==============================================================================

@configclass
class SkidSteerLegObsCfg:
    """
    非对称 Actor-Critic 观测配置。
    """

    # --------------------------------------------------------------------------
    # 1. Policy (Actor) Group: 受限观测，含噪声
    # --------------------------------------------------------------------------
    @configclass
    class Policy(ObsGroup):
        # --- 本体感知 (Proprioception) ---
        # 基础状态加噪声，模拟真实传感器
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Gnoise(std=0.05), clip=(-10,10))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gnoise(std=0.05), clip=(-10,10))
        projected_grav = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))

        # 指令
        cmd_vw = ObsTerm(func=cmd_vel_2d, params={"command_name": "base_velocity"})

        # 关节状态
        wheel_vel = ObsTerm(func=wheel_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="w_.*")}, clip=(-200, 200))
        leg_pos = ObsTerm(func=leg_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_vel = ObsTerm(func=leg_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_pos_norm = ObsTerm(func=leg_pos_normalized, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})

        # 打滑特征
        slip_feat = ObsTerm(func=slip_features,
                            params={"wheel_cfg": SceneEntityCfg("robot", joint_names="w_.*"),
                                    "base_width": 0.5, "wheel_radius": 0.05})

        # 上一步动作
        last_action = ObsTerm(func=mdp.last_action, clip=(-1, 1))

        # --- 外界感知 (Exteroception) ---
        # ★ 关键：这里使用的是 masked_height_scan (带遮挡/盲区)
        height_scan = ObsTerm(
            func=masked_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "mask_region": "front"},
            clip=(-2.0, 2.0)
        )

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True   # 开启噪声

    # --------------------------------------------------------------------------
    # 2. Critic Group: 特权观测 (Privileged Info)，无噪声，无遮挡
    # --------------------------------------------------------------------------
    @configclass
    class Critic(ObsGroup):
        # --- 真实状态 (Ground Truth) ---
        # Critic 看到的是无噪声的真实物理状态
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=None)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=None)
        projected_grav = ObsTerm(func=mdp.projected_gravity, noise=None)
        
        # 关节信息 (复制 Policy 的内容，但去掉噪声)
        wheel_vel = ObsTerm(func=wheel_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="w_.*")})
        leg_pos = ObsTerm(func=leg_joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_vel = ObsTerm(func=leg_joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        cmd_vw = ObsTerm(func=cmd_vel_2d, params={"command_name": "base_velocity"})
        slip_feat = ObsTerm(func=slip_features, params={"wheel_cfg": SceneEntityCfg("robot", joint_names="w_.*"), "base_width": 0.5, "wheel_radius": 0.05})
        last_action = ObsTerm(func=mdp.last_action)

        # --- 特权外界感知 ---
        # ★ 关键：Critic 看到的是 IsaacLab 原生的 height_scan (无遮挡，上帝视角)
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
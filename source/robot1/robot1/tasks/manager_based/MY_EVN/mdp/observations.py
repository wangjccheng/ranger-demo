import torch
import isaaclab.envs.mdp as mdp
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise, AdditiveUniformNoiseCfg as Unoise

# --- 基础工具 ---

def root_euler_xyz(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """世界系根姿态欧拉角 (x,y,z)，单位 rad。"""
    rx, ry, rz = math_utils.euler_xyz_from_quat(mdp.root_quat_w(env, asset_cfg))  # [3] 
    return torch.stack((rx, ry, rz), dim=-1)

def cmd_vel_2d(env, command_name: str = "base_vel") -> torch.Tensor:
    """SE(2) 速度指令 (v_x, v_y≈0, ω_z)，与动作项的 [v, ω] 对齐（取前两维：v_x, ω_z）。"""
    cmd = mdp.generated_commands(env, command_name)  # [N, 3]
    return torch.stack((cmd[:, 0], cmd[:, 2]), dim=-1)  # (v_x, ω_z)  [10]

def wheel_joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="w.*")) -> torch.Tensor:
    """驱动轮角速度（按 joint_names 正则匹配顺序返回）"""
    return mdp.joint_vel(env, asset_cfg)  # [N, #wheels]

def leg_joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节位置（按 'g_.*' 匹配调距关节）"""
    return mdp.joint_pos(env, asset_cfg)  # [N, #legs]

def leg_joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """调距关节速度"""
    return mdp.joint_vel(env, asset_cfg)  # [N, #legs]

def leg_pos_normalized(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="g_.*")) -> torch.Tensor:
    """将调距关节位置按软限归一化到 [-1, 1]，便于与动作（若采用 rescale_to_limits=True）对齐。"""
    asset = env.scene[asset_cfg.name]
    pos = asset.data.joint_pos[:, asset_cfg.joint_ids]               # [N, M]
    low = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    high = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    mid = 0.5 * (low + high)
    half = torch.clamp(0.5 * (high - low), min=1e-6)
    return torch.clamp((pos - mid) / half, -1.0, 1.0)

def slip_features(env,
                  wheel_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names="w.*"),
                  base_width: float = 0.5,
                  wheel_radius: float = 0.05) -> torch.Tensor:
    """
    简易打滑特征：由左右轮平均角速度反推 v̂ = r/2 (wr+wl)，ω̂ = r/W (wr-wl)，
    与真实 (v_x, ω_z) 的差值及其绝对值/平方，帮助策略判断打滑。
    约定：wheel_cfg 匹配的次序为 [左侧若干, 右侧若干]（与动作项配置一致）。
    """
    v_b = mdp.base_lin_vel(env)[:, 0]          # 真实 v_x（机体系）[3]
    w_b = mdp.base_ang_vel(env)[:, 2]          # 真实 ω_z（机体系）
    w_all = mdp.joint_vel(env, wheel_cfg)      # [N, nL+nR]
    n = w_all.shape[1] // 2
    wl = torch.mean(w_all[:, :n], dim=1)       # 左平均
    wr = torch.mean(w_all[:, n:], dim=1)       # 右平均
    v_hat = wheel_radius * 0.5 * (wr + wl)
    w_hat = wheel_radius / base_width * (wr - wl)
    dv = v_hat - v_b
    dw = w_hat - w_b
    return torch.stack([v_hat, w_hat, dv, dw, dv.abs(), dw.abs()], dim=-1)

# --- 与 SkidSteerLegAction 对齐的观测组 ---

@configclass
class SkidSteerLegObsCfg:
    """
    与二合一动作项（差速 + 调距位置）配套的观测配置。
    - 含底盘姿态/速度、投影重力、命令 (v, ω)、轮速、调距关节状态（pos/vel/归一化）、上一步动作、可选高度扫描与打滑特征。
    """
    @configclass
    class Policy(ObsGroup):
        # 根状态
        #root_pos_w      = ObsTerm(func=mdp.root_pos_w, noise=Gnoise(std=0.05))                 # [N,3]
        root_euler_xyz  = ObsTerm(func=root_euler_xyz,    noise=Gnoise(std=0.02))               # [N,3]
        base_lin_vel    = ObsTerm(func=mdp.base_lin_vel,  noise=Gnoise(std=0.05), clip=(-10,10)) # [N,3]
        base_ang_vel    = ObsTerm(func=mdp.base_ang_vel,  noise=Gnoise(std=0.05), clip=(-10,10)) # [N,3]
        projected_grav  = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))  # [N,3]

        # 指令（与动作 [v, ω] 对齐）
        cmd_vw          = ObsTerm(func=cmd_vel_2d, params={"command_name": "base_velocity"})          # [N,2]

        # 轮速（按关节名正则匹配；确保与动作项 wheel joints 一致）
        wheel_vel       = ObsTerm(func=wheel_joint_vel,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names="w_.*")},
                                   clip=(-200, 200))                                             # [N, nW]

        # 调距关节（与动作项 leg_joint_names 对齐）
        leg_pos         = ObsTerm(func=leg_joint_pos,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_vel         = ObsTerm(func=leg_joint_vel,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})
        leg_pos_norm    = ObsTerm(func=leg_pos_normalized,
                                   params={"asset_cfg": SceneEntityCfg("robot", joint_names="g_.*")})

        # 打滑/一致性特征（需传同一几何参数，默认与动作项一致）
        slip_feat       = ObsTerm(func=slip_features,
                                   params={"wheel_cfg": SceneEntityCfg("robot", joint_names="w_.*"),
                                           "base_width": 0.5,
                                           "wheel_radius": 0.05})

        # 上一步动作（完整拼接后的动作向量：2 + #legs）
        last_action     = ObsTerm(func=mdp.last_action, clip=(-1, 1))

        # 可选：高度扫描（若场景有 height_scanner）
        
        height_scan   = ObsTerm(func=mdp.height_scan,
                                  params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
                                  clip=(-2.0, 2.0), noise=Unoise(n_min=-0.05, n_max=0.05))

        def __post_init__(self):
            self.concatenate_terms = True     # 组内拼接成单一向量 [22]
            self.enable_corruption = False    # 关闭组级腐化（条目噪声仍生效）

    policy: Policy = Policy()
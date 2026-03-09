# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.managers import CurriculumTermCfg as CurrTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    # ★ 修改 1：升级条件不变（跑出当前地形块一半，即 > 5米 就可以升级）
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    
    # ★ 修改 2：大幅放宽降级门槛！
    # 因为有转向，直线距离本来就短。把系数 0.5 改成 0.15 甚至 0.1。
    # 意思是：只要你不是完全卡死在原地（连目标距离的 15% 都没达到），我就不给你留级！
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.15
    move_down *= ~move_up
    
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

def increase_reward_weight_over_time(
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        reward_term_name : str,
        increase : float,
        episodes_per_increase : int = 1,
        max_increases: int = torch.inf,
        ) -> torch.Tensor:
    """
    Increase the weight of a reward term after some amount of given time in episodes.
    Default amount of time is one episode.
    Stops increasing the weight after `stop_after_n_changes` changes. Defaults to inf.
    """
    num_episodes = env.common_step_counter // env.max_episode_length
    num_increases = num_episodes // episodes_per_increase

    if num_increases > max_increases:
        return # do nothing

    if env.common_step_counter % env.max_episode_length != 0:
        return # only process at the beginning of an episode (not per step)

    if (num_episodes + 1) % episodes_per_increase == 0: # discount the first episode
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight += increase
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)

# 添加到 curriculums.py 末尾

def anneal_reward_term_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    start_val: float,
    end_val: float,
    total_steps: int,
) -> torch.Tensor:
    """
    随着时间推移退火（改变）奖励项的参数（如 std）。
    在 total_steps 步内，将 param_name 的值从 start_val 线性插值到 end_val。
    """
    # 计算当前的进度 (0.0 到 1.0)
    current_step = env.common_step_counter
    
    # 如果超过了设定的步数，就固定在 end_val，不再修改
    if current_step >= total_steps:
        return

    # 线性插值计算新的值
    alpha = current_step / float(total_steps)
    new_val = start_val + (end_val - start_val) * alpha

    # 获取当前的配置并更新
    # 注意：这假设 set_term_cfg 开销不大，或者 IsaacLab 能够处理动态参数更新
    try:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # 只有值发生变化时才更新，减少开销
        if term_cfg.params.get(param_name) != new_val:
            term_cfg.params[param_name] = new_val
            env.reward_manager.set_term_cfg(term_name, term_cfg)
    except Exception as e:
        print(f"Error updating curriculum param for {term_name}: {e}")


def anneal_reward_term_weight(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_weight: float,
    end_weight: float,
    total_steps: int,
) -> None:
    """
    权重退火：随着训练步数，将指定奖励项的权重从 start_weight 线性过渡到 end_weight。
    """
    current_step = env.common_step_counter
    
    # 计算新权重
    if current_step >= total_steps:
        new_weight = end_weight
    else:
        alpha = current_step / float(total_steps)
        new_weight = start_weight + (end_weight - start_weight) * alpha

    try:
        # 1. 获取配置
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        
        # 2. 检查是否需要更新（避免每步都重复设置，节省开销）
        # 注意：浮点数比较最好用 math.isclose，或者直接比较
        if term_cfg.weight != new_weight:
            term_cfg.weight = new_weight
            
            # 3. ★★★ 关键修正：必须写回 Manager ★★★
            env.reward_manager.set_term_cfg(term_name, term_cfg)
            
    except Exception as e:
        # 建议打印错误，防止拼写错误导致课程未生效而你却不知道
        print(f"[Warning] Failed to update weight for {term_name}: {e}")
    
    return None        
        
@configclass
class SkidSteerLegCurriculumCfg:
    """课程学习配置：随着训练动态调整环境难度"""
    
    anneal_lin_vel_std = CurrTerm(
        func=anneal_reward_term_param,
        params={
            "term_name": "track_lin_vel_xy_exp", 
            "param_name": "std",
            "start_val": 0.8,           # 初始：允许 ±1m/s 的误差仍有较高奖励
            "end_val": 0.2,             # 最终：必须非常精准 (您原本的设定)
            "total_steps": 1.0e5,       # 在 2亿步(约一半训练程)内完成收紧
        },
    )
    
    # 如果角速度也难学，加上这个
    anneal_ang_vel_std = CurrTerm(
        func=anneal_reward_term_param,
        params={
            "term_name": "track_ang_vel_z_exp",
            "param_name": "std",
            "start_val": 0.8,
            "end_val": 0.2, 
            "total_steps": 1.0e5,
        },
    )
    anneal_flat_orientation_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "flat_orientation_l2",  # 对应 rewards 配置中的名字
            "start_weight": 0.0,                # 初期：轻微惩罚，允许它歪歪扭扭地跑
            "end_weight": -50.0,                # 后期：重罚，强迫它收敛到水平姿态
            "total_steps": 1.2e5,                # 在前 10万~20万步完成过渡
        },
    )

    # --- B. 惩罚项权重：由无到有 (weight: 0.0 -> -0.005) ---
    # 这解决了“因惧怕惩罚而不敢动”的问题
    # 针对 rewards.py [1] 中的惩罚项
    '''
    anneal_slip_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "slip_consistency", # 对应 rewards.py 中的变量名
            "start_weight": 0.0,             # 初始：不惩罚打滑
            "end_weight": -1,            # 最终：施加惩罚 (您原本的设定)
            "total_steps": 1.5e5,            # 较快引入惩罚(1亿步)，尽早规范动作
        },
    )
    '''
    
    residual_rate_pen = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "residual_rate_pen",     # 残差
            "start_weight": 0.0,
            "end_weight": -0.08,
            "total_steps": 1.2e5,
        },
    )
    residual_mean_pen = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "residual_mean_pen",     # 残差
            "start_weight": 0.0,
            "end_weight": -5,
            "total_steps": 1.0e5,
        },
    )
    
    true_wheel_slip = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "true_wheel_slip",     # 真实轮滑惩罚
            "start_weight": 0.0,
            "end_weight": -5,
            "total_steps": 1.2e5,
        },
    )
    
    
    anneal_bounce_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "lin_vel_z_l2",     # 抑制弹跳
            "start_weight": 0.0,
            "end_weight": -20,
            "total_steps": 1.2e5,
        },
    )

    anneal_tilt_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "ang_vel_xy_l2",    # 抑制倾斜
            "start_weight": 0.0,
            "end_weight": -0.1,
            "total_steps": 1.2e5,
        },
    )
    # 抑制调距关节速度
    led_speed_penalty = CurrTerm(func=anneal_reward_term_weight,params={"term_name": "leg_speed_l2", "start_weight": 0.0,"end_weight": -0.2,"total_steps": 1.3e5,},)
    
    
    led_center_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "leg_center_l2",    # 抑制调距关节偏离默认位置
            "start_weight": 0.0,
            "end_weight": -0.05,
            "total_steps": 1.4e5,
        },
    )
    '''  
    dof_torques_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "dof_torques_l2",    # 抑制扭矩
            "start_weight": 0.0,
            "end_weight": -2e-6,
            "total_steps": 2.2e5,
        },
    )
    '''
    action_rate_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "action_rate_l2",    # 抑制动作变化率
            "start_weight": 0.0,
            "end_weight": -2,
            "total_steps": 1.0e5,
        },
    )
    

    dof_acc_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "dof_acc_l2",    # 抑制加速度
            "start_weight": 0.0,
            "end_weight": -2.0e-7,
            "total_steps": 1.4e5,
        },
    )
    
    terrain_levels = CurrTerm(func=terrain_levels_vel)
    
    contact_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "contact_penalty",  # 对应 rewards 配置中的名字
            "start_weight": -0.2,                # 初期：不惩罚接触
            "end_weight": -10.0,                # 后期：重罚，强迫它避免接触障碍物
            "total_steps": 1.0e5,                # 在前 10万~20万步完成过渡
        },
    )
    
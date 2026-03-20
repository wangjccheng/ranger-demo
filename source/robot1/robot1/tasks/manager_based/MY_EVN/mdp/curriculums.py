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
    """根据机器人移动距离调整地形难度"""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
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
    """随回合数增加奖励权重"""
    num_episodes = env.common_step_counter // env.max_episode_length
    num_increases = num_episodes // episodes_per_increase

    if num_increases > max_increases:
        return 

    if env.common_step_counter % env.max_episode_length != 0:
        return 

    if (num_episodes + 1) % episodes_per_increase == 0: 
        term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        term_cfg.weight += increase
        env.reward_manager.set_term_cfg(reward_term_name, term_cfg)

def anneal_reward_term_param(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_name: str,
    start_val: float,
    end_val: float,
    total_steps: int,
) -> torch.Tensor:
    """奖励项参数(如 std)退火"""
    current_step = env.common_step_counter
    if current_step % 50 != 0 and current_step < total_steps:
        return None
    
    if current_step >= total_steps:
        return

    alpha = current_step / float(total_steps)
    new_val = start_val + (end_val - start_val) * alpha

    try:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if term_cfg.params.get(param_name) > 1e-6:
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
    """奖励项权重退火"""
    current_step = env.common_step_counter
    if current_step % 50 != 0 and current_step < total_steps:
        return None
        
    if current_step >= total_steps:
        new_weight = end_weight
    else:
        alpha = current_step / float(total_steps)
        new_weight = start_weight + (end_weight - start_weight) * alpha

    try:
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if abs(term_cfg.weight - new_weight) > 1e-6:
            term_cfg.weight = new_weight
            env.reward_manager.set_term_cfg(term_name, term_cfg)
    except Exception as e:
        print(f"[Warning] Failed to update weight for {term_name}: {e}")
    
    return None        
        
@configclass
class SkidSteerLegCurriculumCfg:
    """课程学习配置"""
    
    anneal_lin_vel_std = CurrTerm(
        func=anneal_reward_term_param,
        params={
            "term_name": "track_lin_vel_xy_exp", 
            "param_name": "std",
            "start_val": 0.8,           
            "end_val": 0.2,             
            "total_steps": 2.0e5,       
        },
    )

    anneal_ang_vel_std = CurrTerm(
        func=anneal_reward_term_param,
        params={
            "term_name": "track_ang_vel_z_exp",
            "param_name": "std",
            "start_val": 0.8,
            "end_val": 0.2, 
            "total_steps": 2.0e5,
        },
    )
    
    anneal_flat_orientation_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "flat_orientation_l2",  
            "start_weight": 0.0,                
            "end_weight": -20.0,                
            "total_steps": 2.0e5,                
        },
    )
    
    action_rate_l1_pen = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "action_rate_l1_pen",
            "start_weight": 0.0,
            "end_weight": -0.1,
            "total_steps": 2.0e5,
        },
    )
    
    led_speed_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "leg_speed_l2", 
            "start_weight": 0.0,
            "end_weight": -0.05,
            "total_steps": 2.0e5,
        },
    )

    terrain_levels = CurrTerm(func=terrain_levels_vel)
    
    contact_penalty = CurrTerm(
        func=anneal_reward_term_weight,
        params={
            "term_name": "contact_penalty",  
            "start_weight": -1.0,                
            "end_weight": -10.0,                
            "total_steps": 1.0e5,                
        },
    )
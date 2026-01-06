
# 可选：便捷暴露 mdp/agents；不导入 env cfg，避免循环
from . import mdp as _mdp  # noqa: F401
from . import agents as _agents  # noqa: F401
from . import register  # noqa: F401: 触发 gym.register
__all__ = ["mdp", "agents"]
'''
from .robot1_env_cfg import ROBOT1RoughEnvCfg
from . import agents
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

gym.register(
    id="sk-Robot1-v0",                      # 建议标准小写 v0
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # 环境配置类：模块路径:类名（按你文件实际类名填写）
        "env_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.robot1_env_cfg:Robot1RoughEnvCfg",

        # 各框架训练配置（按需保留/修改）
        "rl_games_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:sb3_ppo_cfg.yaml",
    },
)
'''
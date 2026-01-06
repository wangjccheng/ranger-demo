import gymnasium as gym

gym.register(
    id="sk-Robot1-v0",  # 注意大小写一致
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.robot1_env_cfg:ROBOT1RoughEnvCfg",
        "rl_games_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": "robot1.tasks.manager_based.MY_EVN.agents:sb3_ppo_cfg.yaml",
    },
)
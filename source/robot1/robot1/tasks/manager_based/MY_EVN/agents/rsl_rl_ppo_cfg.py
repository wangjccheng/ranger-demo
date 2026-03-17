# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlPpoActorCriticRecurrentCfg

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 3000
    save_interval = 500  # 建议改大一点，解决硬盘占用问题
    experiment_name = "youxia_manager"
    empirical_normalization = True
    
    # 使用带有循环层的网络配置
    policy = RslRlPpoActorCriticRecurrentCfg(
        # ★ 1. 注入我们的自定义网络类名
        class_name="CNNActorCriticRecurrent",
        
        init_noise_std=0.5,
        # MLP 部分：处理特征提取 (处理 CNN 潜变量和本体感知的拼接)
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        
        # ★ 2. RNN 部分：切换为 GRU
        rnn_type="gru",         
        rnn_hidden_dim=128,     
        rnn_num_layers=1,       
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=8,#降低显存使用，增加训练时间
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

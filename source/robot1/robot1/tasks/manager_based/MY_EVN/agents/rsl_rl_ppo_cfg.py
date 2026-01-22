# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg,RslRlPpoActorCriticRecurrentCfg


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 128
    max_iterations = 3500
    save_interval = 50
    experiment_name = "youxia_manager"
    empirical_normalization = True
# 使用带有循环层（RNN/LSTM）的网络配置
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=0.8,
        # MLP 部分：处理特征提取
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        # RNN 部分：处理时序记忆
        rnn_type="lstm",        # 可选 "lstm" 或 "gru"
        rnn_hidden_dim=256,     # 记忆单元的维度
        rnn_num_layers=2,       # 循环层的层数
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=3,
        num_mini_batches=2,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.003,
        max_grad_norm=1.0,
    )
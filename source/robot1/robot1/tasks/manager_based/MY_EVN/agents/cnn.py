import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent

class CNNActorCriticRecurrent(ActorCriticRecurrent):
    def __init__(self, num_actor_obs, num_critic_obs, num_actions, 
                 actor_hidden_dims=[256, 256, 256], critic_hidden_dims=[256, 256, 256], 
                 activation='elu', rnn_type='gru', rnn_hidden_size=256, rnn_num_layers=1, 
                 init_noise_std=1.0, **kwargs):
        
        # 你的雷达网格是 2m x 2m, res=0.1, 所以固定是 20x20 = 400 个点
        self.num_height_points = 400
        
        # CNN 降维后的潜变量维度
        self.latent_dim = 32
        
        # 计算原本的本体感知维度 (总维度 - 高程图维度)
        self.num_prop_actor = num_actor_obs - self.num_height_points
        self.num_prop_critic = num_critic_obs - self.num_height_points
        
        # 经过 CNN 融合后的实际 MLP 输入维度
        fused_actor_obs = self.num_prop_actor + self.latent_dim
        fused_critic_obs = self.num_prop_critic + self.latent_dim
        
        # 调用父类初始化，传入融合后的维度！这会让底层的 MLP 按新维度初始化
        super().__init__(
            num_actor_obs=fused_actor_obs,
            num_critic_obs=fused_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            rnn_type=rnn_type,
            rnn_hidden_size=rnn_hidden_size,
            rnn_num_layers=rnn_num_layers,
            init_noise_std=init_noise_std,
            **kwargs
        )
        
        # 定义极其轻量级的 CNN 编码器 (感受野和步长适配 20x20 的输入)
        # 包含两层卷积，将 20x20 降采样为 5x5
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, self.latent_dim),
            nn.ELU()
        )

    def process_obs(self, obs):
        prop_obs = obs[..., :-self.num_height_points]
        height_map = obs[..., -self.num_height_points:]
        
        # 展平输入 CNN
        height_map_2d = height_map.reshape(-1, 1, 20, 20)
        latent_terrain = self.cnn_encoder(height_map_2d)
        
        # ★ 显式根据输入维度还原形状，避免 trace 追踪时出现动态解包错误
        if obs.dim() == 3:
            # 训练更新阶段: [num_steps, num_envs, latent_dim]
            latent_terrain = latent_terrain.reshape(obs.shape[0], obs.shape[1], self.latent_dim)
        else:
            # Rollout 或 部署推理阶段: [num_envs, latent_dim]
            latent_terrain = latent_terrain.reshape(obs.shape[0], self.latent_dim)
            
        fused_obs = torch.cat([prop_obs, latent_terrain], dim=-1)
        return fused_obs

# ==========================================
    # 重写 RSL-RL 的核心前向传播函数
    # 加入 *args, **kwargs 完美兼容所有传参形式
    # ==========================================
    def act(self, observations, *args, **kwargs):
        fused_obs = self.process_obs(observations)
        return super().act(fused_obs, *args, **kwargs)

    def act_inference(self, observations, *args, **kwargs):
        fused_obs = self.process_obs(observations)
        return super().act_inference(fused_obs, *args, **kwargs)

    def evaluate(self, critic_observations, *args, **kwargs):
        fused_critic_obs = self.process_obs(critic_observations)
        return super().evaluate(fused_critic_obs, *args, **kwargs)
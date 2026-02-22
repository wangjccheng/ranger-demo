# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export custom CNN-GRU recurrent policy to a TorchScript (JIT) model."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# 添加 argparse 参数
parser = argparse.ArgumentParser(description="Export an RL agent with custom CNN-GRU to JIT.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--export_path", type=str, default="policy_gru_cnn.pt", help="Path to save the JIT model.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# 引入 RSL-RL 和 AppLauncher 的命令行参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 清理 sys.argv 以供 Hydra 解析
sys.argv = [sys.argv[0]] + hydra_args

# 启动 omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import robot1.tasks    # noqa: F401

# =====================================================================
# ★ 核心改动 1：在实例化 Runner 之前，注册我们的自定义网络类
# =====================================================================
import rsl_rl.runners.on_policy_runner
from robot1.tasks.manager_based.MY_EVN.agents.cnn import CNNActorCriticRecurrent
rsl_rl.runners.on_policy_runner.CNNActorCriticRecurrent = CNNActorCriticRecurrent


# =====================================================================
# ★ 核心改动 2：适配 GRU 的单张量隐状态 (Hidden State)
# =====================================================================
class GRUJITWrapper(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs: torch.Tensor, hidden_states: torch.Tensor):
        """
        显式拆解 Actor 的前向传播流程，完美契合 TorchScript 的纯函数 Trace 追踪规范。
        """
        # 1. 调用你写的融合逻辑：走 CNN 降维，并与本体感知拼接
        fused_obs = self.actor_critic.process_obs(obs)
        
        # 2. 走 GRU (时序记忆层)
        # RSL-RL 的 RNN 默认接收三维张量 [seq_len, batch_size, num_obs]
        # 在推理阶段，seq_len 永远是 1
        seq_in = fused_obs.unsqueeze(0)
        
        # 显式传入隐状态，并接收更新后的隐状态
        gru_out_seq, next_hidden_states = self.actor_critic.memory_a.rnn(seq_in, hidden_states)
        
        # 将 seq_len 维度重新挤压掉 [batch_size, rnn_hidden_dim]
        gru_out = gru_out_seq.squeeze(0)
        
        # 3. 走 主干 MLP (策略动作生成)
        actions = self.actor_critic.actor(gru_out)
        
        return actions, next_hidden_states


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """加载模型并导出 JIT"""
    # 更新配置
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # 指定日志目录以查找检查点
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    # 获取要加载的模型路径
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] 正在从以下路径加载模型权重: {resume_path}")

    # 创建 isaac 环境 (仅用于初始化网络维度)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 创建 Runner 并加载权重
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device=agent_cfg.device)
    runner.load(resume_path)

    # 提取 actor_critic 模块
    actor_critic = runner.alg.policy
    # 切换至评估模式，并将其转移到 CPU
    actor_critic.eval()
    actor_critic.to("cpu")

    # 包装模型
    jit_wrapper = GRUJITWrapper(actor_critic)

    print("[INFO] 开始进行 JIT (TorchScript Trace) 导出...")
    
    # 构造伪造张量进行 Trace 追踪导出
    # 提示：部署推理时，输入永远是 2D 的 [1, num_obs]
    dummy_obs = torch.zeros(1, env.num_obs, device="cpu")
    
    # 获取 GRU 的层数和维度
    rnn_num_layers = agent_cfg.policy.rnn_num_layers
    rnn_hidden_dim = agent_cfg.policy.rnn_hidden_dim
    
    # GRU 隐状态: [num_layers, batch_size, hidden_dim]
    dummy_hidden = torch.zeros(rnn_num_layers, 1, rnn_hidden_dim, device="cpu")

    try:
        # 执行 Trace (strict=False 可以容忍一些条件分支的警告)
        jit_model = torch.jit.trace(jit_wrapper, (dummy_obs, dummy_hidden), strict=False)
        jit_model.save(args_cli.export_path)
        print(f"[SUCCESS] 模型成功通过 Trace 导出并保存至: {args_cli.export_path}")
    except Exception as e:
        print(f"[ERROR] 导出彻底失败，错误原因:\n{e}")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
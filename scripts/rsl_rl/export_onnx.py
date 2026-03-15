# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to export custom CNN-GRU recurrent policy to an ONNX model."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# 添加 argparse 参数 (修改了默认的导出后缀为 .onnx)
parser = argparse.ArgumentParser(description="Export an RL agent with custom CNN-GRU to ONNX.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--export_path", type=str, default="policy_gru_cnn.onnx", help="Path to save the ONNX model.")
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
# ★ 核心改动 2：适配 ONNX 导出的 Wrapper (逻辑与 JIT 完全一致)
# =====================================================================
class GRUONNXWrapper(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor_critic = actor_critic

    def forward(self, obs: torch.Tensor, hidden_states: torch.Tensor):
        """
        显式拆解 Actor 的前向传播流程，完美契合 ONNX 的纯函数追踪规范。
        """
        # 1. 调用你写的融合逻辑：走 CNN 降维，并与本体感知拼接
        fused_obs = self.actor_critic.process_obs(obs)
        
        # 2. 走 GRU (时序记忆层)
        # 提示：ONNX 导出时会记录这些 squeeze/unsqueeze 操作
        seq_in = fused_obs.unsqueeze(0)
        
        # 显式传入隐状态，并接收更新后的隐状态
        gru_out_seq, next_hidden_states = self.actor_critic.memory_a.rnn(seq_in, hidden_states)
        
        # 将 seq_len 维度重新挤压掉
        gru_out = gru_out_seq.squeeze(0)
        
        # 3. 走 主干 MLP (策略动作生成)
        actions = self.actor_critic.actor(gru_out)
        
        return actions, next_hidden_states


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """加载模型并导出 ONNX"""
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
    actor_critic.eval()
    actor_critic.to("cpu")

    # 包装模型
    onnx_wrapper = GRUONNXWrapper(actor_critic)

    print("[INFO] 开始进行 ONNX 导出...")
    
    # 构造伪造张量进行追踪导出
    # 假设导出时 batch_size 为 1，但我们会配置动态轴支持推理时任意 batch
    dummy_obs = torch.zeros(1, env.num_obs, device="cpu")
    
    # 获取 GRU 的层数和维度
    rnn_num_layers = agent_cfg.policy.rnn_num_layers
    rnn_hidden_dim = agent_cfg.policy.rnn_hidden_dim
    
    # GRU 隐状态: [num_layers, batch_size, hidden_dim]
    dummy_hidden = torch.zeros(rnn_num_layers, 1, rnn_hidden_dim, device="cpu")

    try:
        # =====================================================================
        # ★ 核心改动 3：使用 torch.onnx.export 替代 torch.jit.trace
        # =====================================================================
        torch.onnx.export(
            onnx_wrapper,                                   # 包装好的模型
            (dummy_obs, dummy_hidden),                      # 模型输入元组
            args_cli.export_path,                           # 导出路径
            export_params=True,                             # 导出模型权重
            opset_version=14,                               # ONNX 算子集版本 (14 或 17 较稳定)
            do_constant_folding=True,                       # 开启常量折叠优化
            input_names=["obs", "hidden_in"],               # 绑定输入节点名称
            output_names=["actions", "hidden_out"],         # 绑定输出节点名称
            dynamic_axes={                                  # 配置动态 Batch 维度
                "obs": {0: "batch_size"},                   # obs 形状: [batch_size, num_obs]
                "hidden_in": {1: "batch_size"},             # hidden_in 形状: [num_layers, batch_size, hidden_dim]
                "actions": {0: "batch_size"},               # actions 形状: [batch_size, num_actions]
                "hidden_out": {1: "batch_size"}             # hidden_out 形状: [num_layers, batch_size, hidden_dim]
            }
        )
        print(f"[SUCCESS] 模型成功导出并保存至: {args_cli.export_path}")
    except Exception as e:
        print(f"[ERROR] 导出彻底失败，错误原因:\n{e}")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
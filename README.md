

# Ranger Demo: Skid-Steer Legged Robot RL Environment

基于 **Isaac Lab (Omniverse)** 构建的强化学习环境，专为轮腿式（滑移转向 + 支腿调节）机器人设计。本项目重点探索在波浪地形下的鲁棒运动控制，特别是针对传感器遮挡（盲区）和打滑情况的适应性训练。

## 🚀 项目亮点 (Key Features)

* **混合运动学控制 (Hybrid Locomotion)**:
* 自定义 `SkidSteerLegAction` 动作空间，结合了底盘的差速驱动（Skid-Steer）与支腿的关节速度控制。
* 实现了基于几何参数的运动学解算，支持“禁止倒车”约束和速度平滑处理。


* **非对称 Actor-Critic 架构 (Asymmetric Actor-Critic)**:
* **Policy (Actor)**: 模拟受限的真实感知。输入包含带噪声的本体感知、显式的打滑特征 (`slip_features`) 以及**带遮挡的高度图 (Masked Height Scan)**（模拟传感器盲区）。
* **Critic**: 拥有上帝视角。输入无噪声的真实状态 (Ground Truth) 和完整的地形高度图。


* **鲁棒性设计**:
* **视觉盲区适应**: Actor 必须学会在前方地形不可见（被遮挡）的情况下决策。
* **打滑感知**: 引入“理论轮速 vs 真实基座速度”的偏差作为观测特征，帮助 Agent 感知并应对打滑。


* **定制化奖励函数**:
* 包含带死区的姿态稳定性奖励 (`flat_orientation_with_tolerance`)，允许车身在一定范围内随地形起伏。
* 打滑一致性惩罚 (`slip_consistency`) 和足端离地时间惩罚，促进平稳的贴地飞行。



## 📂 项目结构 (Project Structure)

```text
source/robot1/robot1/tasks/manager_based/MY_EVN/
├── agents/                 # PPO 算法配置文件 (rsl_rl, rl_games, sb3, skrl)
├── mdp/
│   ├── actions/            # 自定义动作空间 (skid_steer_leg_actions.py)
│   ├── observations.py     # 观测空间定义 (含 Masked Height Scan 实现)
│   ├── rewards.py          # 自定义奖励函数 (打滑、姿态、调距惩罚)
│   ├── terrain.py          # 波浪地形生成器配置
│   └── ...
├── robot1_env_cfg.py       # 环境总配置 (场景、传感器、各组件组装)
└── robot1.py               # 机器人资产定义

```

## 🛠️ 安装与配置 (Installation)

1. **前置要求**: 请确保已安装 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)。
2. **安装本项目**:
```bash
# 在本仓库根目录下运行
python -m pip install -e source/robot1

```


3. **验证安装**:
```bash
python scripts/list_envs.py
# 确认能看到你的环境名称

```



## 🏃‍♂️ 运行与训练 (Usage)

### 1. 训练 (Training)

使用 `rsl_rl` (或其他支持的库) 开始训练：

```bash
# task_name 请根据你在 list_envs.py 中注册的名字替换 (例如: "Isaac-Robot1-Rough-v0")
python scripts/rsl_rl/train.py --task <TASK_NAME> --headless

```

### 2. 游玩/推理 (Play)

加载训练好的模型进行可视化：

```bash
python scripts/rsl_rl/play.py --task <TASK_NAME> --num_envs 1

```

### 3. 调试环境 (Debugging)

使用随机 Agent 或零动作 Agent 快速测试环境配置是否正确：

```bash
# 测试随机动作
python scripts/random_agent.py --task <TASK_NAME>

# 测试零动作 (检查机器人是否静态平衡)
python scripts/zero_agent.py --task <TASK_NAME>

```

## 🧠 详细设计说明 (Design Details)

### 观测空间 (Observation Space)

| Group | 包含特征 | 说明 |
| --- | --- | --- |
| **Policy** | `base_lin/ang_vel` (Noisy)<br>

<br>`slip_feat`<br>

<br>`height_scan` (Masked) | 模拟真实传感器，包含高斯噪声。**Height Scan 前方区域被人为抹零**，迫使策略记忆地形或保守行动。 |
| **Critic** | `base_lin/ang_vel` (True)<br>

<br>`full_height_scan` | 训练阶段的特权信息，帮助 Critic 准确预估价值，指导 Actor 学习。 |

### 动作空间 (Action Space)

* **输入**: `[v_target, w_target, leg_1_vel, ..., leg_4_vel]`
* 前两维控制底盘的线速度和角速度。
* 后四维控制四个支腿关节的**速度** (Velocity Control)。


* **处理逻辑**: 系统会自动根据轮距 () 和轮半径 () 将  解算为四个轮子的转速。

---

**Note**: 本项目是基于 Isaac Lab 模板开发的扩展库。详细文档请参考 [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/).


python scripts/rsl_rl/sim2sim.py
python scripts/rsl_rl/export_gru_cnn_jit.py --task sk-Robot1-v0 --export_path /home/wjc/robot1/jit_models/policy_3.pt --load_run 2026-02-27_15-31-18 --checkpoint model_2999.pt --export_path /home/wjc/robot1/jit_models/policy_lstm_2.pt
watch -n 1 nvidia-smi
python scripts/rsl_rl/train.py     --task sk-Robot1-v0     --logger wandb     --log_project_name robot1-rough  --video --headless

python scripts/rsl_rl/play_keyboard.py     --task sk-Robot1-v0     --load_run 2026-01-29_17-24-08 --checkpoint model_3499.pt
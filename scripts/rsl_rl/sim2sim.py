import mujoco
import mujoco.viewer
import numpy as np
import torch
import time

# ==========================================
# 1. 配置参数 (TODO: 请核对这些参数)
# ==========================================
# 模型路径
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_gru_cnn.pt"  # 你的 JIT 模型路径
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger.xml"             # 你的 MuJoCo 机器人描述文件

# 控制频率配置
SIM_DT = 0.002           # MuJoCo 物理步长 (2ms = 500Hz)
POLICY_DT = 0.02         # 神经网络推理步长 (20ms = 50Hz)
DECIMATION = int(POLICY_DT / SIM_DT) # 控制降采样率 (通常是 10)

# 动作缩放因子 (Action Scale) - 必须和 Isaac Lab 中一模一样
WHEEL_VEL_SCALE = 10.0   # 轮子速度缩放
LEG_POS_SCALE = 0.5      # 腿部关节位置缩放

# 默认关节位置 (Default Joint Positions)
# 顺序必须严格对应你输入给网络的顺序
DEFAULT_LEG_POS = np.array([0.0, 0.0, 0.0, 0.0]) # 假设4个EHA悬挂的初始位置

# 期望速度指令 [vx, vy, yaw_rate]
COMMAND = np.array([0.5, 0.0, 0.0]) 

# ==========================================
# 2. 观测构建函数 (核心难点)
# ==========================================
def get_observation(data, model, prev_action, command):
    """
    从 MuJoCo 中提取传感器数据，并严格按照 Isaac Lab 的顺序拼接
    """
    # 1. 获取基础物理量
    # 注意：MuJoCo 的四元数是 [w, x, y, z]，而 Isaac Lab 通常是 [w, x, y, z] 或 [x, y, z, w]，请核对！
    quat = data.qpos[3:7] 
    
    # 计算重力在机身坐标系的投影 (Projected Gravity)
    # 利用四元数将世界坐标系的重力 [0, 0, -1] 转换到机身系
    # 这里用一个简化的 numpy 转换，实际中可以使用 scipy.spatial.transform.Rotation
    # 简单模拟: 假设机身水平，投影重力为 [0, 0, -1]
    proj_grav = np.array([0.0, 0.0, -1.0]) 
    
    # 获取机身线速度和角速度 (局部坐标系)
    base_lin_vel = data.qvel[0:3]
    base_ang_vel = data.qvel[3:6]
    
    # 2. 获取关节状态 (必须与动作输出顺序对应，比如先4个轮子，再4个腿)
    # TODO: 这里需要根据你的 XML 关节索引填写
    # 假设 data.qpos[7:11] 是轮子位置 (测速用 data.qvel), data.qpos[11:15] 是腿部位置
    wheel_vel = data.qvel[6:10] # 假设的索引
    leg_pos = data.qpos[11:15] - DEFAULT_LEG_POS # 往往输入网络的是残差位置
    leg_vel = data.qvel[10:14]
    
    # 3. 构建高程图 (Height Map) 20x20 = 400D
    # 初步测试时，假设地面绝对平坦，高程为 0
    height_map = np.zeros(400, dtype=np.float32)
    
    # 4. 拼接观测向量 (★ 顺序必须与你的 observations.py 100% 一致 ★)
    obs_list = [
        base_lin_vel,
        base_ang_vel,
        proj_grav,
        command,
        wheel_vel,
        leg_pos,
        leg_vel,
        prev_action,
        # ... 这里可能还有你的打滑特征 slip_feat 等 ...
        height_map
    ]
    
    # 展平成一维数组
    obs_flat = np.concatenate(obs_list).astype(np.float32)
    return obs_flat

# ==========================================
# 3. 主控制循环
# ==========================================
def main():
    # 加载 MuJoCo 模型
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # 加载 PyTorch 策略 (确保在 CPU 上运行，加快单步推理)
    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()

    # 初始化 GRU 隐状态 (1层, batch=1, 256维)
    hidden_state = torch.zeros(1, 1, 256, dtype=torch.float32)
    
    # 初始化历史变量
    action = np.zeros(8, dtype=np.float32) # 假设 4轮 + 4腿 = 8自由度
    
    # 启动可视化器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        step_counter = 0
        
        # 允许用户在 Viewer 里按空格暂停，这里只在运行时更新控制
        while viewer.is_running():
            step_start = time.time()
            
            # --- 神经网络推理层 (50Hz) ---
            if step_counter % DECIMATION == 0:
                # 1. 获取观测
                obs_np = get_observation(data, model, action, COMMAND)
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0) # 变维为 [1, num_obs]
                
                # 2. 运行模型推理 (无梯度模式极速运行)
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                # 3. 将动作从 Tensor 转回 Numpy，并作为 prev_action 存下来
                action = action_tensor.squeeze(0).numpy()
                
                # 4. 解析并缩放动作 (Action -> Target)
                # 假设网络输出前4个是轮子速度，后4个是腿部位置
                target_wheel_vel = action[0:4] * WHEEL_VEL_SCALE
                target_leg_pos = DEFAULT_LEG_POS + action[4:8] * LEG_POS_SCALE
            
            # --- 物理引擎底层控制 (500Hz) ---
            # TODO: 将目标指令发给 MuJoCo 的执行器
            # 假设 model.ctrl[0:4] 配置为了轮毂电机的 velocity 模式
            # 假设 model.ctrl[4:8] 配置为了腿部 EHA 的 position 模式
            data.ctrl[0:4] = target_wheel_vel
            data.ctrl[4:8] = target_leg_pos
            
            # 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 时间对齐，保持真实的仿真速度
            step_counter += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
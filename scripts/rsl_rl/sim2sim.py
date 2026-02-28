import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 配置参数 (已根据你的配置文件修正)
# ==========================================
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_lstm_2.pt"  
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger.xml"             

SIM_DT = 0.002           
POLICY_DT = 0.02         
DECIMATION = int(POLICY_DT / SIM_DT) 

# ★ 修正点1：通过 robot1.py 和 urdf 计算出的真实参数 ★
# URDF中关节limit是0.35, robot1.py中 soft_joint_pos_limit_factor=0.7
# 所以反归一化的最大范围是 0.35 * 0.7 = 0.245
LEG_POS_SCALE = 0.245      

# 差速运动学参数 (来自 actions_cfg.py)
BASE_WIDTH = 0.5       # 轮距 W
WHEEL_RADIUS = 0.05    # 轮半径 r
ACTION_ALPHA = 0.3     # 动作低通滤波系数

# 默认关节位置 (根据 robot1.py 中的 init_state)
# 顺序: LB, LF, RF, RB (与你在 robot1.py actuators 中定义的顺序一致)
DEFAULT_LEG_POS = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)

COMMAND = np.array([0.5, 0.0, 0.0]) # [vx, vy, yaw_rate]

# ==========================================
# 2. 观测构建函数
# ==========================================
def get_observation(data, prev_action, command):
    """
    严格按照 471 维的 SkidSteerLegObsCfg 顺序，以及 Isaac Lab 的【字母表排序】拼接观测
    """
    # 1. base_ang_vel (3,)
    base_ang_vel = data.qvel[3:6].astype(np.float32)
    
    # 2. projected_grav (3,)
    quat_wxyz = data.qpos[3:7]
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
    proj_grav = r.inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

    # 3. cmd_vw (2,) -> [v_x, omega_z]
    cmd_vw = command[:2].astype(np.float32)

    # =====================================================================
    # ★ 极其关键：Isaac Lab 的 "g_.*" 和 "w_.*" 是按字母表排序的！
    # 字母表顺序为: 1. lb, 2. lf, 3. rb, 4. rf
    # 对应 MuJoCo 的索引 (根据你 XML 中 joint 的定义顺序):
    # g_lb: qpos[7], qvel[6]
    # g_lf: qpos[9], qvel[8]
    # g_rb: qpos[13], qvel[12]
    # g_rf: qpos[11], qvel[10]
    #
    # w_lb: qvel[7]
    # w_lf: qvel[9]
    # w_rb: qvel[13]
    # w_rf: qvel[11]
    # =====================================================================
    
    # 4. wheel_vel (4,) [LB, LF, RB, RF]
    wheel_vel = np.array([data.qvel[7], data.qvel[9], data.qvel[13], data.qvel[11]], dtype=np.float32)
    
    # 5. leg_pos (4,) [LB, LF, RB, RF]
    leg_pos = np.array([data.qpos[7], data.qpos[9], data.qpos[13], data.qpos[11]], dtype=np.float32)
    
    # 6. leg_vel (4,) [LB, LF, RB, RF]
    leg_vel = np.array([data.qvel[6], data.qvel[8], data.qvel[12], data.qvel[10]], dtype=np.float32)
    
    # 7. leg_pos_norm (4,) -> [-1, 1]
    leg_pos_norm = np.clip(leg_pos / 0.245, -1.0, 1.0).astype(np.float32)

    # 8. last_action (6,) 
    # 9. height_scan (441,)
    height_scan = np.zeros(441, dtype=np.float32)

    obs_list = [
        base_ang_vel,   # 3
        proj_grav,      # 3
        cmd_vw,         # 2
        wheel_vel,      # 4
        leg_pos,        # 4
        leg_vel,        # 4
        leg_pos_norm,   # 4
        prev_action,    # 6
        height_scan     # 441
    ]
    return np.concatenate(obs_list).astype(np.float32)


def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()
    hidden_state = torch.zeros(1, 1, 128, dtype=torch.float32) 
    
    raw_action = np.zeros(6, dtype=np.float32)      
    filtered_action = np.zeros(6, dtype=np.float32) 
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 神经网络推理层 (50Hz) ---
            if step_counter % DECIMATION == 0:
                obs_np = get_observation(data, raw_action, COMMAND) # 传给网络的是网络上一步输出的裸动作
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0) 
                
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                raw_action = action_tensor.squeeze(0).numpy()
                
                # 动作低通滤波
                filtered_action = ACTION_ALPHA * raw_action + (1.0 - ACTION_ALPHA) * filtered_action
                
# =======================================================
                # 动作解包: 网络输出顺序为 [v_x, omega_z, LF, RF, LB, RB]
                # 依据: actions_cfg.py 中的 leg_joint_names=["g_lf", "g_rf", "g_lb", "g_rb"]
                # =======================================================
                cmd_v_x = filtered_action[0]      
                cmd_omega_z = filtered_action[1]  
                
                # 1. 差速逆运动学
                v_left = cmd_v_x - cmd_omega_z * (BASE_WIDTH / 2.0)
                v_right = cmd_v_x + cmd_omega_z * (BASE_WIDTH / 2.0)
                
                omega_left = v_left / WHEEL_RADIUS
                omega_right = v_right / WHEEL_RADIUS
                
                # 2. 腿部目标角度 (注意这里的索引映射：2=LF, 3=RF, 4=LB, 5=RB)
                target_LF = 0.05 + filtered_action[2] * 0.245
                target_RF = 0.05 + filtered_action[3] * 0.245
                target_LB = 0.05 + filtered_action[4] * 0.245
                target_RB = 0.05 + filtered_action[5] * 0.245

            # --- 物理引擎底层控制 (500Hz) ---
            # 1. 下发腿部位置 (XML 执行器顺序: act_g_lb, act_g_lf, act_g_rf, act_g_rb)
            data.ctrl[0] = target_LB
            data.ctrl[1] = target_LF
            data.ctrl[2] = target_RF
            data.ctrl[3] = target_RB  
            
            # 2. 下发轮毂速度 (XML 执行器顺序: act_w_lb, act_w_lf, act_w_rf, act_w_rb)
            # ★ 修复 runaway(无限加速) BUG：绝不能加任何负号，原汁原味地喂给执行器！
            data.ctrl[4] = omega_left   # 左后
            data.ctrl[5] = omega_left   # 左前
            data.ctrl[6] = omega_right  # 右前
            data.ctrl[7] = omega_right  # 右后
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_counter += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
    '''

def get_observation(data, prev_action, command):
    """
    严格按照 471 维的 SkidSteerLegObsCfg 顺序拼接观测
    """
    # 1. base_ang_vel (3,)
    base_ang_vel = data.qvel[3:6].astype(np.float32)
    
    # 2. projected_grav (3,)
    quat_wxyz = data.qpos[3:7]
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
    proj_grav = r.inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

    # 3. cmd_vw (2,) -> [v_x, omega_z]
    cmd_vw = command[:2].astype(np.float32)

    # --- 提取关节状态并重组顺序为: LB, LF, RF, RB ---
    # 根据 URDF qpos/qvel 的索引：6:g_lb, 7:w_lb, 8:g_lf, 9:w_lf, 10:g_rf, 11:w_rf, 12:g_rb, 13:w_rb
    
    # 4. wheel_vel (4,)
    wheel_vel = np.array([data.qvel[7], data.qvel[9], data.qvel[11], data.qvel[13]], dtype=np.float32)
    
    # 5. leg_pos (4,) - 直接取绝对位置
    leg_pos = np.array([data.qpos[7], data.qpos[9], data.qpos[11], data.qpos[13]], dtype=np.float32)
    
    # 6. leg_vel (4,)
    leg_vel = np.array([data.qvel[6], data.qvel[8], data.qvel[10], data.qvel[12]], dtype=np.float32)
    
    # 7. leg_pos_norm (4,) - 软限位归一化
    # URDF 限位为 [-0.35, 0.35]，Isaac Lab 软限位系数为 0.7
    # 因此合法范围是 [-0.245, 0.245]，中点 mid=0，半区间 half=0.245
    leg_pos_norm = np.clip(leg_pos / 0.245, -1.0, 1.0).astype(np.float32)

    # 8. last_action (6,) 
    # prev_action 作为参数传入，保持 6 维
    
    # 9. height_scan (441,)
    height_scan = np.zeros(441, dtype=np.float32)

    # ★ 严格按照你定义的顺序拼接 ★
    obs_list = [
        base_ang_vel,   # (3,)
        proj_grav,      # (3,)
        cmd_vw,         # (2,)
        wheel_vel,      # (4,)
        leg_pos,        # (4,)
        leg_vel,        # (4,)
        leg_pos_norm,   # (4,)
        prev_action,    # (6,)
        height_scan     # (441,)
    ]
    
    # 返回 471 维的一维张量
    return np.concatenate(obs_list).astype(np.float32)

# ==========================================
# 3. 主控制循环
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    policy = torch.jit.load(POLICY_PATH, map_location="cpu")
    policy.eval()
    hidden_state = torch.zeros(1, 1, 128, dtype=torch.float32) # 注意：你代码中一般用 128 维
    
    # 变量初始化
    raw_action = np.zeros(6, dtype=np.float32)      # 神经网络原始输出 (6维)
    filtered_action = np.zeros(6, dtype=np.float32) # 滤波后的动作 (6维)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_counter = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- 神经网络推理层 (50Hz) ---
            if step_counter % DECIMATION == 0:
                obs_np = get_observation(data, filtered_action, COMMAND)
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0) 
                
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                raw_action = action_tensor.squeeze(0).numpy()
                
                # ★ 修正点3：复刻 actions_cfg.py 中的低通滤波器
                filtered_action = ACTION_ALPHA * raw_action + (1.0 - ACTION_ALPHA) * filtered_action
                
                # ★ 修正点4：差速运动学解析 (6维网络输出 -> 8维硬件执行器)
                # SkidSteerLegAction 默认输出顺序：[v_x, omega_z, leg1, leg2, leg3, leg4]
                cmd_v_x = filtered_action[0]      # 缩放系数 base_scale 默认是 1.0
                cmd_omega_z = filtered_action[1]  
                
                # 逆运动学计算左右侧理论线速度
                v_left = cmd_v_x - cmd_omega_z * (BASE_WIDTH / 2.0)
                v_right = cmd_v_x + cmd_omega_z * (BASE_WIDTH / 2.0)
                
                # 转为轮子角速度 (rad/s)
                omega_left = v_left / WHEEL_RADIUS
                omega_right = v_right / WHEEL_RADIUS
                
                # 腿部目标位置
                target_leg_pos = DEFAULT_LEG_POS + filtered_action[2:6] * LEG_POS_SCALE

            # --- 物理引擎底层控制 (500Hz) ---
            # 这里的赋值顺序必须和你在 XML 文件中 <actuator> 标签里的定义顺序一模一样！
            # 假设 XML actuator 顺序为: g_lb, g_lf, g_rf, g_rb, w_lb, w_lf, w_rf, w_rb
            
            # 4个腿的位置控制
            data.ctrl[0:4] = target_leg_pos  
            
            # 4个轮子的速度控制 (对应 LF, LB 赋左侧速度；RF, RB 赋右侧速度)
            # 根据你之前的重组顺序: LB, LF, RF, RB
            data.ctrl[4] = omega_left   # w_lb
            data.ctrl[5] = omega_left   # w_lf
            data.ctrl[6] = omega_right  # w_rf
            data.ctrl[7] = omega_right  # w_rb
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_counter += 1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
    '''

'''
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
'''
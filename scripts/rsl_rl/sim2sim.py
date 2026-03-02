import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 基础配置参数
# ==========================================
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_4.pt" # 请确保路径和文件名正确
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger2.xml"             

SIM_DT = 0.002           
POLICY_DT = 0.02         
DECIMATION = int(POLICY_DT / SIM_DT) 

# 运动学与动作缩放参数
LEG_POS_SCALE = 0.3  # URDF 极限 0.35 * 软限位 0.7 = 0.245
BASE_WIDTH = 0.5       
WHEEL_RADIUS = 0.05    
ACTION_ALPHA = 0.3     

# 默认状态
DEFAULT_LEG_POS = 0.05 

BASE_LIN_VEL_SCALE = 1.0   # 对应 base_scale[0]
BASE_ANG_VEL_SCALE = 0.5  # 对应 base_scale[1]

# ==========================================
# 2. 观测构建函数 (471维)
# ==========================================
def get_observation(data, prev_action, command):
    """
    严格按照 SkidSteerLegObsCfg 顺序拼接
    关节顺序严格匹配 URDF: LB, LF, RF, RB
    """
    # 1. base_ang_vel (3,)
    base_ang_vel = data.qvel[3:6].astype(np.float32)
    
    # 2. projected_grav (3,)
    quat_wxyz = data.qpos[3:7]
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]) 
    proj_grav = r.inv().apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)

    # 3. cmd_vw (2,)
    cmd_vw = np.array([command[0], command[2]], dtype=np.float32)

    # 4. wheel_vel (4,) [LB, LF, RF, RB]
    wheel_vel = np.array([data.qvel[7], data.qvel[9], data.qvel[11], data.qvel[13]], dtype=np.float32)
    
    # 5. leg_pos (4,) [LB, LF, RF, RB]
    leg_pos = np.array([data.qpos[7], data.qpos[9], data.qpos[11], data.qpos[13]], dtype=np.float32)
    
    # 6. leg_vel (4,) [LB, LF, RF, RB]
    leg_vel = np.array([data.qvel[6], data.qvel[8], data.qvel[10], data.qvel[12]], dtype=np.float32)
    
    # 7. leg_pos_norm (4,) -> [-1, 1]
    #leg_pos_norm = np.clip(leg_pos / LEG_POS_SCALE, -1.0, 1.0).astype(np.float32)

    # 8. last_action (6,) 已作为参数传入
    
    # 9. height_scan (441,)
    # 修复“活埋BUG”：平地的高度差 = 0.0 - 车身绝对Z坐标
    base_z = data.qpos[2]
    height_scan = np.full(441, 0.0 - base_z, dtype=np.float32)

    obs_list = [
        base_ang_vel,   # 3
        proj_grav,      # 3
        cmd_vw,         # 2
        wheel_vel,      # 4
        leg_pos,        # 4
        leg_vel,        # 4
        #leg_pos_norm,   # 4
        prev_action,    # 6
        height_scan     # 441
    ]
    return np.concatenate(obs_list).astype(np.float32)

# ==========================================
# 3. 主控制循环
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # 加载带有 RNN(LSTM/GRU) 的策略网络
    device = torch.device("cpu")
    policy = torch.jit.load(POLICY_PATH, map_location=device)
    policy.eval()
    
    # 请根据你导出的模型确认 hidden_state 的形状
    # GRU 通常是 (1, 128) 或 (1, 1, 128)
    hidden_state = torch.zeros((1, 1, 128), dtype=torch.float32, device=device) 
    
    raw_action = np.zeros(6, dtype=np.float32)      
    filtered_action = np.zeros(6, dtype=np.float32) 
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # ----------------------------------------------------
        # ★ 物理引擎冷启动配置：消除开机抽搐
        # ----------------------------------------------------
        mujoco.mj_resetData(model, data)
        
        # 将四条腿的初始位置手动掰到 0.05，与控制器期望保持一致
        data.qpos[7] = DEFAULT_LEG_POS   # LB
        data.qpos[9] = DEFAULT_LEG_POS   # LF
        data.qpos[11] = DEFAULT_LEG_POS  # RF
        data.qpos[13] = DEFAULT_LEG_POS  # RB
        
        mujoco.mj_forward(model, data)
        
        step_counter = 0
        while viewer.is_running():
            step_start = time.time()
            
            # =======================================================
            # ★ 核心修复：监听 Reset 事件，对大脑和物理姿态进行“彻底洗脑”
            # =======================================================
            if data.time == 0.0:
                # 1. 清空 GRU 记忆和动作滤波器
                hidden_state = torch.zeros((1, 1, 128), dtype=torch.float32, device=device)
                raw_action = np.zeros(6, dtype=np.float32)
                filtered_action = np.zeros(6, dtype=np.float32)
                
                # 2. 重新把腿掰回默认角度 (应对 Viewer 的暴力归零)
                data.qpos[7] = DEFAULT_LEG_POS   # LB
                data.qpos[9] = DEFAULT_LEG_POS   # LF
                data.qpos[11] = DEFAULT_LEG_POS  # RF
                data.qpos[13] = DEFAULT_LEG_POS  # RB
                mujoco.mj_forward(model, data)
            
            # =======================================================
            # ★ 使用仿真时间 data.time 进行 1.5 秒物理锁死预热
            # =======================================================
            if data.time < 1.5:
                # 预热期：底盘死死锁住，不调用神经网络
                data.ctrl[0] = DEFAULT_LEG_POS
                data.ctrl[1] = DEFAULT_LEG_POS
                data.ctrl[2] = DEFAULT_LEG_POS
                data.ctrl[3] = DEFAULT_LEG_POS
                
                data.ctrl[4:8] = 0.0  # 轮子全部刹车
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
                # 保持实时率
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                continue # 直接跳过下方的神经网络推理，进入下一帧
            
            # =======================================================
            # --- 1.5 秒后，正常接管控制逻辑 ---
            # =======================================================
            current_cmd = np.array([0.2, 0.0, 0.2], dtype=np.float32)
            
            # --- 神经网络推理层 (50Hz) ---
            if step_counter % DECIMATION == 0:
                obs_np = get_observation(data, filtered_action, current_cmd) 
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                raw_action = action_tensor.squeeze().cpu().numpy()
                filtered_action = ACTION_ALPHA * raw_action + (1.0 - ACTION_ALPHA) * filtered_action
                
                # 动作解包
                cmd_v_x = filtered_action[0] * BASE_LIN_VEL_SCALE      
                cmd_omega_z = filtered_action[1] * BASE_ANG_VEL_SCALE
                
                v_left = cmd_v_x - cmd_omega_z * (BASE_WIDTH / 2.0)
                v_right = cmd_v_x + cmd_omega_z * (BASE_WIDTH / 2.0)
                
                omega_left = v_left / WHEEL_RADIUS
                omega_right = v_right / WHEEL_RADIUS
                
                target_LF = DEFAULT_LEG_POS + filtered_action[2] * LEG_POS_SCALE
                target_RF = DEFAULT_LEG_POS + filtered_action[3] * LEG_POS_SCALE
                target_LB = DEFAULT_LEG_POS + filtered_action[4] * LEG_POS_SCALE
                target_RB = DEFAULT_LEG_POS + filtered_action[5] * LEG_POS_SCALE

            # --- 物理引擎底层控制 (500Hz) ---
            data.ctrl[0] = target_LB
            data.ctrl[1] = target_LF
            data.ctrl[2] = target_RF
            data.ctrl[3] = target_RB  
            
            data.ctrl[4] = omega_left   
            data.ctrl[5] = omega_left   
            data.ctrl[6] = omega_right  
            data.ctrl[7] = omega_right  
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            step_counter += 1
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()



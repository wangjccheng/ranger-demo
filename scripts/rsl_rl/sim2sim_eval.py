import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch
import time
import os
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 基础配置参数
# ==========================================
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_4.pt" 
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger2.xml"             

SIM_DT = 0.002           
POLICY_DT = 0.02         
DECIMATION = int(POLICY_DT / SIM_DT) 

LEG_POS_SCALE = 0.3  
BASE_WIDTH = 0.5       
WHEEL_RADIUS = 0.05    
ACTION_ALPHA = 0.3     

DEFAULT_LEG_POS = 0.05 
CMD_VX = 0.5
NUM_EVAL_STEPS = 500  # 评估步数 (500 * 0.02s = 10秒)

# ==========================================
# 2. 观测与物理状态解析
# ==========================================
def get_robot_state(data):
    """提取机器人在 MuJoCo 中的真实物理状态 (用于记录和 PID 控制)"""
    # 四元数 [w, x, y, z] 转 scipy 需要的 [x, y, z, w]
    quat_wxyz = data.qpos[3:7]
    r = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    
    # 欧拉角 (roll, pitch, yaw)
    euler = r.as_euler('xyz', degrees=False)
    roll, pitch, yaw = euler[0], euler[1], euler[2]
    
    # 机身局部线速度 (World -> Body)
    v_world = data.qvel[0:3]
    v_body = r.inv().apply(v_world)
    actual_vx = v_body[0]
    
    # 机身局部角速度 (World -> Body)
    w_world = data.qvel[3:6]
    w_body = r.inv().apply(w_world)
    pitch_rate = w_body[1]
    
    # Z轴加速度 (直接取世界坐标系的 Z 轴线加速度)
    z_accel = data.qacc[2]
    
    return roll, pitch, actual_vx, pitch_rate, z_accel, r

def get_observation(data, prev_action, cmd_vw, r_inv):
    """构建 RL 策略的 471 维观测输入 (已修复字母表排序)"""
    base_ang_vel = r_inv.apply(data.qvel[3:6]).astype(np.float32)
    proj_grav = r_inv.apply(np.array([0.0, 0.0, -1.0])).astype(np.float32)
    
    # 顺序: LB, LF, RB, RF
    wheel_vel = np.array([data.qvel[7], data.qvel[9], data.qvel[13], data.qvel[11]], dtype=np.float32)
    leg_pos = np.array([data.qpos[7], data.qpos[9], data.qpos[13], data.qpos[11]], dtype=np.float32)
    leg_vel = np.array([data.qvel[6], data.qvel[8], data.qvel[12], data.qvel[10]], dtype=np.float32)
    leg_pos_norm = np.clip(leg_pos / LEG_POS_SCALE, -1.0, 1.0).astype(np.float32)

    current_base_z = data.qpos[2]
    height_scan = np.full(441, -current_base_z, dtype=np.float32)

    obs_list = [
        base_ang_vel, proj_grav, cmd_vw.astype(np.float32), 
        wheel_vel, leg_pos, leg_vel, leg_pos_norm, 
        prev_action, height_scan
    ]
    return np.concatenate(obs_list).astype(np.float32)

# ==========================================
# 3. 核心评估流程
# ==========================================
def run_evaluation_mode(mode_name, model, data, viewer, policy, device):
    print(f"🚀 开始测试模式: {mode_name} ...")
    
    # 重置环境与状态
    mujoco.mj_resetData(model, data)
    data.qpos[7] = data.qpos[9] = data.qpos[11] = data.qpos[13] = DEFAULT_LEG_POS
    mujoco.mj_forward(model, data)
    
    hidden_state = torch.zeros((1, 1, 128), dtype=torch.float32, device=device)
    raw_action = np.zeros(6, dtype=np.float32)      
    filtered_action = np.zeros(6, dtype=np.float32) 
    
    integral_error_v = 0.0
    current_cmd = np.array([CMD_VX, 0.0], dtype=np.float32) # [vx, omega_z]
    
    data_log = {"time": [], "pitch": [], "roll": [], "cmd_vx": [], "actual_vx": [], "z_accel": []}
    
    step_counter = 0
    eval_step = 0
    
    # 物理底层控制指令初始化
    target_LB = target_LF = target_RF = target_RB = DEFAULT_LEG_POS
    omega_left = omega_right = 0.0

    while viewer.is_running() and eval_step < NUM_EVAL_STEPS:
        # --- 1.5秒物理锁死预热 ---
        if data.time < 1.5:
            data.ctrl[0:4] = DEFAULT_LEG_POS
            data.ctrl[4:8] = 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            continue
            
        # --- 50Hz 控制器/策略计算 ---
        if step_counter % DECIMATION == 0:
            roll, pitch, actual_vx, pitch_rate, z_accel, r = get_robot_state(data)
            
            # 记录数据 (50Hz)
            data_log["time"].append(eval_step * POLICY_DT)
            data_log["pitch"].append(np.rad2deg(pitch))
            data_log["roll"].append(np.rad2deg(roll))
            data_log["cmd_vx"].append(CMD_VX)
            data_log["actual_vx"].append(actual_vx)
            data_log["z_accel"].append(z_accel)
            
            if mode_name == "ours":
                # RL 推理
                obs_np = get_observation(data, raw_action, current_cmd, r.inv())
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                raw_action = action_tensor.squeeze().cpu().numpy()
                filtered_action = ACTION_ALPHA * raw_action + (1.0 - ACTION_ALPHA) * filtered_action
                
                cmd_v_x = filtered_action[0]      
                cmd_omega_z = filtered_action[1]  
                
                target_LB = DEFAULT_LEG_POS + filtered_action[2] * LEG_POS_SCALE
                target_LF = DEFAULT_LEG_POS + filtered_action[3] * LEG_POS_SCALE
                target_RB = DEFAULT_LEG_POS + filtered_action[4] * LEG_POS_SCALE
                target_RF = DEFAULT_LEG_POS + filtered_action[5] * LEG_POS_SCALE
                
            else:
                # 传统 PI 速度控制
                error_v = CMD_VX - actual_vx
                integral_error_v += error_v * POLICY_DT
                v_out = CMD_VX + 1.5 * error_v + 0.5 * integral_error_v
                cmd_v_x = np.clip(v_out, -1.0, 1.0)
                cmd_omega_z = 0.0
                
                if mode_name == "passive":
                    target_LB = target_LF = target_RF = target_RB = DEFAULT_LEG_POS
                elif mode_name == "active_pid":
                    Kp_pitch, Kd_pitch = 3.0, 0.5
                    pd_output = pitch * Kp_pitch + pitch_rate * Kd_pitch
                    pd_output = np.clip(pd_output, -1.0, 1.0)
                    
                    # 抬头(pitch>0)时：前腿收缩，后腿伸长
                    target_LF = DEFAULT_LEG_POS - pd_output * LEG_POS_SCALE
                    target_RF = DEFAULT_LEG_POS - pd_output * LEG_POS_SCALE
                    target_LB = DEFAULT_LEG_POS + pd_output * LEG_POS_SCALE
                    target_RB = DEFAULT_LEG_POS + pd_output * LEG_POS_SCALE

            # 差速运动学 (三种模式通用)
            v_left = cmd_v_x - cmd_omega_z * (BASE_WIDTH / 2.0)
            v_right = cmd_v_x + cmd_omega_z * (BASE_WIDTH / 2.0)
            omega_left = v_left / WHEEL_RADIUS
            omega_right = v_right / WHEEL_RADIUS
            
            eval_step += 1

        # --- 500Hz 物理引擎底层控制 ---
        data.ctrl[0] = target_LB
        data.ctrl[1] = target_LF
        data.ctrl[2] = target_RF
        data.ctrl[3] = target_RB  
        
        data.ctrl[4] = omega_left   
        data.ctrl[5] = omega_left   
        data.ctrl[6] = omega_right  
        data.ctrl[7] = omega_right  
        
        mujoco.mj_step(model, data)
        
        # 为了加速采集，渲染频率降低，或者注释掉 viewer.sync() 实现真·后台运行
        if step_counter % 10 == 0:
            viewer.sync()
            
        step_counter += 1

    df = pd.DataFrame(data_log)
    df["method"] = mode_name
    print(f"✅ {mode_name} 测试完成，收集到 {len(df)} 条数据。")
    return df

# ==========================================
# 4. 主函数：连续执行对比实验
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    device = torch.device("cpu")
    policy = torch.jit.load(POLICY_PATH, map_location=device)
    policy.eval()
    
    all_dataframes = []
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 依次运行三个基准测试
        df_passive = run_evaluation_mode("Passive Decoupled", model, data, viewer, policy, device)
        all_dataframes.append(df_passive)
        
        df_pid = run_evaluation_mode("Traditional Active PID", model, data, viewer, policy, device)
        all_dataframes.append(df_pid)
        
        df_ours = run_evaluation_mode("Ours (Integrated RL)", model, data, viewer, policy, device)
        all_dataframes.append(df_ours)

    # 合并并保存数据
    final_df = pd.concat(all_dataframes)
    save_dir = "/home/wjc/robot1/data"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "mujoco_experiments_data.csv")
    final_df.to_csv(save_path, index=False)
    
    print("===========================================")
    print(f"🎉 跨仿真器对比评估全部完成！")
    print(f"📊 数据已严格保存至: {save_path}")

if __name__ == "__main__":
    main()
import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 配置参数 (已根据你的配置文件修正)
# ==========================================
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_3.pt"  
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger2.xml"             

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
    # data.qpos[2] 是机器人在 MuJoCo 中的绝对高度 (Z轴)
    current_base_z = data.qpos[2]
    # 模拟平地上的 height scan：相对高度 = 0 - base_z
    height_scan = np.full(441, -current_base_z, dtype=np.float32)

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


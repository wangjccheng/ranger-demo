import mujoco
import mujoco.viewer
import numpy as np
import torch
import time
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. 基础配置参数
# ==========================================
POLICY_PATH = "/home/wjc/robot1/jit_models/policy_6.pt" # 请确保路径和文件名正确
XML_PATH = "/home/wjc/mujoco-3.5.0/urdf/xml/ranger2.xml"             

SIM_DT = 0.002           
POLICY_DT = 0.02         
DECIMATION = int(POLICY_DT / SIM_DT) 

# 运动学与动作缩放参数
LEG_POS_SCALE = 0.3  # URDF 极限 0.35 * 软限位 0.7 = 0.245
BASE_WIDTH = 0.68       
WHEEL_RADIUS = 0.19    
ACTION_ALPHA = 0.3     
RESIDUAL_SCALE = 5
# 默认状态
DEFAULT_LEG_POS = 0.05 

BASE_LIN_VEL_SCALE = 1.0   # 对应 base_scale[0]
BASE_ANG_VEL_SCALE = 0.5  # 对应 base_scale[1]

def get_real_height_scan(model, data):
    base_pos = data.qpos[:3]  
    base_quat = data.qpos[3:7] 
    
    # 算偏航角，保证雷达网格跟着车头转
    r = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
    yaw = r.as_euler('zyx')[0]
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

    grid_points = np.linspace(-1.0, 1.0, 25)
    lx, ly = np.meshgrid(grid_points, grid_points, indexing='xy')
    lx_flat = lx.flatten()
    ly_flat = ly.flatten()

    world_x = base_pos[0] + lx_flat * cos_yaw - ly_flat * sin_yaw
    world_y = base_pos[1] + lx_flat * sin_yaw + ly_flat * cos_yaw

    height_scan = np.zeros(625, dtype=np.float32)
    hit_points = np.zeros((625, 3), dtype=np.float32) 
    geomid = np.zeros(1, dtype=np.int32)
    
    # ★ 新增：定义一个射线过滤器。数组长度必须为6。
    # [1, 0, 0, 0, 0, 0] 代表：只检测第 0 层，忽略第 1~5 层
    ray_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)

    for i in range(625):
        pnt = np.array([world_x[i], world_y[i], 2.0], dtype=np.float64)
        vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        
        # ★ 修改：传入 ray_group 过滤器，并将 body_exclude 参数设为 -1 (不使用 Body 排除)
        dist = mujoco.mj_ray(model, data, pnt, vec, ray_group, 1, -1, geomid)
        
        if dist > 0:
            hit_z = 2.0 - dist
        else:
            hit_z = -1.0
            
        height_scan[i] = hit_z - base_pos[2]
        hit_points[i] = [world_x[i], world_y[i], hit_z + 0.02] 
        
    return height_scan, hit_points

def generate_wavy_terrain(model):
    """
    在内存中直接修改 hfield 数据，生成正弦波浪地形
    """
    hfield_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, "wavy_terrain")
    if hfield_id == -1: return

    nrow = model.hfield_nrow[hfield_id]
    ncol = model.hfield_ncol[hfield_id]
    start_idx = model.hfield_adr[hfield_id]

    # 遍历网格点，注入波浪高度 (归一化到 0~1 之间)
    for i in range(nrow):
        for j in range(ncol):
            # 将网格索引映射到物理空间相位
            x_phase = (i / nrow) * 6.0 * np.pi  # X方向生成3个波峰
            y_phase = (j / ncol) * 4.0 * np.pi  # Y方向生成2个波峰
            
            # 生成混合波浪 (0.0 表示最低点，1.0 表示达到 XML 中设置的 max_z)
            z_normalized = 0.5 * (np.sin(x_phase) * np.cos(y_phase)) + 0.5
            # 如果你想让波浪平缓一点，可以乘以一个衰减系数，比如 0.2
            model.hfield_data[start_idx + i * ncol + j] = z_normalized * 0.2

# ==========================================
# 2. 观测构建函数 (471维)
# ==========================================
def get_observation(model, data, prev_action, command):
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
    # ★ 调用真实的物理射线扫描：
    height_scan, hit_points = get_real_height_scan(model, data)
    
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
    return np.concatenate(obs_list).astype(np.float32), hit_points

# ==========================================
# 3. 主控制循环
# ==========================================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    # ★ 注入波浪地形
    generate_wavy_terrain(model)
    # =======================================================
    # ★ 新增：将机器人和地形的射线检测层分开
    # =======================================================
    for i in range(model.ngeom):
        # 如果所属的 body_id 是 0 (worldbody，即地面和波浪地形)
        if model.geom_bodyid[i] == 0:
            model.geom_group[i] = 0  # 留在第 0 层
        else:
            # 机器人身上的所有几何体，全部赶到第 1 层
            model.geom_group[i] = 1
            
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT

    # 加载带有 RNN(LSTM/GRU) 的策略网络
    device = torch.device("cpu")
    policy = torch.jit.load(POLICY_PATH, map_location=device)
    policy.eval()
    
    # 请根据你导出的模型确认 hidden_state 的形状
    # GRU 通常是 (1, 128) 或 (1, 1, 128)
    hidden_state = torch.zeros((1, 1, 128), dtype=torch.float32, device=device) 
    
    raw_action = np.zeros(10, dtype=np.float32)      
    filtered_action = np.zeros(10, dtype=np.float32) 
# ★ 1. 新增：定义暂停状态和运动指令
    paused = False
    cmd_x = 0.0    # 初始线速度 (默认停在原地)
    cmd_yaw = 0.0  # 初始角速度

    def key_callback(keycode):
        nonlocal paused, cmd_x, cmd_yaw
        
        if keycode == 32:  # 空格键 (Space): 暂停 / 继续
            paused = not paused
            
        elif keycode == 265:  # ⬆️ 向上箭头 (Up Arrow): 增加前进速度
            cmd_x += 0.2
        elif keycode == 264:  # ⬇️ 向下箭头 (Down Arrow): 减少前进速度 / 后退
            cmd_x -= 0.2
            
        elif keycode == 263:  # ⬅️ 向左箭头 (Left Arrow): 左转
            cmd_yaw += 0.2
        elif keycode == 262:  # ➡️ 向右箭头 (Right Arrow): 右转
            cmd_yaw -= 0.2
            
        elif keycode == 257:  # ⏎ 回车键 (Enter): 紧急刹车
            cmd_x = 0.0
            cmd_yaw = 0.0
            
        # 安全限制：防止多次按键导致指令爆炸
        cmd_x = np.clip(cmd_x, -1.0, 1.0)
        cmd_yaw = np.clip(cmd_yaw, -0.5, 0.5)
        
        # 打印当前指令
        if keycode != 32: 
            print(f"当前指令 -> 线速度: {cmd_x:.1f} m/s, 角速度: {cmd_yaw:.1f} rad/s")
    
    with mujoco.viewer.launch_passive(model, data,key_callback=key_callback) as viewer:
        
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
            
            if paused:
                viewer.sync()
                time.sleep(model.opt.timestep) # 稍微休眠，防止死循环跑满 CPU
                continue
            
            # =======================================================
            # ★ 核心修复：监听 Reset 事件，对大脑和物理姿态进行“彻底洗脑”
            # =======================================================
            if data.time == 0.0:
                # 1. 清空 GRU 记忆和动作滤波器
                hidden_state = torch.zeros((1, 1, 128), dtype=torch.float32, device=device)
                raw_action = np.zeros(10, dtype=np.float32)
                filtered_action = np.zeros(10, dtype=np.float32)
                
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
            current_cmd = np.array([cmd_x, 0.0, cmd_yaw], dtype=np.float32)
            
            # --- 神经网络推理层 (50Hz) ---
            if step_counter % DECIMATION == 0:
                obs_np, hit_points = get_observation(model, data, filtered_action, current_cmd) 
                obs_tensor = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action_tensor, hidden_state = policy(obs_tensor, hidden_state)
                
                raw_action = action_tensor.squeeze().cpu().numpy()
                filtered_action = ACTION_ALPHA * raw_action + (1.0 - ACTION_ALPHA) * filtered_action
                
                # 动作解包
                cmd_v_x = filtered_action[0] * BASE_LIN_VEL_SCALE      
                cmd_omega_z = filtered_action[1] * BASE_ANG_VEL_SCALE
                residual_raw = filtered_action[2:6]
                residual_cmd = residual_raw * RESIDUAL_SCALE
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
            
            data.ctrl[4] = omega_left+residual_cmd[0]
            data.ctrl[5] = omega_left+residual_cmd[1]
            data.ctrl[6] = omega_right+residual_cmd[2]
            data.ctrl[7] = omega_right+residual_cmd[3]   
 
            
            mujoco.mj_step(model, data)
            # =======================================================
            # ★ 极其惊艳的可视化渲染：在画面里画出 441 个红点
            # =======================================================
            # 使用 viewer.lock() 锁住渲染线程，防止边画边渲染导致内存崩溃
            with viewer.lock():
                viewer.user_scn.ngeom = 0 # 每次清空上一帧的几何体
                
                if 'hit_points' in locals():
                    for pt in hit_points:
                        # 向渲染器强行塞入一个球体 (Sphere)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.015, 0, 0]), # 半径 1.5 厘米
                            pos=pt,                       # 绝对坐标
                            mat=np.eye(3).flatten(),      # 默认无旋转
                            rgba=np.array([1, 0, 0, 1], dtype=np.float32) # 红色, 不透明度 1
                        )
                        viewer.user_scn.ngeom += 1
            
            # 同步画面
            viewer.sync()
            
            step_counter += 1
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()



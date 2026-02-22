import torch
import time
import numpy as np

def main():
    # ==========================================
    # 1. 加载模型
    # ==========================================
    model_path = "/home/wjc/robot1/jit_models/policy_lstm_1.pt"  # 替换为你导出的 JIT 模型的实际路径
    print(f"[INFO] 正在加载模型: {model_path}")
    try:
        policy = torch.jit.load(model_path)
        policy.eval()
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return

    # ==========================================
    # 2. 确认输入维度 (重点)
    # ==========================================
    # 你需要根据你在 SkidSteerLegObsCfg.Policy 中的定义，计算出精确的总维度
    # 本体特征大致为:
    # base_lin_vel(3) + base_ang_vel(3) + proj_grav(3) + cmd_vw(2) + 
    # wheel_vel(4) + leg_pos(4) + leg_vel(4) + leg_pos_norm(4) + 
    # slip_feat(6) + last_action(8) = 41 维
    # 高度扫描 (height_scan): GridPatternCfg 2m x 2m, res=0.1 -> 20x20 = 400 维
    # 假设总维度为 441，请将此数字替换为你代码报错时提示的真实 Observation 维度！
    num_obs = 480  
    
    # LSTM 配置 (与你在 rsl_rl_ppo_cfg.py 中的配置对齐)
    rnn_hidden_dim = 128
    rnn_num_layers = 1
    batch_size = 1 # 测试时只跑 1 个环境 (实车就是 1 个)

    # ==========================================
    # 3. 初始化隐藏状态 (LSTM 专属)
    # ==========================================
    # 对于 LSTM，隐状态由 hidden_state (h_0) 和 cell_state (c_0) 组成
    # 维度形状通常为: [rnn_num_layers, batch_size, rnn_hidden_dim]
    hidden_in = torch.zeros(rnn_num_layers, batch_size, rnn_hidden_dim)
    
    
    print(f"[INFO] 模型加载成功！准备进入控制循环 (50Hz)...")
    print("-" * 50)

    # ==========================================
    # 4. 模拟实车控制循环
    # ==========================================
    dt = 0.02  # 50Hz 控制频率
    num_actions = 8 # 4个轮子 + 4条腿

    for step in range(50): # 试跑 50 步
        start_time = time.time()
        
        # ----------------------------------------
        # A. 构造伪造观测 (Mock Observation)
        # 实车部署时，这一步会被替换为从你的传感器(LiDAR、轮速计等)读取数据并拼接
        # ----------------------------------------
        obs = torch.zeros(1, num_obs) # [batch_size, num_obs]
        
        # 我们可以给指令位强行塞入一个向前的速度，看看网络输出是否会有反应
        # 假设 cmd_vw 在第 9、10 位
        obs[0, 9] = 1.0  # target v_x = 1.0 m/s
        obs[0, 10] = 0.0 # target w_z = 0.0 rad/s
        
        # ----------------------------------------
        # B. 神经网络前向推理
        # ----------------------------------------
        with torch.no_grad():
            # RSL-RL 导出的带有 RNN 的模型，输入为 (obs, hidden_states)
            # 输出通常为 (action, hidden_states_out)
            output = policy(obs, hidden_in)
            
            action = output[0]       # 获取动作输出
            hidden_in = output[1]    # 更新隐状态供下一步使用！(千万不能漏掉)
        
        # ----------------------------------------
        # C. 动作解析与下发
        # ----------------------------------------
        action_np = action.squeeze(0).numpy() # 转为 numpy 数组 [8]
        
        # 按照 ActionsCfg 中的排列解析
        # 假设前 4 个是左右轮 (w_lf, w_lb, w_rf, w_rb)，后 4 个是腿 (g_lf, g_rf, g_lb, g_rb)
        wheel_actions = action_np[:4]
        leg_actions = action_np[4:]
        
        # 在实车中，你还要按照 base_scale 和 leg_scale/soft_limits 进行反归一化解算
        
        compute_time_ms = (time.time() - start_time) * 1000
        print(f"Step {step:03d} | 推理耗时: {compute_time_ms:.2f} ms")
        print(f"  -> 轮子指令 (Wheel) : {np.round(wheel_actions, 3)}")
        print(f"  -> 腿部指令 (Leg)   : {np.round(leg_actions, 3)}")
        print("-" * 50)
        
        # 模拟物理时间流逝
        time.sleep(dt)

if __name__ == "__main__":
    main()
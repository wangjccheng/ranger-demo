''''
import isaaclab.terrains as terrain_utils
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
from isaaclab.utils import configclass


# 如果你喜欢用 @configclass 也可以包一层，这里直接用原始 cfg 就行
WAVE_TERRAINS_CFG = terrain_utils.TerrainGeneratorCfg(
    seed=0,
    curriculum=True,           # 如果要配合 terrain_levels 做课程，可以先开
    size=(10.0, 10.0),         # 每个子地形的物理尺寸 [m]
    num_rows=8,                # 子地形行数
    num_cols=8,                # 子地形列数
    horizontal_scale=0.05,     # 高度场格子大小 (x,y 方向) [m]
    vertical_scale=0.01,       # 高度步长 (z 方向) [m]
    slope_threshold=None,      # 如需把过陡的坡改成竖直面可设置阈值
    sub_terrains={
        # 关键字段：这里用 HfWaveTerrainCfg 做子地形
        "waves": HfWaveTerrainCfg(
            size=(10.0, 10.0),
            horizontal_scale=0.05,
            vertical_scale=0.01,
            amplitude_range=(0.07, 0.12),  # ★ 波浪振幅区间 [m]
            num_waves=5,                 # ★ 波数（越大，波越密）
            proportion=1.0,              # 只生成这一种子地形
        ),
    },
)
'''
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# 定义波浪地形配置
WAVE_TERRAINS_CFG = TerrainGeneratorCfg(
    # 地形总尺寸设置
    size=(8.0, 8.0),
    border_width=20.0,
    
    # 难度等级数量（例如 10 级）
    num_rows=10,  
    num_cols=10, 
    
    # ★ 核心：定义不同地形块的生成规则
    sub_terrains={
        "waves": terrain_gen.MeshWaveTerrainCfg(
            proportion=1.0,  # 占 100% 的比例
            
            # --- 难度参数 (由简单到复杂) ---
            # amplitude: 波浪高度。范围 (0.1, 1.0) 表示 Level 0 时高 0.1m，Level 9 时高 1.0m
            amplitude_range=(0.0, 1.0), 
            
            # frequency: 波浪频率。范围 (1.0, 2.0) 表示波浪越来越密集/陡峭
            frequency_range=(1.0, 2.0),
        )
    },
    
    # 是否开启课程学习（会被 robot1_env_cfg.py 覆盖，但默认设为 True 也没问题）
    curriculum=True, 
)
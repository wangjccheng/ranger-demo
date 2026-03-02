
import isaaclab.terrains as terrain_utils
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg,HfRandomUniformTerrainCfg
from isaaclab.utils import configclass


# 如果你喜欢用 @configclass 也可以包一层，这里直接用原始 cfg 就行
WAVE_TERRAINS_CFG = terrain_utils.TerrainGeneratorCfg(
    seed=0,
    curriculum=True,           # 如果要配合 terrain_levels 做课程，可以先开
    size=(10.0, 10.0),         # 每个子地形的物理尺寸 [m]
    num_rows=8,                # 子地形行数
    num_cols=8,                # 子地形列数
    horizontal_scale=0.05,     # 高度场格子大小 (x,y 方向) [m]
    vertical_scale=0.005,       # 高度步长 (z 方向) [m]
    slope_threshold=None,      # 如需把过陡的坡改成竖直面可设置阈值
    sub_terrains={
            # 🌟 修改后的绝对平坦地形
        #"flat": HfRandomUniformTerrainCfg(
        #    proportion=0.3,          
        #    size=(10.0, 10.0),
        #    horizontal_scale=0.02,   
        #    vertical_scale=0.005,    
        #    noise_range=(0.0, 0.0),  # 高度范围全为0，保证是平地
        #    noise_step=0.01,         # 🌟 把这里的 0.0 改为 0.01 (只要不是0就行)
        #),
        # 关键字段：这里用 HfWaveTerrainCfg 做子地形
        "waves": HfWaveTerrainCfg(
            size=(10.0, 10.0),
            horizontal_scale=0.05,
            vertical_scale=0.01,
            amplitude_range=(0.00, 0.12),  # ★ 波浪振幅区间 [m]
            num_waves=4,                 # ★ 波数（越大，波越密）
            proportion=1.0,              # 只生成这一种子地形
        ),
    },
)

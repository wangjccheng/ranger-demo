import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 1. 学术图表全局设置 ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2.5,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

# 颜色主题 (突出 Ours 的优势)
palette = {
    "Passive Decoupled": "#A0A0A0",         # 灰色 (最差的 Baseline)
    "Traditional Active PID": "#4A90E2",    # 蓝色 (传统方法)
    "Ours (Integrated RL)": "#D0021B"       # 亮红色 (我们的方法)
}

# 加载数据
try:
    df = pd.read_csv("/home/wjc/robot1/data/paper_experiments_data1.csv")
except FileNotFoundError:
    print("Error: 找不到 paper_experiments_data1.csv。请先运行 evaluate_paper.py！")
    exit()

# =========================================================
# Figure 1: 速度追踪能力 (Velocity Tracking & Slip Analysis)
# 体现“全身协同”解决传统解耦控制导致车轮悬空打滑的痛点
# =========================================================
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x="time", y="actual_vx", hue="method", palette=palette, ax=ax1)

ax1.axhline(0.5, color='black', linestyle=':', linewidth=2, label="Command (0.5 m/s)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Forward Velocity $v_x$ (m/s)")
ax1.set_title("Velocity Tracking on Wave Terrain")
ax1.set_xlim(0, df["time"].max())

# 调整图例
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels, loc='lower right')
plt.tight_layout()
fig1.savefig("/home/wjc/robot1/paper/Fig1_Velocity_Tracking.pdf")
print("Saved Fig1_Velocity_Tracking.pdf")

# =========================================================
# Figure 2: 机身俯仰角稳定性 (Pitch Stability)
# 体现端到端 RL 隐式地形预判带来的平顺性
# =========================================================
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x="time", y="pitch", hue="method", palette=palette, ax=ax2)

ax2.axhline(0.0, color='black', linestyle=':', linewidth=2)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Chassis Pitch Angle (deg)")
ax2.set_title("Chassis Posture Stabilization")
ax2.set_xlim(0, df["time"].max())
ax2.legend(loc='upper right')
plt.tight_layout()
fig2.savefig("/home/wjc/robot1/paper/Fig2_Pitch_Stability.pdf")
print("Saved Fig2_Pitch_Stability.pdf")

# =========================================================
# Figure 3: 定量化柱状图 (RMSE 追踪误差 & 加速度震动方差)
# 论文 Results 表格/图表必备的硬核统计数据
# =========================================================
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

# 3a. 速度追踪 RMSE
df["vel_error_sq"] = (df["actual_vx"] - df["cmd_vx"]) ** 2
vel_rmse = np.sqrt(df.groupby("method")["vel_error_sq"].mean()).reset_index()
vel_rmse.rename(columns={"vel_error_sq": "RMSE"}, inplace=True)

sns.barplot(data=vel_rmse, x="method", y="RMSE", palette=palette, ax=ax3a)
ax3a.set_title("Velocity Tracking Error (RMSE)")
ax3a.set_ylabel("Error (m/s)")
ax3a.set_xlabel("")
ax3a.set_xticklabels(["Passive", "Trad. PID", "Ours\n(RL)"])
for i, v in enumerate(vel_rmse["RMSE"]):
    ax3a.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')

# 3b. Z轴加速度 Std (震动吸收能力)
z_accel_std = df.groupby("method")["z_accel"].std().reset_index()
z_accel_std.rename(columns={"z_accel": "Std"}, inplace=True)

sns.barplot(data=z_accel_std, x="method", y="Std", palette=palette, ax=ax3b)
ax3b.set_title("Vertical Vibration (Z-Accel Std)")
ax3b.set_ylabel("Std Dev (m/s²)")
ax3b.set_xlabel("")
ax3b.set_xticklabels(["Passive", "Trad. PID", "Ours\n(RL)"])
for i, v in enumerate(z_accel_std["Std"]):
    ax3b.text(i, v + 0.1, f"{v:.2f}", ha='center', fontweight='bold')

plt.tight_layout()
fig3.savefig("/home/wjc/robot1/paper/Fig3_Quantitative_Metrics.pdf")
print("Saved Fig3_Quantitative_Metrics.pdf")

plt.show()
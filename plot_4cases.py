import numpy as np
import matplotlib.pyplot as plt

# 设置基本参数
x = np.linspace(0, 10, 100)
mT = np.sin(x) + 1.5  # 生成一个正弦波形的掩蔽阈值

# 定义点的位置 (固定x位置为5)
x_point = 5

# 更新y和yhat为点的位置
# 1. y和yhat都大于masking threshold
y1_point = mT[50] + 0.5
yhat1_point = mT[50] + 0.25

# 2. yhat大于masking threshold而y小于masking threshold
y2_point = mT[50] - 0.5
yhat2_point = mT[50] + 0.5

# 3. y大于masking threshold而yhat小于masking threshold
y3_point = mT[50] + 0.5
yhat3_point = mT[50] - 0.5

# 4. y和yhat都小于masking threshold
y4_point = max(0.1, mT[50] - 0.3)  # 防止y小于0，也不与yhat重合
yhat4_point = mT[50] - 0.5

# 重新绘图
# fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 绘制每个子图
for i, ((y, yhat), ax) in enumerate(
    zip(
        [
            (y1_point, yhat1_point),
            (y2_point, yhat2_point),
            (y3_point, yhat3_point),
            (y4_point, yhat4_point),
        ],
        axs.flatten(),
    )
):
    ax.plot(x, mT, label="Masking Threshold", color="green")  # Masking threshold line
    ax.plot(x_point, y, "bo", label="True Amplitude")  # True point
    ax.plot(x_point, yhat, "ro", label="Predicted Amplitude")  # Predicted point

    # 特别处理图2，距离是yhat到masking threshold的距离
    # 对于图4，不画距离线，因为距离为0
    if i == 1:
        ax.plot(
            [x_point, x_point], [yhat, mT[50]], "k--", label="Distance to Compute"
        )  # Distance line
    elif i != 3:
        ax.plot(
            [x_point, x_point], [y, yhat], "k--", label="Distance to Compute"
        )  # Distance line

    ax.set_xlim(4.5, 5.5)
    ax.set_ylim(0, 1.25)
    ax.set_title(f"Case {i+1}")
    ax.legend()

plt.tight_layout()
# plt.title("Four Cases of Psychoacoustic Loss")
# plt.show()
plt.savefig("4cases.pdf")

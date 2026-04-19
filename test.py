import numpy as np
import matplotlib.pyplot as plt

# ✅ 必须放在 plt.subplots 之前！！！
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 正确的 4x4 哈尔变换矩阵
def haar_matrix(N):
    if N == 4:
        s = 1 / np.sqrt(2)
        H = np.array([
            [s,  s,  0,  0],
            [0,  0,  s,  s],
            [s, -s,  0,  0],
            [0,  0,  s, -s]
        ]) 
    return H

H = haar_matrix(4)
print("哈尔变换矩阵 T：")
print(H)

# 生成 4 个 2x2 基图像
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for i in range(2):
    for j in range(2):
        ti = H[i].reshape(-1, 1)
        tj = H[j].reshape(1, -1)
        basis = ti @ tj
        
        ax = axes[i, j]
        ax.imshow(basis, cmap='gray', vmin=-0.5, vmax=0.5)
        ax.set_title(f'BASE B{i}{j}')
        ax.axis('off')

#plt.suptitle("4x4 哈尔变换 基图像", fontsize=16)
plt.tight_layout()

# ✅ 保存图片（关键代码）
plt.savefig("haar_basis.png", dpi=150, bbox_inches='tight')
print("✅ 图片已保存为：haar_basis.png")

plt.show()
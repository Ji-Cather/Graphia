import numpy as np
from scipy.stats import boxcox, genextreme, norm
from scipy.stats import rankdata

def quantile_mapping(data, target_dist='uniform'):
    # 原始数据分位数
    ranks = rankdata(data) / len(data)
    if target_dist == 'uniform':
        return ranks
    elif target_dist == 'lognormal':
        from scipy.stats import lognorm
        return lognorm.ppf(ranks, s=1)  # 假设目标为对数正态分布
    else:
        raise ValueError("Unsupported target distribution")

# 原始数据（右偏）
data = np.array([0.1, 0.2, 0.3, 0.4, 0.5,0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,6.18])

# 方法1: Box-Cox 变换
transformed_data, lambda_opt = boxcox(data)
print(f"Optimal λ: {lambda_opt}")

# 检查Box-Cox变换后是否符合正态分布
from scipy.stats import shapiro, normaltest

# Shapiro-Wilk测试（样本量小时更适用）
stat, p_value = shapiro(transformed_data)
print(f"Shapiro-Wilk测试: 统计量={stat:.4f}, p值={p_value:.4f}")
print(f"数据{'符合' if p_value > 0.05 else '不符合'}正态分布 (p > 0.05)")

# D'Agostino's K^2测试
k2_stat, k2_p_value = normaltest(transformed_data)
print(f"D'Agostino's K^2测试: 统计量={k2_stat:.4f}, p值={k2_p_value:.4f}")

# 可视化比较
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 5))

# 原始数据直方图
plt.subplot(1, 2, 1)
plt.hist(data, bins=10, alpha=0.7)
plt.title("原始数据分布")

# 变换后数据直方图与正态分布拟合曲线
plt.subplot(1, 2, 2)
plt.hist(transformed_data, bins=10, alpha=0.7, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(transformed_data), np.std(transformed_data))
plt.plot(x, p, 'k', linewidth=2)
plt.title("Box-Cox变换后数据分布与正态拟合")
plt.tight_layout()

# 方法2: 广义极值分布（GEV）
shape, loc, scale = genextreme.fit(data)
gev_data = genextreme.rvs(shape, loc, scale, size=len(data))

# # 方法3: 分位数映射到正态分布
# ranked_data = quantile_mapping(data, target_dist='normal')

# 方法4: 分段变换（截断大值）
threshold = 3
piecewise_data = np.where(data > threshold, threshold, np.log(data + 1e-5))

import numpy as np
import torch
from scipy.stats import powerlaw

# 生成符合PowerLaw分布的度值（离散化）
def generate_powerlaw_degrees(n_samples=100, alpha=2.5, min_degree=0, max_degree=1000):
    # 生成连续PowerLaw分布，然后离散化为整数
    samples = powerlaw.rvs(alpha, size=n_samples) * (max_degree - min_degree) + min_degree
    return np.floor(samples).astype(int)

# 生成序列化度值（每个节点的历史度值序列）
def generate_degree_sequences(degrees, seq_length=5):
    sequences = []
    for i in range(len(degrees) - seq_length):
        seq = degrees[i:i+seq_length]
        target = degrees[i+seq_length]  # 下一个度值
        sequences.append((seq, target))
    return sequences

# 示例数据
np.random.seed(42)
powerlaw_degrees = generate_powerlaw_degrees(n_samples=10000, alpha=2.5)
sequences = generate_degree_sequences(powerlaw_degrees, seq_length=5)


import torch.nn as nn
# 加权重采样（保留PowerLaw特性）
def weighted_resample(data, alpha=2.5):
    weights = 1 / (data ** alpha)  # 权重与度值倒数成正比
    weights /= np.sum(weights)    # 归一化
    resampled_indices = np.random.choice(len(data), size=len(data), p=weights)
    return data[resampled_indices]

# 应用加权重采样
resampled_degrees = weighted_resample(powerlaw_degrees)

class FlowMatchingModel(nn.Module):
    def __init__(self, seq_length=5, hidden_dim=64):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.vector_field = nn.Sequential(
            nn.Linear(hidden_dim + 2, 64),  # 输入: [hidden_state, t, current_degree]
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出: 向量场 v_t
        )

    def forward(self, t, history, current_degree):
        # history: [batch_size, seq_length, 1]
        # t: [batch_size, 1]
        # current_degree: [batch_size, 1]
        out, _ = self.encoder(history)
        context = out[:, -1, :]  # 取最后一个时间步的隐藏状态
        input_vec = torch.cat([context, t, current_degree], dim=1)
        return self.vector_field(input_vec)


from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# 转换为Tensor
def collate_fn(batch):
    histories, targets = zip(*batch)
    histories = torch.tensor([h for h in histories], dtype=torch.float32).unsqueeze(-1)
    targets = torch.tensor([t for t in targets], dtype=torch.float32).unsqueeze(-1)
    return histories, targets

dataset = TensorDataset(*collate_fn(sequences))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化模型和优化器
model = FlowMatchingModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 训练循环
epochs = 10
for epoch in range(epochs):
    for histories, targets in tqdm(dataloader):
        t = torch.rand_like(targets)  # 随机时间步 t ∈ [0,1]
        # 真实向量场 v_true = d/dt x(t) = x(1) - x(0) （简化假设）
        v_true = (targets - histories[:, -1, :])  # 简单近似
        # 模型预测
        v_pred = model(t, histories, histories[:, -1, :])
        # 损失函数
        loss = ((v_pred - v_true) ** 2).mean()
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# 生成新度值（反向ODE求解）
def generate_degrees(model, n_samples=100, seq_length=5):
    # 初始噪声（x_0 ~ N(0,1)）
    x_0 = torch.randn(n_samples, 1)
    t_steps = torch.linspace(1, 0, 100)  # 时间步从1到0
    history = torch.randint(1, 100, (n_samples, seq_length, 1)).float()  # 假设历史序列
    with torch.no_grad():
        for t in t_steps:
            v = model(t.unsqueeze(0).repeat(n_samples, 1), history, x_0)
            x_0 -= v  # 反向ODE: dx/dt = -v_t
    return x_0.squeeze().numpy()

# 校准生成度值（分位数映射）
def calibrate_degrees(generated_degrees, powerlaw_cdf):
    from scipy.stats import rankdata
    ranks = rankdata(generated_degrees) / len(generated_degrees)  # 生成分布的分位数
    return powerlaw.ppf(ranks, 2.5)  # PowerLaw分位数映射

# 生成并校准
generated_degrees = generate_degrees(model, n_samples=1000)
calibrated_degrees = calibrate_degrees(generated_degrees, powerlaw)

# 评估生成分布
from scipy.stats import kstest
ks_stat, p_value = kstest(calibrated_degrees, lambda x: powerlaw.cdf(x, 2.5))
print(f"Kolmogorov-Smirnov Test: Stat={ks_stat:.4f}, p-value={p_value:.4f}")


import matplotlib.pyplot as plt

# 绘制直方图
plt.figure(figsize=(10, 5))
plt.hist(powerlaw_degrees, bins=100, alpha=0.5, label="True PowerLaw")
plt.hist(calibrated_degrees, bins=100, alpha=0.5, label="Generated + Calibrated")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Degree Distribution Comparison")
plt.show()

import torch
import torch.nn as nn
import math
import networkx as nx
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
from scipy import stats
import time
from tqdm import tqdm


def log_prop(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-30
    logw = torch.log(x + eps)
    assert not torch.isnan(logw).any(), "logw contains NaN values"
    return logw

# 添加直方图匹配损失函数
class HistogramMatchingLoss(nn.Module):
    def __init__(self, num_bins=10, use_wasserstein=True, use_mmd=True, kernel_type='gaussian', sigma=1.0):
        """
        基于直方图匹配的损失函数
        :param num_bins: 直方图的箱数
        :param use_wasserstein: 是否使用Wasserstein距离，否则使用KL散度
        :param use_mmd: 是否使用MMD距离
        :param kernel_type: 核函数类型，'gaussian'或'rbf'
        :param sigma: 高斯核的带宽参数
        """
        super().__init__()
        self.num_bins = num_bins
        self.use_wasserstein = use_wasserstein
        self.use_mmd = use_mmd
        self.kernel_type = kernel_type
        self.sigma = sigma
        
    def compute_histogram(self, degrees, num_bins=None):
        """
        计算度数直方图
        :param degrees: 节点度数张量
        :param num_bins: 直方图的箱数
        :return: 直方图分布
        """
        if num_bins is None:
            num_bins = self.num_bins
            
        # 将度数展平
        flat_degrees = degrees.flatten().cpu().detach().numpy()
        
        # 计算直方图
        hist, _ = np.histogram(flat_degrees, bins=num_bins, density=True)
        
        # 归一化
        if np.sum(hist) > 0:
            hist = hist / np.sum(hist)
            
        return hist
    
    def gaussian_kernel(self, x, y):
        """
        计算高斯核
        :param x: 第一个样本
        :param y: 第二个样本
        :return: 核函数值
        """
        return torch.exp(-torch.sum((x - y) ** 2) / (2 * self.sigma ** 2))
    
    def mmd_distance(self, x, y):
        """
        计算MMD距离
        :param x: 第一个分布的样本
        :param y: 第二个分布的样本
        :return: MMD距离
        """
        # 确保x和y是张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
            
        # 计算核矩阵
        xx = torch.zeros((x.shape[0], x.shape[0]))
        yy = torch.zeros((y.shape[0], y.shape[0]))
        xy = torch.zeros((x.shape[0], y.shape[0]))
        
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                xx[i, j] = self.gaussian_kernel(x[i], x[j])
                
        for i in range(y.shape[0]):
            for j in range(y.shape[0]):
                yy[i, j] = self.gaussian_kernel(y[i], y[j])
                
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                xy[i, j] = self.gaussian_kernel(x[i], y[j])
                
        # 计算MMD距离
        mmd = torch.mean(xx) + torch.mean(yy) - 2 * torch.mean(xy)
        return mmd
    
    def forward(self, pred_degrees, target_degrees):
        """
        计算直方图匹配损失
        :param pred_degrees: 预测的节点度数
        :param target_degrees: 目标节点度数
        :return: 损失值
        """
        # 计算预测和目标直方图
        pred_hist = self.compute_histogram(pred_degrees)
        target_hist = self.compute_histogram(target_degrees)
        
        if self.use_mmd:
            # 使用MMD距离
            # 将直方图转换为样本点
            pred_samples = []
            target_samples = []
            
            # 获取直方图的bin边界
            pred_flat = pred_degrees.flatten().cpu().detach().numpy()
            target_flat = target_degrees.flatten().cpu().detach().numpy()
            
            pred_hist, pred_bins = np.histogram(pred_flat, bins=self.num_bins, density=True)
            target_hist, target_bins = np.histogram(target_flat, bins=self.num_bins, density=True)
            
            # 为每个bin创建样本点
            for i in range(len(pred_bins) - 1):
                bin_center = (pred_bins[i] + pred_bins[i+1]) / 2
                count = int(pred_hist[i] * 100)  # 根据bin的高度创建样本点
                pred_samples.extend([bin_center] * max(1, count))
                
            for i in range(len(target_bins) - 1):
                bin_center = (target_bins[i] + target_bins[i+1]) / 2
                count = int(target_hist[i] * 100)  # 根据bin的高度创建样本点
                target_samples.extend([bin_center] * max(1, count))
                
            # 确保有足够的样本点
            if len(pred_samples) == 0:
                pred_samples = [0]
            if len(target_samples) == 0:
                target_samples = [0]
                
            # 转换为张量
            pred_samples = torch.tensor(pred_samples, dtype=torch.float32).reshape(-1, 1)
            target_samples = torch.tensor(target_samples, dtype=torch.float32).reshape(-1, 1)
            
            # 计算MMD距离
            mmd_dist = self.mmd_distance(pred_samples, target_samples)
            return mmd_dist.to(pred_degrees.device)
            
        elif self.use_wasserstein:
            # 使用Wasserstein距离
            # 创建累积分布函数
            pred_cdf = np.cumsum(pred_hist)
            target_cdf = np.cumsum(target_hist)
            
            # 计算Wasserstein距离
            distance = np.sum(np.abs(pred_cdf - target_cdf)) / len(pred_cdf)
            return torch.tensor(distance, device=pred_degrees.device)
        else:
            # 使用KL散度
            eps = 1e-10
            kl_div = np.sum(target_hist * np.log((target_hist + eps) / (pred_hist + eps)))
            return torch.tensor(kl_div, device=pred_degrees.device)

# 添加批量归一化损失函数
class BatchNormDistributionLoss(nn.Module):
    def __init__(self):
        """
        批量归一化分布损失函数
        """
        super().__init__()
        
    def forward(self, pred_degrees, target_degrees):
        """
        计算批量归一化分布损失
        :param pred_degrees: 预测的节点度数
        :param target_degrees: 目标节点度数
        :return: 损失值
        """
        # 对预测和目标度数进行归一化
        pred_mean = pred_degrees.mean()
        pred_std = pred_degrees.std() + 1e-6
        norm_pred = (pred_degrees - pred_mean) / pred_std
        
        target_mean = target_degrees.mean()
        target_std = target_degrees.std() + 1e-6
        norm_target = (target_degrees - target_mean) / target_std
        
        # 计算归一化后的分布差异
        # 使用均方误差
        mse_loss = torch.mean((norm_pred - norm_target) ** 2)
        
        # 使用分布统计量的差异
        mean_loss = torch.abs(pred_mean - target_mean)
        std_loss = torch.abs(pred_std - target_std)
        
        # 总损失
        total_loss = mse_loss + 0.1 * mean_loss + 0.1 * std_loss
        
        return total_loss

class DegreeQuantileConverter(nn.Module):
    def __init__(self, max_degree: int, 
                 num_quantiles:int, 
                 log2_quantile:bool = False,
                 degree_list: list = None):
        """
        将连续的degree值转换为k个quantile的表示
        :param max_degree: int, 最大度数
        :param num_quantiles: int, quantile的数量
        :param log2_quantile: bool, 是否使用log2方式计算quantile
        :param degree_list: list, 可选的degree列表，用于计算分位数
        """
        super().__init__()

        # 使用提供的degree_list计算分位数
        sorted_degrees = sorted(degree_list)
        total = len(sorted_degrees)
        if degree_list is not None:
            self.k = num_quantiles
            quantile_values = [0]
            positions = np.linspace(0, 1, num_quantiles)
            for position in positions:
                # 使用线性插值计算分位数
                pos = position * (total - 1)
                idx_low = int(pos)
                idx_high = min(idx_low + 1, total - 1)
                frac = pos - idx_low
                value = sorted_degrees[idx_low] * (1 - frac) + sorted_degrees[idx_high] * frac
                quantile_values.append(value)
            quantile_values = torch.tensor(quantile_values).unique()
            self.k = len(quantile_values)
            self.register_buffer('quantile_values', quantile_values.float())
        elif log2_quantile:
            log_2_k = int(torch.log2(torch.tensor(max_degree))) + 1
            self.k = log_2_k + 1
            self.register_buffer('quantile_values', torch.tensor([0,*[2**i for i in range(log_2_k)]]).float())
        else:
            self.k = num_quantiles
            # 创建均匀分布的分位数
            self.register_buffer('quantile_values', torch.linspace(0, max_degree, num_quantiles).float())

        
    def transform(self, degrees: torch.Tensor) -> torch.Tensor:
        """
        将degrees转换为quantile表示
        :param degrees: torch.Tensor, shape (batch_size, seq_len, 1)
        :return: torch.Tensor, shape (batch_size, seq_len, k)
        """
        batch_size, seq_len, _ = degrees.shape
        result = torch.zeros(batch_size, seq_len, self.k, device=degrees.device)
        
        # 对每个degree值计算其对应的quantile分布
        quantile_values_expanded = self.quantile_values.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # 计算权重
        weights = torch.zeros(batch_size, seq_len, self.k, device=degrees.device)
        
        # 处理小于等于最大quantile值的情况
        for i in range(self.k - 1):
            # 计算每个degree在quantile区间的相对位置
            lower_bound = quantile_values_expanded[:, :, i]
            upper_bound = quantile_values_expanded[:, :, i+1]
            
            # 计算权重
            mask = (degrees.squeeze(-1) >= lower_bound) & (degrees.squeeze(-1) < upper_bound)
            pos = (degrees.squeeze(-1) - lower_bound) / (upper_bound - lower_bound + 1e-10)
            pos = torch.clamp(pos, 0, 1)
            
            # 分配权重
            weights[:, :, i] = (1 - pos) * mask.float()
            weights[:, :, i+1] += pos * mask.float()
        
        # 处理大于最大quantile值的情况
        max_mask = (degrees.squeeze(-1) >= self.quantile_values[-1]).unsqueeze(-1).expand(-1, -1, self.k)
        if max_mask.sum() > 0:
            max_mask[:,:,:-1] = 0
            weights[max_mask] = 1
        
        result = weights
                
        log_result = log_prop(result)
        return log_result
        
    def inverse_transform(self, log_weights: torch.Tensor) -> torch.Tensor:
        """
        将quantile权重转换回degree值
        :param log_weights: torch.Tensor, shape (batch_size, k)
        :return: torch.Tensor, shape (batch_size, 1)
        """
        # 计算加权和得到degree值
        weights = torch.exp(log_weights)
        degrees = torch.sum(weights * self.quantile_values, dim=-1, keepdim=True)
        return degrees

# 添加组合损失函数
class CombinedDegreeLoss(nn.Module):
    def __init__(self, degree_weight=0.5, 
                 quantile_weight=0.3, histogram_weight=0.2, num_bins=10, 
                 use_mmd=False,
                 use_wasserstein = False):
        """
        组合损失函数，包括MSE、quantile和直方图匹配损失
        :param degree_weight: MSE损失的权重
        :param quantile_weight: quantile损失的权重
        :param histogram_weight: 直方图匹配损失的权重
        :param num_bins: 直方图的箱数
        :param use_mmd: 是否使用MMD距离
        """
        super().__init__()
        self.degree_criterion = nn.MSELoss()
        self.quantile_criterion = nn.NLLLoss()
        self.histogram_criterion = HistogramMatchingLoss(num_bins=num_bins, use_mmd=use_mmd, use_wasserstein=use_wasserstein)
        self.degree_weight = degree_weight
        self.quantile_weight = quantile_weight
        self.histogram_weight = histogram_weight
        
        # 添加损失计算统计
        self.degree_loss_count = 0
        self.quantile_loss_count = 0
        self.histogram_loss_count = 0
        self.degree_loss_time = 0.0
        self.quantile_loss_time = 0.0
        self.histogram_loss_time = 0.0
        
    def forward(self, reconstructed_degrees, target_degree, log_quantile_weights, target_quantile_weights):
        """
        计算组合损失
        :param reconstructed_degrees: 重构的度数
        :param target_degree: 目标度数
        :param log_quantile_weights: 预测的quantile权重
        :param target_quantile_weights: 目标quantile权重
        :return: 总损失
        """
        # 初始化损失值
        degree_loss = torch.tensor(0.0, device=reconstructed_degrees.device)
        quantile_loss = torch.tensor(0.0, device=reconstructed_degrees.device)
        histogram_loss = torch.tensor(0.0, device=reconstructed_degrees.device)
        
        # MSE损失
        if self.degree_weight > 0:
            import time
            start_time = time.time()
            degree_loss = self.degree_criterion(reconstructed_degrees, target_degree)
            self.degree_loss_time += time.time() - start_time
            self.degree_loss_count += 1
        
        # Quantile损失
        if self.quantile_weight > 0:
            import time
            start_time = time.time()
            quantile_loss = -torch.mean(torch.sum(target_quantile_weights * log_quantile_weights, dim=1))
            self.quantile_loss_time += time.time() - start_time
            self.quantile_loss_count += 1
        
        # 直方图匹配损失
        if self.histogram_weight > 0:
            import time
            start_time = time.time()
            histogram_loss = self.histogram_criterion(reconstructed_degrees, target_degree)
            self.histogram_loss_time += time.time() - start_time
            self.histogram_loss_count += 1
        
        # 总损失
        total_loss = (self.degree_weight * degree_loss + 
                      self.quantile_weight * quantile_loss + 
                      self.histogram_weight * histogram_loss)
        
        return total_loss, degree_loss, quantile_loss, histogram_loss
    
    def get_loss_stats(self):
        """
        获取损失计算统计信息
        :return: 损失计算统计信息
        """
        return {
            'degree_loss_count': self.degree_loss_count,
            'quantile_loss_count': self.quantile_loss_count,
            'histogram_loss_count': self.histogram_loss_count,
            'degree_loss_time': self.degree_loss_time,
            'quantile_loss_time': self.quantile_loss_time,
            'histogram_loss_time': self.histogram_loss_time,
            'avg_degree_loss_time': self.degree_loss_time / max(1, self.degree_loss_count),
            'avg_quantile_loss_time': self.quantile_loss_time / max(1, self.quantile_loss_count),
            'avg_histogram_loss_time': self.histogram_loss_time / max(1, self.histogram_loss_count)
        }

class DegreePredictor(nn.Module):
    def __init__(self, seq_len: int, max_degree: int, hidden_dim: int = 128, num_heads: int = 4, \
        num_layers: int = 2, dropout: float = 0.1, condition_dim: int = 0, degree_list: list = None, num_quantiles: int = None,
        log2_quantile:bool = False):
        """
        基于Transformer的度序列预测模型
        :param seq_len: 输入序列长度
        :param max_degree: 最大度数(输出分布的维度)
        :param hidden_dim: 隐藏层维度
        :param num_heads: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比率
        :param condition_dim: 条件维度
        :param degree_list: 可选的degree列表，用于计算分位数
        :param num_quantiles: 可选的quantile数量，如果提供degree_list则忽略此参数
        """
        super().__init__()
        
        if condition_dim > 0:
            self.hidden_dim = hidden_dim + condition_dim
        else:
            self.hidden_dim = hidden_dim
            
        self.seq_len = seq_len
        self.max_degree = max_degree
        
        # 初始化degree quantile转换器
        self.degree_converter = DegreeQuantileConverter(max_degree=max_degree, 
                                                        degree_list=degree_list, 
                                                        num_quantiles=num_quantiles,
                                                        log2_quantile=log2_quantile)
        
        self.k_quantiles = self.degree_converter.k
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 序列嵌入层 - 现在输入是k_quantiles维的quantile向量
        self.seq_embedding = nn.Linear(self.k_quantiles, hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出层,预测下一时刻的度数分布
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.k_quantiles),
            nn.LogSoftmax(dim=-1)  # 输出log概率
        )

    def forward(self, 
                x_degree: torch.Tensor, # shape (batch_size, seq_len, 1)
                x_condition: torch.Tensor = None, # shape (batch_size, seq_len, condition_dim)
                return_quantile: bool = False):
                
        """
        前向传播
        :param x: 输入序列张量, shape (batch_size, seq_len, 1)
        :param x_condition: 条件张量, shape (batch_size, seq_len, condition_dim)
        :param return_quantile: 是否返回quantile分布
        :return: 如果return_quantile为True，返回quantile分布和degree值；否则只返回degree值
        """
        # 将degree转换为quantile表示
        x_quantile = self.degree_converter.transform(x_degree)
        assert torch.sum(self.degree_converter.inverse_transform(x_quantile) - x_degree) < 1e-2, "quantile reconstruction error is too large"
        
        # 序列嵌入
        x = self.seq_embedding(x_quantile)  # (batch_size, seq_len, hidden_dim)
        if x_condition is not None:
            x = torch.cat((x, x_condition), dim=-1) # (batch_size, seq_len, hidden_dim + condition_dim)
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)  # (batch_size, seq_len, hidden_dim)
        
        x = x[:,-1,:]
        # 预测下一时刻的度数分布（log概率）
        log_quantile_weights = self.output_layer(x)  # (batch_size, k_quantiles)
        
        
        # 将quantile权重转换回degree值
        degree_values = self.degree_converter.inverse_transform(log_quantile_weights)
        
        
        
        if return_quantile:
            return log_quantile_weights, degree_values
        return degree_values

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        位置编码层
        :param d_model: 模型维度
        :param dropout: dropout比率
        :param max_len: 最大序列长度
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        """
        添加位置编码
        :param x: 输入张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
    



if __name__ == "__main__":
    # 设置随机种子以确保结果可复现
    torch.manual_seed(42)
    
    # 加载图数据
    import pickle as pkl
    import random
    import networkx as nx
    from torch.utils.data import Dataset, DataLoader
    
    # 加载图数据
    nx_graphs = pkl.load(open(f"/data/jiarui_ji/graph-generation-EDGE/graphs/community.pkl", 'rb'))
    random.shuffle(nx_graphs)
    l = len(nx_graphs)
    train_nx_graphs = nx_graphs[:int(0.2*l)]
    eval_nx_graphs = nx_graphs[int(0.8*l):int(0.9*l)]
    test_nx_graphs = nx_graphs[int(0.9*l):] 
    
    # 创建数据集类
    class DegreeDataset(Dataset):
        def __init__(self, graphs, seq_len=1):
            self.graphs = graphs
            self.seq_len = seq_len
            self.degrees = []
            
            # 收集所有图的节点度数
            for graph in graphs:
                degrees = [d for n, d in graph.degree()]
                self.degrees.extend(degrees)
            
            # 转换为张量并复制最后一个维度
            self.degrees = torch.tensor(self.degrees, dtype=torch.float32).reshape(-1, 1, 1).repeat(1, seq_len, 1)
            
        def __len__(self):
            return len(self.degrees)
        
        def __getitem__(self, idx):
            return self.degrees[idx]
    
    # 创建数据集
    seq_len = 5  # 使用更长的序列长度
    train_dataset = DegreeDataset(train_nx_graphs, seq_len=seq_len)
    eval_dataset = DegreeDataset(eval_nx_graphs, seq_len=seq_len)
    test_dataset = DegreeDataset(test_nx_graphs, seq_len=seq_len)
    
    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 获取最大度数
    max_degree = max([max(d for n, d in graph.degree()) for graph in nx_graphs])
    
    # 收集所有度数用于计算分位数
    all_degrees = []
    for graph in nx_graphs:
        degrees = [d for n, d in graph.degree()]
        all_degrees.extend(degrees)
    
    # 创建模型 - 使用degree_list计算分位数
    model = DegreePredictor(seq_len=seq_len, max_degree=max_degree, degree_list=all_degrees, num_quantiles=10)
    
    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建组合损失函数
    combined_loss = CombinedDegreeLoss(degree_weight=0.5, quantile_weight=0.3, histogram_weight=0.2, use_wasserstein=True)
    
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"训练数据大小: {len(train_dataset)}")
    print(f"验证数据大小: {len(eval_dataset)}")
    print(f"测试数据大小: {len(test_dataset)}")
    print(f"最大度数: {max_degree}")
    print(f"使用设备: {device}")
    print(f"分位数数量: {model.k_quantiles}")
    print(f"分位数值: {model.degree_converter.quantile_values}")
    
    # 训练循环 - 使用因果推理方式训练
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, degrees in tqdm(enumerate(train_loader),"training step"):
            degrees = degrees.to(device)
            
            # 对于每个时间步，使用前面的序列预测下一个值
            batch_loss = 0
            for t in range(1, seq_len):
                # 使用前t个时间步预测第t+1个时间步
                input_seq = degrees[:, :t, :]
                target_degree = degrees[:, t, :]
                
                # 前向传播，获取quantile权重和重构的degree值
                log_quantile_weights, predicted_degrees = model(input_seq, return_quantile=True)
                
                # 计算原始degree的quantile表示
                target_quantile = model.degree_converter.transform(target_degree.unsqueeze(1))
                target_quantile_weights = torch.exp(target_quantile).squeeze(1)
                
                # 使用组合损失函数
                loss, degree_loss, quantile_loss, histogram_loss = combined_loss(
                    predicted_degrees, target_degree, log_quantile_weights, target_quantile_weights
                )
                
                batch_loss += loss
            
            # 平均每个时间步的损失
            loss = batch_loss / (seq_len - 1)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # 打印损失计算统计
        loss_stats = combined_loss.get_loss_stats()
        print("\n训练阶段损失计算统计:")
        print(f"  - MSE损失: 计算次数={loss_stats['degree_loss_count']}, 总时间={loss_stats['degree_loss_time']:.4f}秒, 平均时间={loss_stats['avg_degree_loss_time']:.6f}秒")
        print(f"  - Quantile损失: 计算次数={loss_stats['quantile_loss_count']}, 总时间={loss_stats['quantile_loss_time']:.4f}秒, 平均时间={loss_stats['avg_quantile_loss_time']:.6f}秒")
        print(f"  - 直方图匹配损失: 计算次数={loss_stats['histogram_loss_count']}, 总时间={loss_stats['histogram_loss_time']:.4f}秒, 平均时间={loss_stats['avg_histogram_loss_time']:.6f}秒")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for degrees in eval_loader:
                degrees = degrees.to(device)
                
                # 验证时也使用因果推理方式
                batch_loss = 0
                for t in range(1, seq_len):
                    input_seq = degrees[:, :t, :]
                    target_degree = degrees[:, t, :]
                    
                    log_quantile_weights, predicted_degrees = model(input_seq, return_quantile=True)
                    target_quantile = model.degree_converter.transform(target_degree.unsqueeze(1))
                    target_quantile_weights = torch.exp(target_quantile).squeeze(1)
                    
                    # 使用组合损失函数
                    loss, degree_loss, quantile_loss, histogram_loss = combined_loss(
                        predicted_degrees, target_degree, log_quantile_weights, target_quantile_weights
                    )
                    
                    batch_loss += loss
                
                loss = batch_loss / (seq_len - 1)
                val_loss += loss.item()
        
        val_loss /= len(eval_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # 打印验证阶段损失计算统计
        loss_stats = combined_loss.get_loss_stats()
        print("\n验证阶段损失计算统计:")
        print(f"  - MSE损失: 计算次数={loss_stats['degree_loss_count']}, 总时间={loss_stats['degree_loss_time']:.4f}秒, 平均时间={loss_stats['avg_degree_loss_time']:.6f}秒")
        print(f"  - Quantile损失: 计算次数={loss_stats['quantile_loss_count']}, 总时间={loss_stats['quantile_loss_time']:.4f}秒, 平均时间={loss_stats['avg_quantile_loss_time']:.6f}秒")
        print(f"  - 直方图匹配损失: 计算次数={loss_stats['histogram_loss_count']}, 总时间={loss_stats['histogram_loss_time']:.4f}秒, 平均时间={loss_stats['avg_histogram_loss_time']:.6f}秒")
    
    # 测试 - 使用seq-1的输入预测seq的输出
    model.eval()
    test_inputs = []
    test_targets = []
    predicted_degrees_list = []
    
    with torch.no_grad():
        for degrees in test_loader:
            degrees = degrees.to(device)
            
            # 使用前seq_len-1个时间步预测最后一个时间步
            input_seq = degrees[:, :-1, :]
            target_degree = degrees[:, -1, :]
            
            # 预测最后一个时间步的度数
            _, predicted_degrees = model(input_seq, return_quantile=True)
            
            test_inputs.append(input_seq)
            test_targets.append(target_degree)
            predicted_degrees_list.append(predicted_degrees)
    
    # 合并所有批次的结果
    test_inputs = torch.cat(test_inputs, dim=0)
    test_targets = torch.cat(test_targets, dim=0)
    predicted_degrees = torch.cat(predicted_degrees_list, dim=0)
    
    # 计算MSE和MAE
    degree_mse = torch.mean((test_targets - predicted_degrees) ** 2)
    degree_mae = torch.mean(torch.abs(test_targets - predicted_degrees))
    
    # 计算MMD距离
    import time
    start_time = time.time()
    histogram_loss = HistogramMatchingLoss(num_bins=10, use_mmd=True, sigma=1.0)
    mmd_distance = histogram_loss(predicted_degrees, test_targets)
    mmd_time = time.time() - start_time
    
    # 计算各个损失的时间
    start_time = time.time()
    mse_loss = nn.MSELoss()(predicted_degrees, test_targets)
    mse_time = time.time() - start_time
    
    start_time = time.time()
    # 计算quantile损失
    target_quantile = model.degree_converter.transform(test_targets.unsqueeze(1))
    target_quantile_weights = torch.exp(target_quantile).squeeze(1)
    _, predicted_quantile = model(test_inputs, return_quantile=True)
    quantile_loss = -torch.mean(torch.sum(target_quantile_weights * predicted_quantile, dim=1))
    quantile_time = time.time() - start_time
    
    print(f"测试输入形状: {test_inputs.shape}")
    print(f"测试目标形状: {test_targets.shape}")
    print(f"预测输出形状: {predicted_degrees.shape}")
    print(f"Degree MSE: {degree_mse.item():.4f}")
    print(f"Degree MAE: {degree_mae.item():.4f}")
    print(f"MMD Distance: {mmd_distance.item():.4f}")
    print(f"损失计算时间:")
    print(f"  - MSE损失: {mse_time:.6f}秒")
    print(f"  - Quantile损失: {quantile_time:.6f}秒")
    print(f"  - MMD损失: {mmd_time:.6f}秒")
    
    # 打印测试阶段损失计算统计
    loss_stats = combined_loss.get_loss_stats()
    print("\n测试阶段损失计算统计:")
    print(f"  - MSE损失: 计算次数={loss_stats['degree_loss_count']}, 总时间={loss_stats['degree_loss_time']:.4f}秒, 平均时间={loss_stats['avg_degree_loss_time']:.6f}秒")
    print(f"  - Quantile损失: 计算次数={loss_stats['quantile_loss_count']}, 总时间={loss_stats['quantile_loss_time']:.4f}秒, 平均时间={loss_stats['avg_quantile_loss_time']:.6f}秒")
    print(f"  - 直方图匹配损失: 计算次数={loss_stats['histogram_loss_count']}, 总时间={loss_stats['histogram_loss_time']:.4f}秒, 平均时间={loss_stats['avg_histogram_loss_time']:.6f}秒")
    
    # 可视化分布比较
    # 将张量转换为numpy数组
    pred_degrees_np = predicted_degrees.cpu().numpy().flatten()
    target_degrees_np = test_targets.cpu().numpy().flatten()
    
    # 绘制直方图
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(pred_degrees_np, bins=20, alpha=0.7, label='Predicted Distribution')
    plt.hist(target_degrees_np, bins=20, alpha=0.7, label='Target Distribution')
    plt.title('Degree Distribution Histogram Comparison')
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 绘制核密度估计
    plt.subplot(1, 2, 2)
    pred_kde = stats.gaussian_kde(pred_degrees_np)
    target_kde = stats.gaussian_kde(target_degrees_np)
    
    x = np.linspace(min(min(pred_degrees_np), min(target_degrees_np)), 
                    max(max(pred_degrees_np), max(target_degrees_np)), 100)
    
    plt.plot(x, pred_kde(x), label='Predicted Distribution')
    plt.plot(x, target_kde(x), label='Target Distribution')
    plt.title('Degree Distribution Kernel Density Estimation')
    plt.xlabel('Degree')
    plt.ylabel('Density')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('degree_distribution_comparison.png')
    print("Distribution comparison plot saved as 'degree_distribution_comparison.png'")
    
    # 打印一些具体的例子
    print("\n具体例子:")
    for i in range(min(5, len(test_targets))):
        print(f"输入序列: {test_inputs[i, :, 0].cpu().numpy()}")
        print(f"目标degree值: {test_targets[i, 0].item():.2f}")
        print(f"预测degree值: {predicted_degrees[i, 0].item():.2f}")
        print(f"误差: {abs(test_targets[i, 0].item() - predicted_degrees[i, 0].item()):.2f}")
        print("---")
    
    # 尝试使用训练好的模型进行自回归生成
    print("\n自回归生成示例:")
    # 从测试集中选择一个序列作为起点
    start_seq = test_inputs[0].to(device)
    generated_seq = start_seq.clone().unsqueeze(0)
    
    # 生成10个新的时间步
    for i in range(10):
        # 使用当前序列预测下一个值
        _, next_degree = model(generated_seq, return_quantile=True)
        
        # 将预测值添加到序列中
        next_degree = next_degree.view(1, 1, 1)
        generated_seq = torch.cat([generated_seq[:, 1:, :], next_degree], dim=1)
        
        print(f"时间步 {i+1}: 预测degree值 = {next_degree.item():.2f}")
    
    print(f"生成序列: {generated_seq[0, :, 0].cpu().numpy()}") 